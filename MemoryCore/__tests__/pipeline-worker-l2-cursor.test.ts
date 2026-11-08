/**
 * After a successful L2 task the worker stores l2_last_extraction_time in the
 * session state; the gateway reads it back as the next L2 `updatedAfter`
 * cursor. The executor here drives the real createL2Runner (extractor faked,
 * no model call), so an L2 run that throws leaves the cursor unset.
 */
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/core/scene/scene-extractor.js", () => ({
  SceneExtractor: class {
    async extract(memories: unknown[]) {
      return { success: true, memoriesProcessed: memories.length, emptyExtraction: false };
    }
  },
}));

import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { createL2Runner } from "../src/utils/pipeline-factory.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { LocalStorageBackend } from "../src/core/storage/local-backend.js";
import type { IMemoryStore } from "../src/core/store/types.js";
import type { MemoryTdaiConfig } from "../src/config.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

describe("pipeline worker records the L2 cursor after a successful L2 run", () => {
  let dataDir: string;
  let worker: PipelineWorker | undefined;

  beforeEach(async () => {
    dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "l2-cursor-"));
  });

  afterEach(async () => {
    await worker?.stop();
    worker = undefined;
    await fs.rm(dataDir, { recursive: true, force: true });
  });

  it("writes l2_last_extraction_time and clears l2_pending_l1_count", async () => {
    const instanceId = "svc";
    const session = "proj:l2-cursor-test";
    const updated = "2026-11-01T10:00:00.000Z";
    const store = {
      isDegraded: () => false,
      queryL1Records: async () => [{
        record_id: "r1", content: "memory r1", type: "episodic", priority: 50, scene_name: "",
        session_key: session, session_id: session, team_id: "", task_id: "", user_id: "", agent_id: "",
        version: 1, timestamp_str: updated, timestamp_start: updated, timestamp_end: updated,
        created_time: updated, updated_time: updated, metadata_json: "{}",
      }],
    } as unknown as IMemoryStore;
    const l2Runner = createL2Runner({
      pluginDataDir: dataDir,
      cfg: { persona: { model: "fake-model", maxScenes: 10, sceneBackupCount: 0 } } as unknown as MemoryTdaiConfig,
      openclawConfig: {},
      vectorStore: store,
      logger: silent,
      storage: new StorageAdapter(new LocalStorageBackend(dataDir)),
    });

    const backend = new LocalStateBackend();
    await backend.updateSessionState(instanceId, session, { l2_pending_l1_count: 1 }, undefined, "default");
    const l2Errors: unknown[] = [];
    const executor: TaskExecutor = {
      async executeL1() {},
      async executeL2(task) {
        try {
          await l2Runner(task.sessionId);
        } catch (err) {
          l2Errors.push(err);
          throw err;
        }
      },
      async executeL3() {},
    };
    worker = new PipelineWorker(backend, executor, { concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000 }, silent);
    await worker.start();

    const before = Date.now();
    await backend.enqueueTask({
      id: "L2-cursor", type: "L2", instanceId, sessionId: session, agentId: "default", priority: 1, createdAt: before,
      data: { instanceId, agentId: "default" },
    });

    // A completed L2 also enqueues L3, so wait on the session state itself.
    await vi.waitFor(async () => {
      expect(l2Errors).toEqual([]);
      const s = await backend.getSessionState(instanceId, session, undefined, "default");
      expect(s?.l2_last_extraction_time).toBeTruthy();
    }, { timeout: 5_000, interval: 10 });

    const state = await backend.getSessionState(instanceId, session, undefined, "default");
    expect(state?.l2_pending_l1_count).toBe(0);
    expect(typeof state?.l2_last_extraction_time).toBe("string");
    expect(Date.parse(state!.l2_last_extraction_time as string)).toBeGreaterThanOrEqual(before - 1_000);
  });
});
