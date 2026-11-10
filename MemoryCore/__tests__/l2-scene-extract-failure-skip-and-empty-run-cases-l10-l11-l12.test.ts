/**
 * Records the L2 scene-extraction failure and skip paths as they behave today
 * (cases L10, L11 and L12 of the memory-path failure table). The real
 * SceneExtractor runs; only the LLM is a fake.
 *
 *   L10 the scene LLM fails: scene_blocks/ is wiped and restored from the
 *       pre-run backup, extract() returns success:false, and the L2 runner
 *       `continue`s to return undefined (nothing thrown).
 *   L11 that undefined with creditUsed 0 becomes skipped=true in
 *       TdaiCore.runL2WithStore; the gateway executor then marks the task
 *       _l2Skipped, so the worker completes it with no L3, no
 *       l2_last_extraction_time and no retry.
 *   L12 an L2 run that leaves the scene index empty before and after is an
 *       "empty run" -> {skipped:true}; with an existing scene and no file
 *       change the run is NOT treated as empty.
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SceneExtractor } from "../src/core/scene/scene-extractor.js";
import { readSceneIndex, writeSceneIndex } from "../src/core/scene/scene-index.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import type { IMemoryStore, L1RecordRow } from "../src/core/store/types.js";
import { TdaiCore } from "../src/core/tdai-core.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL2Runner } from "../src/utils/pipeline-factory.js";
import type { MemoryTdaiConfig } from "../src/config.js";
import type { LLMRunner } from "../src/core/types.js";
import { providerError, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const SESSION = "proj:l2-cases";
const GROUP_DIR = path.join("profiles", "team%3Adefault%7Cagent%3Adefault");
const SCENE = [
  "---",
  "scene: coding",
  "created: 2026-11-01T09:00:00.000Z",
  "updated: 2026-11-01T10:00:00.000Z",
  "heat: 1",
  "---",
  "",
  "## Summary",
  "Original coding scene.",
  "",
].join("\n");

const cfg = {
  llm: { enabled: false },
  persona: { model: "fake-model", promptMode: undefined, maxScenes: 10, sceneBackupCount: 3 },
} as unknown as MemoryTdaiConfig;

function l1Row(id: string, updated: string): L1RecordRow {
  return {
    record_id: id, content: `memory ${id}`, type: "episodic", priority: 50, scene_name: "",
    session_key: SESSION, session_id: SESSION, team_id: "", task_id: "", user_id: "", agent_id: "",
    version: 1, timestamp_str: updated, timestamp_start: updated, timestamp_end: updated,
    created_time: updated, updated_time: updated, metadata_json: "{}",
  };
}

function fakeStore(): IMemoryStore {
  return {
    isDegraded: () => false,
    queryL1Records: async () => [l1Row("r1", "2026-11-02T10:00:00.000Z")],
  } as unknown as IMemoryStore;
}

/** LLM that scribbles over scene_blocks/ (when given a workspace) and then fails. */
function scribbleThenFail(): { runner: LLMRunner; run: ReturnType<typeof vi.fn> } {
  const run = vi.fn(async (p: { workspaceDir?: string }) => {
    if (p.workspaceDir) {
      fs.writeFileSync(path.join(p.workspaceDir, "coding.md"), "half-written garbage", "utf-8");
      fs.writeFileSync(path.join(p.workspaceDir, "new-scene.md"), "partial new scene", "utf-8");
    }
    throw providerError(502);
  });
  return { runner: { run } as unknown as LLMRunner, run };
}

/** LLM that answers but writes no file. */
function writesNothing(): { runner: LLMRunner; run: ReturnType<typeof vi.fn> } {
  const run = vi.fn(async () => "done, nothing to change");
  return { runner: { run } as unknown as LLMRunner, run };
}

async function seedScene(dataDir: string): Promise<void> {
  fs.mkdirSync(path.join(dataDir, "scene_blocks"), { recursive: true });
  fs.writeFileSync(path.join(dataDir, "scene_blocks", "coding.md"), SCENE, "utf-8");
  await writeSceneIndex(dataDir, [{ filename: "coding.md", summary: "Original coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" }]);
}

describe("L10 L2 scene LLM failure", () => {
  it("L10 extract(): scene_blocks/ is restored from the pre-run backup (scribbles and new files gone), result success:false", async () => {
    const dataDir = tmpDir("l2-l10-extract-");
    await seedScene(dataDir);
    const { runner } = scribbleThenFail();
    const extractor = new SceneExtractor({ dataDir, config: {}, llmRunner: runner, logger: silent, sceneBackupCount: 3 });

    const res = await extractor.extract([{ content: "a new memory", created_at: "2026-11-02T10:00:00.000Z", id: "r1" }]);

    expect(res).toMatchObject({ success: false, memoriesProcessed: 0 });
    expect(res.error).toContain("HTTP 502");
    expect(fs.readFileSync(path.join(dataDir, "scene_blocks", "coding.md"), "utf-8")).toBe(SCENE);
    expect(fs.existsSync(path.join(dataDir, "scene_blocks", "new-scene.md"))).toBe(false);
  });

  it("L10 extract(): the scene LLM call uses a 300 s timeout and the scene_blocks/ workspace", async () => {
    const dataDir = tmpDir("l2-l10-timeout-");
    const { runner, run } = scribbleThenFail();
    await new SceneExtractor({ dataDir, config: {}, llmRunner: runner, logger: silent }).extract([{ content: "m", created_at: "2026-11-02T10:00:00.000Z" }]);

    expect(run.mock.calls[0][0]).toMatchObject({ timeoutMs: 300_000, workspaceDir: path.join(dataDir, "scene_blocks") });
  });

  it("L10 the L2 runner swallows the failed group: it returns undefined and the scoped checkpoint does not count a scene", async () => {
    const dataDir = tmpDir("l2-l10-runner-");
    const { runner, run } = scribbleThenFail();
    const l2 = createL2Runner({ pluginDataDir: dataDir, cfg, openclawConfig: undefined, vectorStore: fakeStore(), logger: silent, llmRunner: runner });

    await expect(l2(SESSION)).resolves.toBeUndefined();

    expect(run).toHaveBeenCalledTimes(1);
    expect((await new CheckpointManager(path.join(dataDir, GROUP_DIR), silent).read()).scenes_processed).toBe(0);
  });
});

describe("L11 a failed L2 with creditUsed 0 is a skip, not a failure", () => {
  /** TdaiCore with only the fields runL2WithStore reads; its runner factory hands out `runner`. */
  function coreWith(runner: LLMRunner, dataDir: string): TdaiCore {
    const core = Object.create(TdaiCore.prototype) as TdaiCore;
    Object.assign(core, {
      cfg,
      hostAdapter: { hostType: "standalone" },
      runnerFactory: { createRunner: () => runner },
      instanceId: "svc",
      dataDir,
      logger: silent,
    });
    return core;
  }

  it("L11 runL2WithStore turns the failed scene LLM call into {creditUsed: 0, skipped: true}", async () => {
    const dataDir = tmpDir("l2-l11-core-");
    const { runner } = scribbleThenFail();

    const res = await coreWith(runner, dataDir).runL2WithStore(SESSION, fakeStore());

    expect(res).toMatchObject({ creditUsed: 0, skipped: true });
  });

  it("L11 through the gateway executor and the worker: task completes, no retry, no L3, no l2_last_extraction_time, no L2 timer", async () => {
    const dataDir = tmpDir("l2-l11-worker-");
    const { runner, run } = scribbleThenFail();
    const backend = new LocalStateBackend();
    const gw = Object.create(TdaiGateway.prototype) as Record<string, unknown>;
    Object.assign(gw, {
      config: { deployMode: "standalone", instanceId: "svc", stateBackend: "local", data: { baseDir: dataDir }, redis: {} },
      logger: silent,
      core: coreWith(runner, dataDir),
      configProvider: {},
      storePool: { mode: "sqlite", getStore: async () => ({ store: fakeStore(), embedding: undefined }) },
      stateBackend: backend,
    });
    const executor = (gw as { buildTaskExecutor(): never }).buildTaskExecutor();
    const onL2Complete = vi.fn(async () => {});
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000, onL2Complete }, silent);
    await worker.start();
    try {
      await backend.enqueueTask({ id: "L2-l11", type: "L2", instanceId: "svc", sessionId: SESSION, agentId: "default", priority: 1, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } });

      await vi.waitFor(() => { expect(worker.getMetrics().tasksCompleted).toBe(1); }, { timeout: 10_000, interval: 10 });

      expect(run).toHaveBeenCalledTimes(1);
      const m = worker.getMetrics();
      expect(m).toMatchObject({ tasksFailed: 0, tasksRetried: 0, tasksDeadLettered: 0 });
      expect(await backend.listQueuedTasks()).toEqual([]);
      expect((await backend.getSessionState("svc", SESSION, undefined, "default"))?.l2_last_extraction_time ?? "").toBe("");
      expect(onL2Complete).not.toHaveBeenCalled();
    } finally {
      await worker.stop();
    }
  });
});

describe("L12 L2 run that changes no file", () => {
  it("L12 scene index empty before and after -> runner returns {skipped: true} and scenes_processed stays 0", async () => {
    const dataDir = tmpDir("l2-l12-empty-");
    const { runner, run } = writesNothing();
    const l2 = createL2Runner({ pluginDataDir: dataDir, cfg, openclawConfig: undefined, vectorStore: fakeStore(), logger: silent, llmRunner: runner });

    await expect(l2(SESSION)).resolves.toEqual({ skipped: true });

    expect(run).toHaveBeenCalledTimes(1);
    expect((await new CheckpointManager(path.join(dataDir, GROUP_DIR), silent).read()).scenes_processed).toBe(0);
  });

  it("L12 with an existing scene and no file change the run is not \"empty\": it returns a latestCursor and counts a scene", async () => {
    const dataDir = tmpDir("l2-l12-existing-");
    await seedScene(path.join(dataDir, GROUP_DIR));
    const { runner } = writesNothing();
    const l2 = createL2Runner({ pluginDataDir: dataDir, cfg, openclawConfig: undefined, vectorStore: fakeStore(), logger: silent, llmRunner: runner });

    const res = await l2(SESSION);

    expect(res).toMatchObject({ latestCursor: "2026-11-02T10:00:00.000Z" });
    expect((await new CheckpointManager(path.join(dataDir, GROUP_DIR), silent).read()).scenes_processed).toBe(1);
    expect((await readSceneIndex(path.join(dataDir, GROUP_DIR))).map((e) => e.filename)).toEqual(["coding.md"]);
  });
});
