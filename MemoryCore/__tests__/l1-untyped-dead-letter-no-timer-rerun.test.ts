/**
 * An L1 dead-letter whose last failure is not an L1ExtractionError never arms
 * the L1_dlq re-run timer.
 *
 * Every LLM provider failure on the L1 path reaches PipelineWorker typed (the
 * extractor wraps the extraction call; dedup and embedding failures are
 * swallowed). What stays untyped is local, and may come after a paid LLM call
 * that succeeded: here the extraction call answers, the memory is written,
 * then the checkpoint write fails (disk full), so the cursor never moves.
 * Re-running that on a 15 min timer would pay for the same extraction every
 * cycle with no end; instead the task runs 1 + maxRetries times and stops
 * until the session gets a new message (the behaviour before L1_dlq).
 *
 * Nothing here touches the real file system: the storage handed to the real
 * createL1Runner is in memory, so fake timers alone drive every step.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

const hoisted = vi.hoisted(() => ({ captured: undefined as undefined | Record<string, never> }));
vi.mock("../src/core/state/index.js", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../src/core/state/index.js")>();
  return {
    ...orig,
    createStateBackend: vi.fn(async (o: Record<string, never>) => {
      hoisted.captured = o;
      throw new Error("stop-after-state-backend");
    }),
  };
});

import { L1ExtractionError } from "../src/core/record/l1-extraction-error.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { StoragePaths } from "../src/core/storage/types.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  armedTimers, gatewayHarness, makeCfg, makeL0Store, msg, sceneJson, scriptedLlm, silent,
} from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";
const AGENT = "default";
const RETRY_DELAY_MS = 15 * 60_000;
/** 5 + 15 + 45 s of worker retries, plus slack. */
const RETRY_WINDOW_MS = 70_000;
const DLQ = buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: AGENT });

/** In-memory storage whose checkpoint writes fail like a full disk. */
function storageWithBrokenCheckpoint() {
  const objects = new Map<string, string>();
  let checkpointWrites = 0;
  const backend = {
    type: "local" as const,
    async putObject(key: string, content: string | Buffer) {
      if (key === StoragePaths.checkpoint) {
        checkpointWrites++;
        throw new Error("ENOSPC: no space left on device, write");
      }
      objects.set(key, content.toString());
    },
    async appendObject(key: string, content: string | Buffer) {
      objects.set(key, (objects.get(key) ?? "") + content.toString());
    },
    async getObject(key: string) {
      const v = objects.get(key);
      return v === undefined ? null : { content: Buffer.from(v) };
    },
    async exists(key: string) { return objects.has(key); },
    async deleteObject(key: string) { objects.delete(key); },
    async listObjects() { return { entries: [] }; },
  };
  return { storage: new StorageAdapter(backend as never), checkpointWrites: () => checkpointWrites };
}

/** Advance fake time in steps until `cond()` holds; no real-time waits. */
async function advanceUntil(cond: () => boolean, maxMs: number, step = 500): Promise<boolean> {
  for (let done = 0; done < maxMs && !cond(); done += step) {
    await vi.advanceTimersByTimeAsync(step);
  }
  return cond();
}

async function startWorker(runL1: Parameters<typeof gatewayHarness>[0]["runL1"]) {
  const { backend, executor } = await gatewayHarness({
    TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend, instanceId: "svc", baseDir: "unused-in-memory", runL1,
  });
  // Default maxRetries (3) and deadLetterRetryDelayMs (15 min).
  const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockRenewIntervalMs: 60_000 }, silent);
  await worker.start();
  const enqueue = (id: string) => backend.enqueueTask({
    id, type: "L1", instanceId: "svc", sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(), data: { instanceId: "svc", agentId: AGENT },
  });
  return { backend, worker, enqueue };
}

describe("untyped L1 dead-letter -> no L1_dlq re-run", () => {
  afterEach(() => { vi.useRealTimers(); hoisted.captured = undefined; });

  it("LLM answers, checkpoint write fails every run: 4 runner runs and 4 LLM calls over 4 retry cycles, no L1_dlq", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const { store, l1Records } = makeL0Store([
      { id: "L0-k1", sessionId: "sess-k", agentId: AGENT, content: msg("kilo one"), timestamp: 10, recordedAtMs: 11 },
    ]);
    const { runner, run: llmCalls } = scriptedLlm(() => sceneJson("kilo", ["L0-k1"]));
    const { storage, checkpointWrites } = storageWithBrokenCheckpoint();
    const real = createL1Runner({
      pluginDataDir: "unused-in-memory", cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner, storage,
    });
    let runs = 0;
    const { backend, worker, enqueue } = await startWorker(async (sessionKey) => { runs++; return real({ sessionKey }); });
    await enqueue("L1-orig");

    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 1, RETRY_WINDOW_MS)).toBe(true);
    expect(runs).toBe(4);
    expect(llmCalls).toHaveBeenCalledTimes(4);
    expect(checkpointWrites()).toBe(4);
    expect(l1Records).toHaveLength(4); // the write before the failing checkpoint landed each time
    const entry = worker.getDeadLetterQueue()[0];
    expect(entry.errorKind).toBeUndefined();
    expect(entry.error).toContain("ENOSPC");

    // Run counts first, so a regression reports how many runs it made.
    await vi.advanceTimersByTimeAsync(4 * RETRY_DELAY_MS);
    expect(runs).toBe(4);
    expect(llmCalls).toHaveBeenCalledTimes(4);
    expect(worker.getMetrics().tasksDeadLettered).toBe(1);
    expect(armedTimers(backend).has(DLQ)).toBe(false);
    await worker.stop();
  });

  it("an untyped dead-letter removes the L1_dlq a typed round armed, so no timer re-runs the slice after it", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    let runs = 0;
    const { backend, worker, enqueue } = await startWorker(async () => {
      runs++;
      if (runs <= 4) throw new L1ExtractionError("provider", "LLM API call failed: HTTP 503");
      throw new Error("ENOSPC: no space left on device, write");
    });

    await enqueue("L1-orig");
    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 1, RETRY_WINDOW_MS)).toBe(true);
    expect(worker.getDeadLetterQueue()[0].errorKind).toBe("provider");
    expect(armedTimers(backend).get(DLQ)).toBe(worker.getDeadLetterQueue()[0].deadAt + RETRY_DELAY_MS);

    // A new message triggers a run before L1_dlq fires; it fails untyped.
    await enqueue("L1-new-message");
    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 2, RETRY_WINDOW_MS)).toBe(true);
    expect(runs).toBe(8);
    expect(worker.getDeadLetterQueue()[1].errorKind).toBeUndefined();

    await vi.advanceTimersByTimeAsync(2 * RETRY_DELAY_MS);
    expect(runs).toBe(8);
    expect(worker.getMetrics().tasksDeadLettered).toBe(2);
    expect(armedTimers(backend).has(DLQ)).toBe(false);
    await worker.stop();
  });
});
