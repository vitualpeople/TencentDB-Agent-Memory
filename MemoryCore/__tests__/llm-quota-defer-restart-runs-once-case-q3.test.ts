/**
 * Q3: a container restart while an L1 task is deferred on an LLM quota.
 *
 * The local state backend loses its queue and timers on restart, so the
 * deferral lives in a FileQuotaDeferralStore (the gateway puts it under
 * <data>/metadata/pipeline-quota-deferrals.json). A fresh worker on a fresh
 * backend re-arms it from that file and the task runs again exactly once:
 *
 *   - restart before the delay is over: it runs once, after the delay;
 *   - restart after the re-run was enqueued but before it finished (the old
 *     process died mid-run): it runs once on the new process, and never
 *     again once settled (the file is empty).
 */
import fs from "node:fs";
import path from "node:path";
import { APICallError } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { FileQuotaDeferralStore } from "../src/services/quota-deferral-store.js";
import { advance, advanceUntil, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const INSTANCE = "svc";
const SESSION = "proj:restart";
const AGENT = "default";
const DEFER_MS = 15 * 60_000;

function quota429(): APICallError {
  return new APICallError({ message: "Too Many Requests", url: "https://api.minimax.io/v1/chat/completions", requestBodyValues: {}, statusCode: 429 });
}

function l1Task(): TaskPayload {
  return { id: "L1-orig", type: "L1", instanceId: INSTANCE, sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: AGENT } };
}

function newWorker(backend: LocalStateBackend, executor: TaskExecutor, file: string): PipelineWorker {
  return new PipelineWorker(backend, executor, {
    concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS, quotaDeferralStore: new FileQuotaDeferralStore(file),
  } as never, silent);
}

function storedEntries(file: string): Array<{ id: string; task: TaskPayload }> {
  return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, "utf-8")) as { entries: Array<{ id: string; task: TaskPayload }> }).entries : [];
}

const quotaDeferred = (w: PipelineWorker) => (w.getMetrics() as unknown as Record<string, number>).quotaDeferred;

afterEach(() => { vi.useRealTimers(); });

describe("Q3 restart while a quota-deferred task waits", () => {
  it("Q3 restart before the delay: the new process re-arms the task from disk and runs it exactly once", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const file = path.join(tmpDir("quota-restart-"), "metadata", "pipeline-quota-deferrals.json");

    const before: TaskPayload[] = [];
    const backend1 = new LocalStateBackend();
    const worker1 = newWorker(backend1, {
      executeL1: async (t) => { before.push(t); throw quota429(); }, executeL2: async () => {}, executeL3: async () => {},
    }, file);
    await worker1.start();
    await backend1.enqueueTask(l1Task());
    expect(await advanceUntil(() => quotaDeferred(worker1) === 1, 30_000)).toBe(true);
    expect(storedEntries(file)).toHaveLength(1);
    await advance(5 * 60_000, 30_000);
    await worker1.stop();
    await backend1.destroy();

    // Restart: empty in-memory backend, same data dir.
    const after: TaskPayload[] = [];
    const backend2 = new LocalStateBackend();
    const worker2 = newWorker(backend2, {
      executeL1: async (t) => { after.push({ ...t }); }, executeL2: async () => {}, executeL3: async () => {},
    }, file);
    await worker2.start();
    expect(worker2.getMetrics()).toMatchObject({ quotaDeferredPending: 1 });

    // Still inside the original 15 min: nothing yet.
    await advance(DEFER_MS - 5 * 60_000 - 60_000, 30_000);
    expect(after).toHaveLength(0);

    expect(await advanceUntil(() => worker2.getMetrics().tasksCompleted === 1, 120_000)).toBe(true);
    await advance(60 * 60_000, 60_000);
    expect(before).toHaveLength(1);
    expect(after).toHaveLength(1);
    expect(after[0].sessionId).toBe(SESSION);
    expect(after[0].data?.retryCount ?? 0).toBe(0);
    expect(storedEntries(file)).toEqual([]);
    expect(worker2.getMetrics()).toMatchObject({ quotaDeferredPending: 0, tasksDeadLettered: 0 });
    await worker2.stop();
  });

  it("Q3 restart while the re-run was in flight: it runs once on the new process and is then forgotten", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const file = path.join(tmpDir("quota-restart-inflight-"), "metadata", "pipeline-quota-deferrals.json");

    let calls1 = 0;
    const backend1 = new LocalStateBackend();
    const worker1 = newWorker(backend1, {
      executeL1: async () => {
        if (++calls1 === 1) throw quota429();
        return new Promise<void>(() => {}); // the process dies during the re-run
      },
      executeL2: async () => {}, executeL3: async () => {},
    }, file);
    await worker1.start();
    await backend1.enqueueTask(l1Task());
    expect(await advanceUntil(() => calls1 === 2, DEFER_MS + 60_000, 30_000)).toBe(true);
    expect(storedEntries(file)).toHaveLength(1);
    await worker1.stop();
    await backend1.destroy();

    let calls2 = 0;
    const backend2 = new LocalStateBackend();
    const worker2 = newWorker(backend2, {
      executeL1: async () => { calls2++; }, executeL2: async () => {}, executeL3: async () => {},
    }, file);
    await worker2.start();
    expect(await advanceUntil(() => worker2.getMetrics().tasksCompleted === 1, 60_000)).toBe(true);
    await advance(60 * 60_000, 60_000);
    expect(calls2).toBe(1);
    expect(storedEntries(file)).toEqual([]);
    await worker2.stop();
  });
});
