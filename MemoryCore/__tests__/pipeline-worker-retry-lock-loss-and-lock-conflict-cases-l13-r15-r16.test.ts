/**
 * Records the PipelineWorker retry and lock paths on the local (in-process)
 * state backend, as they behave today (cases L13, R15 and R16 of the
 * memory-path failure table). The executor is a fake; the worker and the
 * LocalStateBackend are real; time is a fake clock.
 *
 *   L13 an L2 task that throws is retried 3 times after 5 s, 15 s and 45 s,
 *       then dead-lettered in process; the session's L1_idle / L2_schedule
 *       timers are removed and (unlike L1) no L1_dlq re-run is armed.
 *   R15 the lock lapses (TTL ends between renewals and another owner takes
 *       it): the renewal fails, the executor's AbortSignal fires, and the
 *       task is neither retried, acked-as-done nor cascaded.
 *   R16 the lock is held by someone else: back off 200 ms, 600 ms, 1.8 s,
 *       then 5 s steps until lockTtlMs, then requeue as <id>-lr<n>; at
 *       lockRetryCount 15 the task is dropped with "Lock conflict exhausted".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { armedTimers, silent } from "./helpers/l1-spike-harness.js";

/** Advance the fake clock in steps; these tests do no real I/O, so nothing else needs to settle. */
async function advance(ms: number, step = 500): Promise<void> {
  for (let done = 0; done < ms; done += step) await vi.advanceTimersByTimeAsync(Math.min(step, ms - done));
}

const SESSION = "proj:worker-cases";

function task(type: "L1" | "L2", extra: Partial<TaskPayload> = {}, data: Record<string, unknown> = {}): TaskPayload {
  return {
    id: `${type}-1`, type, instanceId: "svc", sessionId: SESSION, agentId: "default", priority: type === "L1" ? 0 : 1,
    createdAt: Date.now(), ...extra, data: { instanceId: "svc", agentId: "default", ...data },
  };
}

function spyLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function lockKeyOf(worker: PipelineWorker, t: TaskPayload): string {
  return (worker as unknown as { getLockKey(t: TaskPayload): string }).getLockKey(t);
}

afterEach(() => { vi.useRealTimers(); });

describe("L13 L2 task that throws", () => {
  it("L13 retries three times after 5 s, 15 s and 45 s, then dead-letters and removes the session timers (no L1_dlq for L2)", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = new LocalStateBackend({ onTimerExpired: () => {} });
    const runsAt: number[] = [];
    const executor: TaskExecutor = {
      async executeL1() {},
      async executeL2() { runsAt.push(Date.now()); throw new TypeError("store.queryL1Records is not a function"); },
      async executeL3() {},
    };
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, pollIntervalMs: 200, lockRenewIntervalMs: 60_000 }, silent);
    const idle = buildPipelineTimerMember(SESSION, "L1_idle", { agentId: "default" });
    const sched = buildPipelineTimerMember(SESSION, "L2_schedule", { agentId: "default" });
    const dlq = buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: "default" });
    await backend.setTimer("svc", idle, Date.now() + 60 * 60_000);
    await backend.setTimer("svc", sched, Date.now() + 60 * 60_000);
    await worker.start();
    await backend.enqueueTask(task("L2"));

    await advance(5_000 + 15_000 + 45_000 + 5_000);

    expect(runsAt).toHaveLength(4);
    const gaps = runsAt.slice(1).map((t, i) => t - runsAt[i]);
    expect(gaps[0]).toBeGreaterThanOrEqual(5_000);
    expect(gaps[0]).toBeLessThan(6_000);
    expect(gaps[1]).toBeGreaterThanOrEqual(15_000);
    expect(gaps[1]).toBeLessThan(16_000);
    expect(gaps[2]).toBeGreaterThanOrEqual(45_000);
    expect(gaps[2]).toBeLessThan(46_000);
    expect(worker.getMetrics()).toMatchObject({ tasksFailed: 4, tasksRetried: 3, tasksDeadLettered: 1, deadLetterCount: 1 });
    expect(worker.getDeadLetterQueue()[0]).toMatchObject({ retryCount: 3, task: { type: "L2", sessionId: SESSION } });
    expect(worker.getDeadLetterQueue()[0].error).toContain("is not a function");
    const timers = armedTimers(backend);
    expect(timers.has(idle)).toBe(false);
    expect(timers.has(sched)).toBe(false);
    expect(timers.has(dlq)).toBe(false);

    await advance(60 * 60_000, 60_000);
    expect(runsAt).toHaveLength(4);
    await worker.stop();
  });
});

describe("R15 lock lost during execution on the local backend", () => {
  /** TTL 1 s, renew every 3 s: the lock lapses before the first renewal and another owner takes it at 1.5 s. */
  async function runWithLapsingLock(executeL1: TaskExecutor["executeL1"]) {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = new LocalStateBackend();
    const onL1Complete = vi.fn(async () => {});
    const logger = spyLogger();
    const worker = new PipelineWorker(backend, { executeL1, async executeL2() {}, async executeL3() {} },
      { concurrency: 1, pollIntervalMs: 50, lockTtlMs: 1_000, lockRenewIntervalMs: 3_000, onL1Complete }, logger);
    const t = task("L1");
    const key = lockKeyOf(worker, t);
    await worker.start();
    await backend.enqueueTask(t);
    await advance(1_500, 100);
    const stolen = await backend.acquireLock(key, "other-worker", 60_000);
    await advance(10_000, 100);
    return { worker, backend, onL1Complete, logger, stolen };
  }

  it("R15 an executor wired to the signal is aborted: tasksFailed 1, no retry, no dead letter, nothing requeued", async () => {
    const signals: AbortSignal[] = [];
    const executeL1 = vi.fn((_t: TaskPayload, signal?: AbortSignal) => new Promise<void>((_resolve, reject) => {
      signals.push(signal!);
      signal!.addEventListener("abort", () => reject(signal!.reason));
    }));

    const { worker, backend, onL1Complete, logger, stolen } = await runWithLapsingLock(executeL1);

    expect(stolen).toBe(true);
    expect(executeL1).toHaveBeenCalledTimes(1);
    expect(signals[0].aborted).toBe(true);
    expect(String(signals[0].reason)).toContain("lock lost during execution");
    expect(worker.getMetrics()).toMatchObject({ lockRenewFailed: 1, executionAborted: 1, tasksFailed: 1, tasksRetried: 0, tasksDeadLettered: 0, tasksCompleted: 0 });
    expect(await backend.listQueuedTasks()).toEqual([]);
    expect(onL1Complete).not.toHaveBeenCalled();
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("Lock lost during execution, aborting"));
    await worker.stop();
  });

  it("R15 an executor that ignores the signal and returns: lockLostDuringExecution 1, not completed, no cascade", async () => {
    const executeL1 = vi.fn(() => new Promise<void>((resolve) => { setTimeout(resolve, 5_000); }));

    const { worker, onL1Complete } = await runWithLapsingLock(executeL1);

    expect(worker.getMetrics()).toMatchObject({ lockRenewFailed: 1, executionAborted: 1, lockLostDuringExecution: 1, tasksCompleted: 0, tasksFailed: 0, tasksRetried: 0 });
    expect(onL1Complete).not.toHaveBeenCalled();
    await worker.stop();
  });
});

describe("R16 lock conflict", () => {
  /** Run processTask once by hand (worker marked running, no consume loop). */
  function manualWorker(executor: TaskExecutor, logger = spyLogger()) {
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockTtlMs: 10_000, lockRenewIntervalMs: 60_000 }, logger);
    (worker as unknown as { running: boolean }).running = true;
    const processTask = (t: TaskPayload) => (worker as unknown as { processTask(t: TaskPayload): Promise<void> }).processTask(t);
    return { backend, worker, processTask, logger };
  }

  it("R16 backs off 200 ms, 600 ms, 1.8 s, then 5 s steps until lockTtlMs, then requeues as <id>-lr1 with lockRetryCount 1", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const executeL1 = vi.fn(async () => {});
    const { backend, worker, processTask } = manualWorker({ executeL1, async executeL2() {}, async executeL3() {} });
    const t = task("L1");
    await backend.acquireLock(lockKeyOf(worker, t), "other-worker", 60 * 60_000);
    const attemptsAt: number[] = [];
    const orig = backend.acquireLock.bind(backend);
    vi.spyOn(backend, "acquireLock").mockImplementation(async (...args) => { attemptsAt.push(Date.now()); return orig(...args); });

    const done = processTask(t);
    await vi.advanceTimersByTimeAsync(20_000);
    await done;

    const gaps = attemptsAt.slice(1).map((x, i) => x - attemptsAt[i]);
    expect(gaps).toEqual([200, 600, 1_800, 5_000, 5_000]);
    expect(executeL1).not.toHaveBeenCalled();
    expect(worker.getMetrics().lockConflicts).toBe(1);
    const queued = await backend.listQueuedTasks();
    expect(queued.map((q) => q.id)).toEqual(["L1-1-lr1"]);
    expect(queued[0].data?.lockRetryCount).toBe(1);
  });

  it("R16 the lock freed during the backoff: the task runs on the next attempt and is not requeued", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const executeL1 = vi.fn(async () => {});
    const { backend, worker, processTask } = manualWorker({ executeL1, async executeL2() {}, async executeL3() {} });
    const t = task("L1");
    const key = lockKeyOf(worker, t);
    await backend.acquireLock(key, "other-worker", 60 * 60_000);

    const done = processTask(t);
    await vi.advanceTimersByTimeAsync(1_000);
    await backend.releaseLock(key, "other-worker");
    await vi.advanceTimersByTimeAsync(5_000);
    await done;

    expect(executeL1).toHaveBeenCalledTimes(1);
    expect(worker.getMetrics()).toMatchObject({ lockConflicts: 1, tasksCompleted: 1 });
    expect(await backend.listQueuedTasks()).toEqual([]);
  });

  it("R16 at lockRetryCount 15 the task is dropped with \"Lock conflict exhausted\" and nothing is requeued", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const executeL1 = vi.fn(async () => {});
    const logger = spyLogger();
    const { backend, worker, processTask } = manualWorker({ executeL1, async executeL2() {}, async executeL3() {} }, logger);
    const t = task("L1", { id: "L1-1-lr15" }, { lockRetryCount: 15 });
    await backend.acquireLock(lockKeyOf(worker, t), "other-worker", 60 * 60_000);

    const done = processTask(t);
    await vi.advanceTimersByTimeAsync(20_000);
    await done;

    expect(executeL1).not.toHaveBeenCalled();
    expect(await backend.listQueuedTasks()).toEqual([]);
    expect(logger.error).toHaveBeenCalledWith(expect.stringContaining("Lock conflict exhausted [L1] (task=L1-1-lr15)"));
    expect(logger.error.mock.calls[0][0]).toContain("dropping after 15 requeues");
    expect(worker.getMetrics().tasksDeadLettered).toBe(0);
  });
});
