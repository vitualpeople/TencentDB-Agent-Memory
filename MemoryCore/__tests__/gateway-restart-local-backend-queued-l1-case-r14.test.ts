/**
 * Case R14 of the memory-path failure table: the gateway restarts while an
 * L1 task is queued on the local (in-process) state backend. This is a
 * recording test with no pass/fail target for the behaviour itself; it pins
 * what happens today:
 *
 *   - the queued L1 task, the session's L1_idle timer and its conversation
 *     counter live only in the LocalStateBackend's Maps, so a restarted
 *     backend starts empty;
 *   - the gateway does not restore session counters from the checkpoint:
 *     setStatefulPipelineManager marks the scheduler started, so
 *     StatefulPipelineManager.start(pipeline_states) is never called; and
 *     start() itself only writes counters (no task, no timer);
 *   - after the restart nothing re-runs the lost slice by itself: a fresh
 *     worker idles for 30 min of fake time with zero L1 runs;
 *   - the L1 cursor is on disk (checkpoint file), so the first new message
 *     of that session enqueues an L1 that re-reads the lost rows.
 *
 * Verdict recorded in the phase report: lost until the session gets a new
 * message, then recovered from the checkpoint cursor.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import { TdaiCore } from "../src/core/tdai-core.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner, createStatefulPipelineManager } from "../src/utils/pipeline-factory.js";
import { armedTimers, makeCfg, makeL0Store, msg, sceneJson, scriptedLlm, silent, tmpDir, type L0Row } from "./helpers/l1-spike-harness.js";

const INSTANCE = "svc";
const SESSION = "proj:r14";

function manager(backend: LocalStateBackend) {
  return createStatefulPipelineManager(parseConfig({}), backend, INSTANCE, silent);
}

/** Before the restart: message 1 crosses the warmup threshold (L1 queued), message 2 arms L1_idle. */
async function beforeRestart(): Promise<LocalStateBackend> {
  const backend = new LocalStateBackend({ onTimerExpired: () => {} });
  const m = manager(backend);
  await m.notifyConversation(SESSION, [], INSTANCE, 1, undefined, "default");
  await m.notifyConversation(SESSION, [], INSTANCE, 1, undefined, "default");
  return backend;
}

afterEach(() => { vi.useRealTimers(); });

describe("R14 gateway restart with the local state backend", () => {
  it("R14 before the restart the L1 task is queued, L1_idle is armed and the counter is 1; a restarted backend has none of them", async () => {
    const old = await beforeRestart();
    const queued = await old.listQueuedTasks();
    expect(queued.map((t) => t.type)).toEqual(["L1"]);
    expect(armedTimers(old).has(buildPipelineTimerMember(SESSION, "L1_idle", { agentId: "default" }))).toBe(true);
    expect((await old.getSessionState(INSTANCE, SESSION, undefined, "default"))?.conversation_count).toBe(1);

    const restarted = new LocalStateBackend({ onTimerExpired: () => {} });

    expect(await restarted.listQueuedTasks()).toEqual([]);
    expect(armedTimers(restarted).size).toBe(0);
    expect(await restarted.getSessionState(INSTANCE, SESSION, undefined, "default")).toBeNull();
  });

  it("R14 the gateway path never calls StatefulPipelineManager.start(), so checkpoint pipeline_states are not restored", async () => {
    const dataDir = tmpDir("r14-restore-");
    await new CheckpointManager(dataDir, silent).mergePipelineStates({
      [SESSION]: { conversation_count: 3, last_active_time: Date.now(), l2_pending_l1_count: 0, warmup_threshold: 4, last_extraction_time: "", last_extraction_updated_time: "", l2_last_extraction_time: "" } as never,
    });
    const restarted = new LocalStateBackend();
    const m = manager(restarted);
    const start = vi.spyOn(m, "start");
    const core = Object.create(TdaiCore.prototype) as TdaiCore;
    Object.assign(core, { logger: silent, dataDir });

    core.setStatefulPipelineManager(m);
    await (core as unknown as { ensureSchedulerStarted(): Promise<void> }).ensureSchedulerStarted();

    expect(start).not.toHaveBeenCalled();
    expect(await restarted.getSessionState(INSTANCE, SESSION)).toBeNull();
  });

  it("R14 even a direct start(pipeline_states) only writes the counter: no L1 task is enqueued and no timer is armed", async () => {
    const restarted = new LocalStateBackend({ onTimerExpired: () => {} });
    await manager(restarted).start({
      [SESSION]: { conversation_count: 3, last_active_time: Date.now(), l2_pending_l1_count: 0, warmup_threshold: 4, last_extraction_time: "", last_extraction_updated_time: "", l2_last_extraction_time: "" } as never,
    });

    expect((await restarted.getSessionState(INSTANCE, SESSION))?.conversation_count).toBe(3);
    expect(await restarted.listQueuedTasks()).toEqual([]);
    expect(armedTimers(restarted).size).toBe(0);
  });

  it("R14 after the restart a fresh worker runs no L1 in 30 min when the session gets no new message", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const restarted = new LocalStateBackend({ onTimerExpired: () => {} });
    const executeL1 = vi.fn(async () => {});
    const worker = new PipelineWorker(restarted, { executeL1, async executeL2() {}, async executeL3() {} }, { concurrency: 1, pollIntervalMs: 200 }, silent);
    await worker.start();

    for (let i = 0; i < 30; i++) await vi.advanceTimersByTimeAsync(60_000);

    expect(executeL1).not.toHaveBeenCalled();
    expect(worker.getMetrics().tasksConsumed).toBe(0);
    await worker.stop();
  });

  it("R14 the L1 cursor is on disk: the first new message enqueues L1, which re-reads the rows the lost task was for", async () => {
    const dataDir = tmpDir("r14-recover-");
    await new CheckpointManager(dataDir, silent).markL1ExtractionComplete(SESSION, 0, 500);
    const rows: L0Row[] = [
      { id: "L0-lost1", sessionId: "sess-r14", content: msg("lost one"), timestamp: 10, recordedAtMs: 601 },
      { id: "L0-lost2", sessionId: "sess-r14", content: msg("lost two"), timestamp: 20, recordedAtMs: 602 },
    ];
    const restarted = new LocalStateBackend({ onTimerExpired: () => {} });

    // The new message lands in L0, then notifies the pipeline (as conversation/add does).
    rows.push({ id: "L0-new", sessionId: "sess-r14", content: msg("new one"), timestamp: 30, recordedAtMs: 603 });
    await manager(restarted).notifyConversation(SESSION, [], INSTANCE, 1, undefined, "default");
    const queued = await restarted.listQueuedTasks();
    expect(queued.map((t) => t.type)).toEqual(["L1"]);

    const { store } = makeL0Store(rows);
    const prompts: string[] = [];
    const { runner } = scriptedLlm((prompt) => { prompts.push(prompt); return sceneJson("r14", ["L0-lost1", "L0-lost2", "L0-new"]); });
    const run = createL1Runner({ pluginDataDir: dataDir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });
    await run({ sessionKey: queued[0].sessionId });

    expect(prompts).toHaveLength(1);
    for (const id of ["L0-lost1", "L0-lost2", "L0-new"]) expect(prompts[0]).toContain(id);
    expect((await new CheckpointManager(dataDir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(603);
  });
});
