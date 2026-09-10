import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { parsePipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload, TimerEntry } from "../src/core/state/types.js";
import { PipelineWorker, claimTimerL1Count, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { StatefulPipelineManager } from "../src/utils/stateful-pipeline-manager.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

/**
 * Mirrors the LocalStateBackend `onTimerExpired` wiring in gateway/server.ts:
 * parse the pipeline member and enqueue a timer_scanner task carrying the
 * parsed (teamId, agentId).
 */
function enqueueTimerTask(backend: LocalStateBackend, entry: TimerEntry): void {
  const parsed = parsePipelineTimerMember(entry.member);
  const instanceId = entry.instanceId ?? "default";
  const now = Date.now();
  void backend.enqueueTask({
    id: `${parsed.taskType}-${parsed.sessionId}-${now}`,
    type: parsed.taskType,
    instanceId,
    sessionId: parsed.sessionId,
    teamId: parsed.teamId,
    agentId: parsed.agentId,
    priority: 0,
    createdAt: now,
    data: { triggeredBy: "timer_scanner", timerMember: entry.member, instanceId, teamId: parsed.teamId, agentId: parsed.agentId },
  });
}

describe("timer-fired L1 must not be deduplicated against a count wiped by a completing L1", () => {
  let worker: PipelineWorker | undefined;
  afterEach(async () => { await worker?.stop(); worker = undefined; });

  // Both pairs seen on the running gateway: the hub without team_id
  // (resolveIsolation gives teamId undefined) and the hub sending
  // x-tdai-team-id: default.
  it.each([
    ["teamId undefined, agentId default", undefined, "default"],
    ["teamId default, agentId default", "default", "default"],
  ])("A triggers threshold, B arrives while A runs, A completes, B timer fires -> L1 runs (%s)", async (_label, teamId, agentId) => {
    const instanceId = "svc";
    const session = "proj:obsidian-second-brain";
    let backend!: LocalStateBackend;
    backend = new LocalStateBackend({ onTimerExpired: (entry) => enqueueTimerTask(backend, entry) });

    const manager = new StatefulPipelineManager(
      { everyNConversations: 8, enableWarmup: true, l1: { idleTimeoutSeconds: 0.05 },
        l2: { delayAfterL1Seconds: 600, minIntervalSeconds: 600, maxIntervalSeconds: 3600, sessionActiveWindowHours: 24 } },
      backend, instanceId, silent,
    );

    const runs: TaskPayload[] = [];
    const skips: TaskPayload[] = [];
    let releaseA!: () => void;
    const aReleased = new Promise<void>((r) => { releaseA = r; });
    let aStarted!: () => void;
    const aStartedP = new Promise<void>((r) => { aStarted = r; });

    // Same shape as executeL1 in gateway/server.ts: timer dedup, then "extract".
    const executor: TaskExecutor = {
      async executeL1(task) {
        const inst = task.data?.instanceId as string;
        const tid = task.teamId ?? (task.data?.teamId as string | undefined);
        const aid = task.agentId ?? (task.data?.agentId as string | undefined);
        if (task.data?.triggeredBy === "timer_scanner") {
          if (await claimTimerL1Count(backend, task, inst, tid, aid)) { skips.push(task); return; }
        }
        runs.push(task);
        if (runs.length === 1) { aStarted(); await aReleased; }
      },
      async executeL2() {},
      async executeL3() {},
    };
    worker = new PipelineWorker(backend, executor, { concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000 }, silent);
    await worker.start();

    // Message A: warmup threshold is 1, so the L1 task is enqueued immediately.
    await manager.notifyConversation(session, [], instanceId, 1, teamId, agentId);
    await aStartedP;
    expect(runs[0].data?.triggeredBy).toBeUndefined();

    // Message B: written while A's L1 is running. count=1/2, idle timer armed.
    await manager.notifyConversation(session, [], instanceId, 1, teamId, agentId);
    expect((await backend.getSessionState(instanceId, session, teamId, agentId))?.conversation_count).toBe(1);

    // A completes -> cascadeSchedule. B's count must survive.
    releaseA();
    await vi.waitFor(() => {
      expect(worker!.getMetrics().tasksCompleted).toBe(1);
    }, { timeout: 2_000, interval: 5 });
    expect((await backend.getSessionState(instanceId, session, teamId, agentId))?.conversation_count).toBe(1);

    // B's idle timer fires -> the timer-fired L1 must run, not be skipped.
    await vi.waitFor(() => {
      expect(runs.length + skips.length).toBe(2);
    }, { timeout: 2_000, interval: 5 });
    expect(skips).toHaveLength(0);
    expect(runs[1].data?.triggeredBy).toBe("timer_scanner");
    expect(runs[1].teamId).toBe(teamId);
    expect(runs[1].agentId).toBe(agentId);

    // The timer-fired run consumed B's round: the count is released to 0.
    await vi.waitFor(() => {
      expect(worker!.getMetrics().tasksCompleted).toBe(2);
    }, { timeout: 2_000, interval: 5 });
    expect((await backend.getSessionState(instanceId, session, teamId, agentId))?.conversation_count).toBe(0);
  });

  it("a skipped timer-fired L1 leaves the session state untouched and does not advance L2", async () => {
    const instanceId = "svc";
    const session = "s";
    const backend = new LocalStateBackend();
    await backend.updateSessionState(instanceId, session, { conversation_count: 0, warmup_threshold: 4 }, undefined, "default");

    let ran = 0;
    const executor: TaskExecutor = {
      async executeL1(task) {
        if (await claimTimerL1Count(backend, task, instanceId, undefined, "default")) return;
        ran++;
      },
      async executeL2() {},
      async executeL3() {},
    };
    let l2Advanced = 0;
    worker = new PipelineWorker(backend, executor, {
      concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000,
      onL1Complete: async () => { l2Advanced++; },
    }, silent);
    await worker.start();

    await backend.enqueueTask({
      id: "L1-timer", type: "L1", instanceId, sessionId: session, agentId: "default", priority: 0, createdAt: Date.now(),
      data: { triggeredBy: "timer_scanner", instanceId, agentId: "default" },
    });
    await vi.waitFor(() => expect(worker!.getMetrics().tasksCompleted).toBe(1), { timeout: 2_000, interval: 5 });

    expect(ran).toBe(0);
    expect(l2Advanced).toBe(0);
    expect(await backend.getSessionState(instanceId, session, undefined, "default")).toMatchObject({ conversation_count: 0, warmup_threshold: 4 });
  });
});
