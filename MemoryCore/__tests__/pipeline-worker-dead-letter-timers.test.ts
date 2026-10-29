import { describe, expect, it, vi } from "vitest";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };
const noopExecutor: TaskExecutor = { async executeL1() {}, async executeL2() {}, async executeL3() {} };

/**
 * moveToDeadLetter clears the session's L1_idle / L2_schedule timers inside a
 * best-effort try/catch. Without the buildPipelineTimerMember import the call
 * threw a ReferenceError that the catch swallowed, so no timer was ever
 * removed and a dead-lettered session kept firing ghost L1/L2 triggers.
 */
describe("PipelineWorker dead letter clears the session's pipeline timers", () => {
  it.each([
    ["teamId/agentId on the task", { teamId: "team-a", agentId: "agent-a" }, {}],
    ["teamId/agentId only in task.data", {}, { teamId: "team-b", agentId: "agent-b" }],
  ])("removes the scoped L1_idle and L2_schedule members (%s)", async (_label, onTask, inData) => {
    const backend = new LocalStateBackend();
    const removeTimer = vi.spyOn(backend, "removeTimer");
    const worker = new PipelineWorker(backend, noopExecutor, { concurrency: 1 }, silent);

    const task: TaskPayload = {
      id: "L1-sess-1",
      type: "L1",
      instanceId: "svc",
      sessionId: "proj:demo",
      priority: 0,
      createdAt: Date.now(),
      ...onTask,
      data: { ...inData },
    };
    const ctx = {
      teamId: (onTask as { teamId?: string }).teamId ?? (inData as { teamId?: string }).teamId,
      agentId: (onTask as { agentId?: string }).agentId ?? (inData as { agentId?: string }).agentId,
    };

    await (worker as unknown as {
      moveToDeadLetter(t: TaskPayload, e: string, n: number): Promise<void>;
    }).moveToDeadLetter(task, "boom", 3);

    expect(removeTimer).toHaveBeenCalledTimes(2);
    expect(removeTimer).toHaveBeenNthCalledWith(1, "svc", buildPipelineTimerMember("proj:demo", "L1_idle", ctx));
    expect(removeTimer).toHaveBeenNthCalledWith(2, "svc", buildPipelineTimerMember("proj:demo", "L2_schedule", ctx));
    expect(worker.getMetrics().tasksDeadLettered).toBe(1);
  });
});
