import { describe, expect, it } from "vitest";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember, parsePipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload } from "../src/core/state/types.js";

// resolveIsolation() defaults agentId to "default" and leaves teamId unset when a
// request carries neither header, so (undefined, "default") is the pair every
// client without team_id produces. The timer member must carry that half-set
// scope, otherwise the timer-fired task reads a different session-state key
// than the one captureAtomic counted under.
describe("buildPipelineTimerMember / parsePipelineTimerMember scope round-trip", () => {
  const session = "proj:obsidian-second-brain";

  it.each([
    ["agentId only", { teamId: undefined, agentId: "default" }],
    ["teamId only", { teamId: "t", agentId: undefined }],
    ["both", { teamId: "t", agentId: "a" }],
  ])("round-trips %s and never yields the string \"undefined\"", (_label, ctx) => {
    const member = buildPipelineTimerMember(session, "L1_idle", ctx);
    expect(member.startsWith("scope:")).toBe(true);
    expect(member).not.toContain("undefined");

    const parsed = parsePipelineTimerMember(member);
    expect(parsed.sessionId).toBe(session);
    expect(parsed.timerType).toBe("L1_idle");
    expect(parsed.taskType).toBe("L1");
    expect(parsed.teamId).toBe(ctx.teamId);
    expect(parsed.agentId).toBe(ctx.agentId);
  });

  it("keeps the legacy unscoped member when neither id is set", () => {
    const member = buildPipelineTimerMember(session, "L1_idle", {});
    expect(member).toBe(`${session}:L1_idle`);
    const parsed = parsePipelineTimerMember(member);
    expect(parsed).toMatchObject({ sessionId: session, timerType: "L1_idle", taskType: "L1" });
    expect(parsed.teamId).toBeUndefined();
    expect(parsed.agentId).toBeUndefined();
  });

  it("makes the timer-fired task read the state captureAtomic counted under (agentId only)", async () => {
    const backend = new LocalStateBackend();
    const instanceId = "svc";
    const teamId = undefined;
    const agentId = "default";
    const member = buildPipelineTimerMember(session, "L1_idle", { teamId, agentId });
    const taskPayload: TaskPayload = {
      id: "L1-test", type: "L1", instanceId, sessionId: session, teamId, agentId, priority: 0, createdAt: Date.now(),
    };

    // Writer side: one round below the threshold, so the count stays at 1.
    const result = await backend.captureAtomic({
      instanceId, sessionId: session, teamId, agentId,
      threshold: 4, fireAtMs: Date.now() + 60_000, timerMember: member, taskPayload, nowMs: Date.now(), rounds: 1,
    });
    expect(result).toEqual({ triggered: false, conversationCount: 1 });

    // Timer side: the executor dedup and cascadeSchedule derive their key from
    // the parsed member. It must land on the same entry.
    const parsed = parsePipelineTimerMember(member);
    const seenByTimerTask = await backend.getSessionState(instanceId, parsed.sessionId, parsed.teamId, parsed.agentId);
    expect(seenByTimerTask).not.toBeNull();
    expect(seenByTimerTask!.conversation_count).toBe(1);
  });
});
