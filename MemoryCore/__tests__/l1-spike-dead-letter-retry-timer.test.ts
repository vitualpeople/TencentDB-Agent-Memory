/**
 * Re-run after dead letter: when an L1 task is dead-lettered the worker arms an
 * "L1_dlq" timer; both timer scanners map that member to an L1 task with
 * triggeredBy "dead_letter_retry", which the gateway's executeL1 runs without
 * claimTimerL1Count, so the slice is re-extracted with no new message.
 *
 * The delay is PipelineWorker's deadLetterRetryDelayMs (gateway config
 * worker.deadLetterRetryDelayMs, default 15 min). A successful re-run consumes
 * the live conversation_count, so the next L1_idle is not an empty run.
 *
 * The runner here throws a typed provider L1ExtractionError on its first
 * four calls and then delegates to the real createL1Runner, which isolates
 * this mechanism from how the real runner builds that error. Only a typed
 * dead-letter arms L1_dlq; an untyped one does not (see
 * l1-untyped-dead-letter-no-timer-rerun.test.ts).
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
import type { TaskPayload } from "../src/core/state/types.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { TimerScanner } from "../src/services/timer-scanner.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  advance, advanceUntil, armedTimers, gatewayHarness, makeCfg, makeL0Store, msg, sceneJson, scriptedLlm, silent, tmpDir,
} from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";
const AGENT = "default";
const RETRY_DELAY_MS = 15 * 60_000;

describe("dead letter -> L1_dlq timer -> dead_letter_retry task -> executeL1 runs", () => {
  afterEach(() => { vi.useRealTimers(); hoisted.captured = undefined; });

  it.each([
    ["threshold-fired task (count already 0)", 0, { instanceId: "svc", agentId: AGENT }],
    ["timer-fired task (count 3)", 3, { triggeredBy: "timer_scanner", instanceId: "svc", agentId: AGENT }],
  ])("%s: LLM down for 4 runs, back up, no new message -> slice extracted after the delay", async (_label, count, data) => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const dir = tmpDir("l1-dlq-");
    const { store, l1Records } = makeL0Store([
      { id: "L0-k1", sessionId: "sess-k", agentId: AGENT, content: msg("kilo one"), timestamp: 10, recordedAtMs: 11 },
      { id: "L0-k2", sessionId: "sess-k", agentId: AGENT, content: msg("kilo two"), timestamp: 20, recordedAtMs: 12 },
    ]);
    const { runner } = scriptedLlm(() => sceneJson("kilo", ["L0-k1", "L0-k2"]));
    const real = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });
    let calls = 0;
    const runL1 = vi.fn(async (sessionKey: string) => {
      if (++calls <= 4) throw new L1ExtractionError("provider", "LLM API call failed: HTTP 503");
      return real({ sessionKey });
    });
    const { backend, executor } = await gatewayHarness({
      TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend, instanceId: "svc", baseDir: dir, runL1,
    });
    const seenTasks: TaskPayload[] = [];
    const tracing = { ...executor, executeL1: (t: TaskPayload, s?: AbortSignal) => { seenTasks.push(t); return executor.executeL1(t, s); } };
    await backend.updateSessionState("svc", SESSION, { conversation_count: count }, undefined, AGENT);
    const worker = new PipelineWorker(backend, tracing, {
      concurrency: 1, lockRenewIntervalMs: 60_000, deadLetterRetryDelayMs: RETRY_DELAY_MS,
    } as never, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L1-orig", type: "L1", instanceId: "svc", sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(), data });

    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 1, 120_000)).toBe(true);
    expect(calls).toBe(4);
    const dlqMember = buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: AGENT });
    const deadAt = worker.getDeadLetterQueue()[0].deadAt;
    expect(armedTimers(backend).get(dlqMember)).toBe(deadAt + RETRY_DELAY_MS);
    expect(l1Records).toHaveLength(0);

    await advance(RETRY_DELAY_MS - 60_000, 30_000);
    expect(calls).toBe(4);

    expect(await advanceUntil(() => worker.getMetrics().tasksCompleted === 1, 120_000)).toBe(true);
    expect(l1Records).toHaveLength(1);
    expect(calls).toBe(5);
    expect(seenTasks.at(-1)?.data?.triggeredBy).toBe("dead_letter_retry");
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(12);
    expect(armedTimers(backend).has(dlqMember)).toBe(false);

    // The re-run consumed the live count, so the next L1_idle is deduped
    // instead of running an empty L1.
    expect((await backend.getSessionState("svc", SESSION, undefined, AGENT))?.conversation_count ?? 0).toBe(0);
    await backend.setTimer("svc", buildPipelineTimerMember(SESSION, "L1_idle", { agentId: AGENT }), Date.now() + 1_000);
    expect(await advanceUntil(() => worker.getMetrics().tasksCompleted === 2, 30_000)).toBe(true);
    expect(seenTasks.at(-1)?.data?.triggeredBy).toBe("timer_scanner");
    expect(calls).toBe(5);
    await worker.stop();
  });

  it("the service-mode TimerScanner maps an L1_dlq member to dead_letter_retry too", async () => {
    const enqueued: TaskPayload[] = [];
    let claimed = false;
    const backend = {
      timerShardCount: 1,
      async claimExpiredFromShard() {
        if (claimed) return [];
        claimed = true;
        return [{ instanceId: "svc", member: `svc\x00${buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: AGENT })}`, fireAtMs: 0 }];
      },
      async enqueueTask(t: TaskPayload) { enqueued.push(t); },
    };
    const scanner = new TimerScanner(backend as never, { scanIntervalMs: 1_000 }, silent);
    await (scanner as unknown as { scan(): Promise<void> }).scan();

    expect(enqueued).toHaveLength(1);
    expect(enqueued[0].type).toBe("L1");
    expect(enqueued[0].data?.triggeredBy).toBe("dead_letter_retry");
  });
});
