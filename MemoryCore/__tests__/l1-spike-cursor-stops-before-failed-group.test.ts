/**
 * Cursor on a failed group: when one group of an L1 slice fails, the cursor stops
 * just before the earliest failed group, the message ids of groups that did
 * finish are remembered so the retry skips them, and the profile scopes of
 * those finished groups still reach the worker so their L2 is scheduled.
 * Retrying never re-extracts (and re-writes, record ids are random) a
 * finished group.
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

import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { buildProfileL2Key, createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  advanceUntil, gatewayHarness, makeCfg, makeL0Store, msg, providerError, sceneJson, scriptedLlm, silent, tmpDir, type L0Row,
} from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";
const keyA = buildProfileL2Key({ userId: "u", agentId: "a", sessionId: "sess-a" });
const keyB = buildProfileL2Key({ userId: "u", agentId: "a", sessionId: "sess-b" });

function interleavedRows(): L0Row[] {
  return [
    { id: "L0-a1", sessionId: "sess-a", content: msg("alpha one"), timestamp: 1_000, recordedAtMs: 1_001 },
    { id: "L0-b1", sessionId: "sess-b", content: msg("bravo one"), timestamp: 1_100, recordedAtMs: 1_002 },
    { id: "L0-a2", sessionId: "sess-a", content: msg("alpha two"), timestamp: 1_200, recordedAtMs: 1_003 },
    { id: "L0-b2", sessionId: "sess-b", content: msg("bravo two"), timestamp: 1_300, recordedAtMs: 1_004 },
  ];
}

/** LLM that fails for the group containing `failingId` while `down()` is true. */
function llmFailingFor(failingId: string, down: () => boolean) {
  const prompts: string[] = [];
  const llm = scriptedLlm((prompt) => {
    prompts.push(prompt);
    if (prompt.includes(failingId) && down()) throw providerError(503);
    const ids = ["L0-a1", "L0-a2", "L0-b1", "L0-b2"].filter((id) => prompt.includes(id));
    return sceneJson(ids[0]?.startsWith("L0-a") ? "alpha" : "bravo", ids);
  });
  return { ...llm, prompts };
}

function runnerFor(dir: string, store: unknown, llmRunner: unknown) {
  return createL1Runner({
    pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined,
    logger: silent, llmRunner: llmRunner as never,
  });
}

async function runnerState(dir: string) {
  return (await new CheckpointManager(dir, silent).read()).runner_states[SESSION] as Record<string, unknown>;
}

describe("runner: cursor stops before the earliest failed group", () => {
  it("group one (earliest, first pending row) fails: cursor stays put, group two kept as done, retry writes group one only", async () => {
    const dir = tmpDir("l1-resume-1-");
    const { store, l1Records } = makeL0Store(interleavedRows());
    let down = true;
    const llm = llmFailingFor("L0-a1", () => down);
    const run = runnerFor(dir, store, llm.runner);

    const err = await run({ sessionKey: SESSION }).then(() => undefined, (e: unknown) => e) as Record<string, unknown> | undefined;
    expect(err?.name).toBe("L1ExtractionError");
    expect(err?.kind).toBe("provider");
    expect(err?.frozenMessageIds).toEqual(["L0-a1", "L0-a2"]);
    expect(err?.completedProfileScopes).toEqual([keyB]);
    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-b"]);
    let st = await runnerState(dir);
    expect(st.last_l1_cursor).toBe(0);
    expect(st.l1_done_message_ids).toEqual(["L0-b1", "L0-b2"]);

    down = false;
    const ok = await run({ sessionKey: SESSION });
    expect(ok.profileScopes).toEqual([keyA]);
    expect(llm.prompts.at(-1)).toContain("L0-a1");
    expect(llm.prompts.at(-1)).not.toContain("L0-b1");
    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-b", "sess-a"]);
    st = await runnerState(dir);
    expect(st.last_l1_cursor).toBe(1_004);
    expect(st.l1_done_message_ids ?? []).toEqual([]);
  });

  it("group two fails: cursor = last row before it (<= its first row - 1), only group one's rows past the cursor are kept as done", async () => {
    const dir = tmpDir("l1-resume-2-");
    const { store, l1Records } = makeL0Store(interleavedRows());
    let down = true;
    const llm = llmFailingFor("L0-b1", () => down);
    const run = runnerFor(dir, store, llm.runner);

    await expect(run({ sessionKey: SESSION })).rejects.toMatchObject({ name: "L1ExtractionError", completedProfileScopes: [keyA] });
    let st = await runnerState(dir);
    expect(st.last_l1_cursor).toBe(1_001);
    expect(st.l1_done_message_ids).toEqual(["L0-a2"]);

    down = false;
    await run({ sessionKey: SESSION });
    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-a", "sess-b"]);
    st = await runnerState(dir);
    expect(st.last_l1_cursor).toBe(1_004);
  });
});

describe("worker: no message dropped, no duplicate write, L2 of the finished group scheduled", () => {
  afterEach(() => { vi.useRealTimers(); hoisted.captured = undefined; });

  it("two interleaved groups, group one fails once: L2 for group two on the failed run, group one on the retry", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const dir = tmpDir("l1-resume-w-");
    const { store, l1Records } = makeL0Store(interleavedRows());
    let aCalls = 0;
    const llm = llmFailingFor("L0-a1", () => ++aCalls === 1);
    const run = runnerFor(dir, store, llm.runner);
    const { backend, executor } = await gatewayHarness({
      TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend,
      instanceId: "svc", baseDir: dir, runL1: (sessionKey) => run({ sessionKey }),
    });
    const l2: string[] = [];
    const worker = new PipelineWorker(backend, executor, {
      concurrency: 1, lockRenewIntervalMs: 60_000, onL1Complete: async (key) => { l2.push(key); },
    }, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L1-1", type: "L1", instanceId: "svc", sessionId: SESSION, priority: 0, createdAt: Date.now(), data: { instanceId: "svc" } });

    expect(await advanceUntil(() => l2.length >= 2, 60_000)).toBe(true);

    expect(l2).toEqual([keyB, keyA]);
    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-b", "sess-a"]);
    expect(worker.getMetrics()).toMatchObject({ tasksFailed: 1, tasksRetried: 1, tasksCompleted: 1, tasksDeadLettered: 0 });
    expect((await runnerState(dir)).last_l1_cursor).toBe(1_004);
    await worker.stop();
  });
});
