/**
 * Frozen L1 slice and the give-up cap for replies that never parse.
 *
 * At its first typed failure an L1 slice is frozen by its upper
 * recorded_at_ms bound (PipelineWorker, in RAM); every later run of the
 * session processes only rows up to that bound, so a good message arriving
 * between retries is never folded into the failing slice. After 3
 * consecutive dead-letters caused by parse failures (3 x (1 + 3) = 12 LLM
 * calls) the next run skips the slice, counts l1SliceSkipped and moves on.
 * Provider failures never count toward the cap, and a slice with a parse and
 * a provider group failing together is a provider failure.
 *
 * Real PipelineWorker, real gateway executeL1 and real createL1Runner, with a
 * scripted LLM and a fake clock (default 15 min dead-letter re-run delay).
 */
import http from "node:http";
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
import { TdaiGateway } from "../src/gateway/server.js";
import type { HealthResponse } from "../src/gateway/types.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  advanceUntil, gatewayHarness, makeCfg, makeL0Store, msg, providerError, sceneJson, scriptedLlm, silent, tmpDir, type L0Row,
} from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";
const AGENT = "default";
const PROSE = "I am sorry, there is nothing I can turn into memories here.";
/** Three dead-letter rounds plus the 15 min re-run delay after each, with margin. */
const CAP_WINDOW_MS = 4 * (65_000 + 15 * 60_000) + 60_000;

/** Ids of `rows` the prompt names. */
function idsIn(prompt: string, rows: L0Row[]): string[] {
  return rows.map((r) => r.id).filter((id) => prompt.includes(id));
}

async function pipeline(rows: L0Row[], answer: (prompt: string, call: number) => string) {
  const dir = tmpDir("l1-cap-");
  const { store, l1Records } = makeL0Store(rows);
  const prompts: string[] = [];
  const llm = scriptedLlm((prompt, call) => { prompts.push(prompt); return answer(prompt, call); });
  const real = createL1Runner({
    pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined,
    logger: silent, llmRunner: llm.runner,
  });
  const slices: Array<{ sliceUpperBoundMs?: number; skipThroughMs?: number } | undefined> = [];
  const { backend, executor } = await gatewayHarness({
    TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend, instanceId: "svc", baseDir: dir,
    runL1: (sessionKey, slice) => { slices.push(slice); return real({ sessionKey, ...slice }); },
  });
  const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockRenewIntervalMs: 60_000 }, silent);
  await worker.start();
  await backend.enqueueTask({
    id: "L1-orig", type: "L1", instanceId: "svc", sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(),
    data: { instanceId: "svc", agentId: AGENT },
  });
  const cursor = async () => (await new CheckpointManager(dir, silent).read()).runner_states[SESSION]?.last_l1_cursor;
  return { worker, l1Records, prompts, slices, cursor };
}

describe("a slice whose replies never parse costs at most 12 calls, then is skipped", () => {
  afterEach(() => { vi.useRealTimers(); hoisted.captured = undefined; });

  it("prose for the slice, 3 good messages after it: 12 calls on the slice, then the good ones are extracted", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const rows: L0Row[] = [{ id: "L0-x1", sessionId: "sess-x", agentId: AGENT, content: msg("xray"), timestamp: 10, recordedAtMs: 11 }];
    const good: L0Row[] = [1, 2, 3].map((i) => ({
      id: `L0-g${i}`, sessionId: "sess-x", agentId: AGENT, content: msg(`golf ${i}`), timestamp: 20 + i, recordedAtMs: 20 + i,
    }));
    const p = await pipeline(rows, (prompt) => (prompt.includes("L0-x1") ? PROSE : sceneJson("golf", idsIn(prompt, good))));

    expect(await advanceUntil(() => p.worker.getMetrics().tasksDeadLettered === 1, 120_000)).toBe(true);
    rows.push(...good);

    expect(await advanceUntil(() => p.worker.getMetrics().l1SliceSkipped === 1, CAP_WINDOW_MS, 5_000)).toBe(true);

    const badPrompts = p.prompts.filter((pr) => pr.includes("L0-x1"));
    expect(badPrompts).toHaveLength(12);
    for (const pr of badPrompts) expect(pr).not.toContain("L0-g1");
    expect(p.prompts).toHaveLength(13);
    expect(p.l1Records).toHaveLength(1);
    expect(p.l1Records[0].content).toContain("golf");
    expect(await p.cursor()).toBe(23);
    expect(p.worker.getMetrics()).toMatchObject({ tasksDeadLettered: 3, l1ExtractionFailed: 12, l1SliceSkipped: 1, l1FrozenSlices: 0 });
    expect(p.slices.filter((s) => s?.skipThroughMs === 11)).toHaveLength(1);
    await p.worker.stop();
  }, 60_000);

  it("a good message that arrives between two retries is not skipped with the failing slice", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const rows: L0Row[] = [{ id: "L0-x1", sessionId: "sess-x", agentId: AGENT, content: msg("xray"), timestamp: 10, recordedAtMs: 11 }];
    const late: L0Row = { id: "L0-g1", sessionId: "sess-x", agentId: AGENT, content: msg("golf late"), timestamp: 30, recordedAtMs: 31 };
    const p = await pipeline(rows, (prompt, call) => {
      if (call === 1) rows.push(late); // arrives right after the first failed attempt
      return prompt.includes("L0-x1") ? PROSE : sceneJson("golf", idsIn(prompt, [late]));
    });

    expect(await advanceUntil(() => p.worker.getMetrics().l1SliceSkipped === 1, CAP_WINDOW_MS, 5_000)).toBe(true);

    const badPrompts = p.prompts.filter((pr) => pr.includes("L0-x1"));
    expect(badPrompts.length).toBeLessThanOrEqual(12);
    for (const pr of badPrompts) expect(pr).not.toContain("L0-g1");
    expect(p.l1Records).toHaveLength(1);
    expect(await p.cursor()).toBe(31);
    await p.worker.stop();
  }, 60_000);
});

describe("provider failures never count toward the cap", () => {
  afterEach(() => { vi.useRealTimers(); hoisted.captured = undefined; });

  const mixedRows = (): L0Row[] => [
    { id: "L0-p1", sessionId: "sess-p", agentId: AGENT, content: msg("papa"), timestamp: 10, recordedAtMs: 11 },
    { id: "L0-q1", sessionId: "sess-q", agentId: AGENT, content: msg("quebec"), timestamp: 20, recordedAtMs: 12 },
  ];
  /** Group p always gets prose, group q always a 503. */
  const mixedAnswer = (prompt: string): string => {
    if (prompt.includes("L0-q1")) throw providerError(503);
    return PROSE;
  };

  it("a parse group and a provider group failing together make a provider failure", async () => {
    const dir = tmpDir("l1-cap-mixed-");
    const { store } = makeL0Store(mixedRows());
    const { runner } = scriptedLlm(mixedAnswer);
    const run = createL1Runner({
      pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined,
      logger: silent, llmRunner: runner,
    });

    const err = await run({ sessionKey: SESSION }).then(() => undefined, (e: unknown) => e);

    expect(err).toBeInstanceOf(L1ExtractionError);
    expect((err as L1ExtractionError).kind).toBe("provider");
    expect([...(err as L1ExtractionError).frozenMessageIds].sort()).toEqual(["L0-p1", "L0-q1"]);
  });

  it("4 dead-letter rounds of that mixed slice: nothing skipped, the slice is still retried", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const p = await pipeline(mixedRows(), mixedAnswer);

    expect(await advanceUntil(() => p.worker.getMetrics().tasksDeadLettered === 4, CAP_WINDOW_MS + 16 * 60_000, 5_000)).toBe(true);

    expect(p.worker.getMetrics()).toMatchObject({ l1SliceSkipped: 0, l1FrozenSlices: 1 });
    expect(p.slices.some((s) => s?.skipThroughMs !== undefined)).toBe(false);
    expect(p.prompts.filter((pr) => pr.includes("L0-p1"))).toHaveLength(16);
    expect(await p.cursor() ?? 0).toBe(0); // never past the earliest failed row
    await p.worker.stop();
  }, 60_000);
});

describe("/health reports the L1 failure counters", () => {
  it("services.pipelineWorker carries l1ExtractionFailed and l1SliceSkipped", async () => {
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, {
      async executeL1() { throw new L1ExtractionError("provider", "LLM API call failed: HTTP 401", { upperBoundMs: 11 }); },
      async executeL2() {}, async executeL3() {},
    }, { concurrency: 1, pollIntervalMs: 10, maxRetries: 0, deadLetterRetryDelayMs: 0 }, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L1-1", type: "L1", instanceId: "svc", sessionId: SESSION, priority: 0, createdAt: Date.now(), data: { instanceId: "svc" } });
    for (let i = 0; i < 200 && worker.getMetrics().tasksDeadLettered === 0; i++) await new Promise((r) => setTimeout(r, 10));

    const gw = Object.create(TdaiGateway.prototype) as TdaiGateway;
    Object.assign(gw, {
      startTime: Date.now(),
      core: { getVectorStore: () => undefined, getEmbeddingService: () => undefined },
      timerScanner: null,
      pipelineWorker: worker,
      stateBackend: backend,
    });
    let body = "";
    const res = { writeHead: vi.fn(), end: (s: string) => { body = s; } } as unknown as http.ServerResponse;
    (gw as unknown as { handleHealth(r: http.ServerResponse): void }).handleHealth(res);
    const health = JSON.parse(body) as HealthResponse;

    expect(health.services.pipelineWorker).toMatchObject({ l1ExtractionFailed: 1, l1SliceSkipped: 0, tasksDeadLettered: 1 });
    await worker.stop();
  });
});
