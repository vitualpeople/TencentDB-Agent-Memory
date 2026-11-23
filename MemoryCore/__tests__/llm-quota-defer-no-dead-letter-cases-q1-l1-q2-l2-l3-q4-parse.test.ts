/**
 * An LLM quota / rate-limit failure defers the memory task instead of
 * retrying it into the dead letter (fake LLM, fake clock):
 *
 *   Q1 L1 gets HTTP 429 (an AI SDK APICallError, wrapped by the real L1
 *      extractor): the task is deferred by quotaDeferDelayMs, no retry and no
 *      dead letter, the L1 cursor stays put; after the delay the re-run
 *      extracts the slice. The worker log names the status, not the body.
 *   Q2 the same for L2 and L3 through the worker; and the real L2 / L3
 *      runners, whose generators swallow LLM errors into a result value,
 *      throw a typed LlmQuotaError for a 429 instead of a skip / a generic
 *      "Persona generation failed".
 *   Q4 a parse failure and a non-quota provider failure (HTTP 500) keep the
 *      old path: 3 retries, then the dead letter (and its L1_dlq timer).
 *
 * Restart while deferred (Q3) is in
 * llm-quota-defer-restart-runs-once-case-q3.test.ts.
 */
import fs from "node:fs";
import path from "node:path";
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

import { APICallError } from "ai";
import { parseConfig } from "../src/config.js";
import { L1ExtractionError } from "../src/core/record/l1-extraction-error.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload } from "../src/core/state/types.js";
import type { IMemoryStore, L1RecordRow } from "../src/core/store/types.js";
import type { LLMRunner } from "../src/core/types.js";
import type { MemoryTdaiConfig } from "../src/config.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner, createL2Runner, createL3Runner } from "../src/utils/pipeline-factory.js";
import {
  advance, advanceUntil, armedTimers, gatewayHarness, makeCfg, makeL0Store, msg, sceneJson, scriptedLlm, silent, tmpDir,
} from "./helpers/l1-spike-harness.js";

const INSTANCE = "svc";
const SESSION = "proj:quota";
const AGENT = "default";
const DEFER_MS = 15 * 60_000;
const SECRET_BODY = "PROVIDER-BODY-MUST-NOT-BE-LOGGED";

/** MiniMax-style 429 as the AI SDK throws it; the body must never reach a log. */
function quota429(): APICallError {
  return new APICallError({
    message: "Too Many Requests",
    url: "https://api.minimax.io/v1/chat/completions",
    requestBodyValues: {},
    statusCode: 429,
    responseBody: JSON.stringify({ base_resp: { status_code: 1002, status_msg: SECRET_BODY } }),
    isRetryable: true,
  });
}

function capturingLogger() {
  const lines: string[] = [];
  const at = (m: string) => { lines.push(m); };
  return { logger: { debug: at, info: at, warn: at, error: at }, lines };
}

const metric = (w: PipelineWorker, k: string) => (w.getMetrics() as unknown as Record<string, number>)[k];

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); hoisted.captured = undefined; });

describe("Q1 L1 quota failure is deferred, not dead-lettered", () => {
  it("Q1 429 at L1: deferred 15 min with the cursor in place, no retry, no dead letter; the re-run extracts the slice", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const dir = tmpDir("quota-l1-");
    const { store, l1Records } = makeL0Store([
      { id: "L0-q1", sessionId: "sess-q", agentId: AGENT, content: msg("quota one"), timestamp: 10, recordedAtMs: 11 },
      { id: "L0-q2", sessionId: "sess-q", agentId: AGENT, content: msg("quota two"), timestamp: 20, recordedAtMs: 12 },
    ]);
    const { runner, run } = scriptedLlm((_p, call) => {
      if (call === 1) throw quota429();
      return sceneJson("quota", ["L0-q1", "L0-q2"]);
    });
    const real = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });
    const { backend, executor } = await gatewayHarness({
      TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend, instanceId: INSTANCE, baseDir: dir,
      runL1: (sessionKey, slice) => real({ sessionKey, ...slice }),
    });
    const seen: TaskPayload[] = [];
    const tracing = { ...executor, executeL1: (t: TaskPayload, s?: AbortSignal) => { seen.push({ ...t }); return executor.executeL1(t, s); } };
    const { logger, lines } = capturingLogger();
    const worker = new PipelineWorker(backend, tracing, { concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS } as never, logger);
    await worker.start();
    await backend.enqueueTask({ id: "L1-orig", type: "L1", instanceId: INSTANCE, sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: AGENT } });

    expect(await advanceUntil(() => metric(worker, "quotaDeferred") === 1, 30_000)).toBe(true);
    expect(worker.getMetrics()).toMatchObject({ tasksDeadLettered: 0, tasksRetried: 0, tasksCompleted: 0 });
    expect(worker.getDeadLetterQueue()).toHaveLength(0);
    expect(run).toHaveBeenCalledTimes(1);
    expect(l1Records).toHaveLength(0);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION]?.last_l1_cursor ?? 0).toBe(0);
    expect(armedTimers(backend).has(buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: AGENT }))).toBe(false);

    // Nothing runs before the delay is over.
    await advance(DEFER_MS - 60_000, 30_000);
    expect(run).toHaveBeenCalledTimes(1);

    expect(await advanceUntil(() => worker.getMetrics().tasksCompleted === 1, 120_000)).toBe(true);
    expect(run).toHaveBeenCalledTimes(2);
    expect(l1Records).toHaveLength(1);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(12);
    // Same task again, retry count untouched.
    expect(seen).toHaveLength(2);
    expect(seen[1].data?.retryCount ?? 0).toBe(0);
    expect(worker.getMetrics()).toMatchObject({ tasksDeadLettered: 0, tasksRetried: 0, quotaDeferred: 1, quotaDeferredPending: 0 });

    // The worker log names the status; the provider body is in no log line.
    expect(lines.some((l) => l.includes("LLM quota or rate limit on L1") && l.includes("status=429"))).toBe(true);
    expect(lines.filter((l) => l.includes(SECRET_BODY))).toEqual([]);
    await worker.stop();
  });
});

describe("Q2 L2 and L3 quota failures are deferred, not dead-lettered", () => {
  it("Q2 429 at L2: deferred with no L3 and no L2 cursor move; the re-run completes and cascades to L3", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = new LocalStateBackend();
    let l2Calls = 0;
    const l3Calls: TaskPayload[] = [];
    const executor: TaskExecutor = {
      executeL1: async () => {},
      executeL2: async () => { if (++l2Calls === 1) throw quota429(); },
      executeL3: async (t) => { l3Calls.push(t); },
    };
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS } as never, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L2-orig", type: "L2", instanceId: INSTANCE, sessionId: SESSION, agentId: AGENT, priority: 1, createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: AGENT } });

    expect(await advanceUntil(() => metric(worker, "quotaDeferred") === 1, 30_000)).toBe(true);
    expect(worker.getMetrics()).toMatchObject({ tasksDeadLettered: 0, tasksRetried: 0 });
    expect((await backend.getSessionState(INSTANCE, SESSION, undefined, AGENT))?.l2_last_extraction_time).toBeFalsy();
    await advance(DEFER_MS - 60_000, 30_000);
    expect(l2Calls).toBe(1);
    expect(l3Calls).toHaveLength(0);

    expect(await advanceUntil(() => l3Calls.length === 1, 120_000)).toBe(true);
    expect(l2Calls).toBe(2);
    expect((await backend.getSessionState(INSTANCE, SESSION, undefined, AGENT))?.l2_last_extraction_time).toBeTruthy();
    expect(worker.getMetrics()).toMatchObject({ tasksDeadLettered: 0, tasksRetried: 0, quotaDeferred: 1 });
    await worker.stop();
  });

  it("Q2 429 at L3: deferred past its 1-retry budget without a dead letter; the re-run completes", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = new LocalStateBackend();
    const seen: TaskPayload[] = [];
    const executor: TaskExecutor = {
      executeL1: async () => {},
      executeL2: async () => {},
      executeL3: async (t) => { seen.push({ ...t }); if (seen.length <= 2) throw quota429(); },
    };
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS } as never, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L3-orig", type: "L3", instanceId: INSTANCE, sessionId: SESSION, agentId: AGENT, priority: 2, createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: AGENT } });

    expect(await advanceUntil(() => metric(worker, "quotaDeferred") === 1, 30_000)).toBe(true);
    expect(await advanceUntil(() => metric(worker, "quotaDeferred") === 2, DEFER_MS + 60_000, 30_000)).toBe(true);
    expect(await advanceUntil(() => worker.getMetrics().tasksCompleted === 1, DEFER_MS + 60_000, 30_000)).toBe(true);
    expect(seen).toHaveLength(3);
    expect(seen.map((t) => t.data?.retryCount ?? 0)).toEqual([0, 0, 0]);
    expect(worker.getMetrics()).toMatchObject({ tasksDeadLettered: 0, tasksRetried: 0, quotaDeferredPending: 0 });
    await worker.stop();
  });

  it("Q2 the real L2 runner throws a typed LlmQuotaError for a 429 instead of returning a skip", async () => {
    const dataDir = tmpDir("quota-l2-runner-");
    const updated = "2026-11-02T10:00:00.000Z";
    const row: L1RecordRow = {
      record_id: "r1", content: "memory r1", type: "episodic", priority: 50, scene_name: "",
      session_key: SESSION, session_id: SESSION, team_id: "", task_id: "", user_id: "", agent_id: "",
      version: 1, timestamp_str: updated, timestamp_start: updated, timestamp_end: updated,
      created_time: updated, updated_time: updated, metadata_json: "{}",
    };
    const vectorStore = { isDegraded: () => false, queryL1Records: async () => [row] } as unknown as IMemoryStore;
    const run = vi.fn(async () => { throw quota429(); });
    const cfg = { llm: { enabled: false }, persona: { model: "fake-model", promptMode: undefined, maxScenes: 10, sceneBackupCount: 3 } } as unknown as MemoryTdaiConfig;
    const l2 = createL2Runner({ pluginDataDir: dataDir, cfg, openclawConfig: undefined, vectorStore, logger: silent, llmRunner: { run } as unknown as LLMRunner });

    const err = await l2(SESSION).then(() => undefined, (e: unknown) => e);
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).name).toBe("LlmQuotaError");
    expect((err as Error).message).toContain("status=429");
    expect((err as Error).message).not.toContain(SECRET_BODY);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("Q2 the real L3 runner throws a typed LlmQuotaError for a 429 and leaves the persona checkpoint in place", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = tmpDir("quota-l3-runner-");
    fs.mkdirSync(path.join(dir, "scene_blocks"), { recursive: true });
    fs.writeFileSync(path.join(dir, "scene_blocks", "coding.md"),
      ["---", "scene: coding", "created: 2026-11-01T09:00:00.000Z", "updated: 2026-11-01T10:00:00.000Z", "heat: 1", "---", "", "## Summary", "Coding scene.", ""].join("\n"), "utf-8");
    await writeSceneIndex(dir, [{ filename: "coding.md", summary: "Coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" }]);
    const cp = new CheckpointManager(dir, silent);
    await cp.incrementScenesProcessed();
    await cp.incrementMemoriesSincePersona(4);
    const run = vi.fn(async () => { throw quota429(); });
    const l3 = createL3Runner({ pluginDataDir: dir, cfg: parseConfig({}), openclawConfig: undefined, logger: silent, llmRunner: { run } as unknown as LLMRunner });

    const err = await l3().then(() => undefined, (e: unknown) => e);
    expect((err as Error)?.name).toBe("LlmQuotaError");
    expect((err as Error).message).toContain("status=429");
    expect(run).toHaveBeenCalledTimes(1);
    const after = await new CheckpointManager(dir, silent).read();
    expect(after.last_persona_time).toBe("");
    expect(after.memories_since_last_persona).toBe(4);
  });
});

describe("Q4 non-quota failures keep the old retry and dead-letter path", () => {
  it.each([
    ["parse failure", () => new L1ExtractionError("parse", "No JSON array found in extraction response"), "parse"],
    ["provider HTTP 500", () => new L1ExtractionError("provider", "LLM API call failed", {
      cause: new APICallError({ message: "Internal Server Error", url: "https://api.minimax.io/v1/chat/completions", requestBodyValues: {}, statusCode: 500 }),
    }), "provider"],
  ])("Q4 L1 %s: 3 retries (5 s / 15 s / 45 s), dead letter, L1_dlq armed, nothing deferred", async (_label, makeErr, kind) => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = new LocalStateBackend();
    let calls = 0;
    const executor: TaskExecutor = {
      executeL1: async () => { calls++; throw makeErr(); },
      executeL2: async () => {},
      executeL3: async () => {},
    };
    const worker = new PipelineWorker(backend, executor, { concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS } as never, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L1-bad", type: "L1", instanceId: INSTANCE, sessionId: SESSION, agentId: AGENT, priority: 0, createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: AGENT } });

    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 1, 120_000)).toBe(true);
    expect(calls).toBe(4);
    expect(worker.getMetrics()).toMatchObject({ tasksRetried: 3, quotaDeferred: 0 });
    expect(worker.getDeadLetterQueue()[0].errorKind).toBe(kind);
    expect(armedTimers(backend).has(buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: AGENT }))).toBe(true);
    await worker.stop();
  });
});
