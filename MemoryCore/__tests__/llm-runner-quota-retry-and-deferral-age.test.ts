/**
 * A quota / rate-limit reply reaches PipelineWorker.deferForQuota on the
 * first call, and the age of a deferral counts from its first deferral.
 *
 *   T5a  pipeline runner (StandaloneLLMRunnerFactory): an HTTP 429, or a
 *        MiniMax base_resp quota code (2056) on any status, ends the call
 *        after one fetch with an LlmQuotaError; the AI SDK does not retry it,
 *        and the worker defers the task.
 *   T5b  regression: on the same runner an HTTP 503 or a network failure is
 *        still retried by the AI SDK; a runner built directly (skill
 *        extractor, no deferral behind it) still retries a 429.
 *   T5c  batchDedup (core/record/l1-dedup.ts): a 429 costs one fetch and every
 *        memory falls back to "store"; a 503 then 200 is retried and the
 *        decision of the 200 is returned.
 *   T6   QuotaDeferralEntry.firstDeferredAtMs survives repeated deferrals and
 *        a restart (FileQuotaDeferralStore), an old file without the field
 *        falls back to fireAtMs - delayMs, and /health reports the oldest
 *        age as services.pipelineWorker.maxDeferralAgeMs.
 *
 * No network: fetch is a stub, the gateway LLM url is .invalid.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import { APICallError } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StandaloneLLMRunner, StandaloneLLMRunnerFactory } from "../src/adapters/standalone/llm-runner.js";
import { batchDedup } from "../src/core/record/l1-dedup.js";
import type { IMemoryStore } from "../src/core/store/types.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { FileQuotaDeferralStore } from "../src/services/quota-deferral-store.js";
import { LlmQuotaError, findLlmQuotaSignal, isLlmQuotaError } from "../src/utils/llm-quota-error.js";
import { advance, advanceUntil, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const CONFIG = { baseUrl: "https://llm.invalid/v1", apiKey: "placeholder-not-a-secret", model: "MiniMax-M3.1-Flash" };
const JSON_HEADERS = { "content-type": "application/json" };
// Keeps the AI SDK's own back-off at 1 ms so a retried call stays fast.
const FAST_RETRY = { ...JSON_HEADERS, "retry-after-ms": "1" };

function completion(text: string): Response {
  return new Response(JSON.stringify({
    id: "chatcmpl-1", object: "chat.completion", created: 1, model: CONFIG.model,
    choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: text } }],
    usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
    base_resp: { status_code: 0, status_msg: "" },
  }), { status: 200, headers: JSON_HEADERS });
}

function rateLimited(): Response {
  return new Response(JSON.stringify({ error: { message: "rate limited", type: "rate_limit" } }), { status: 429, headers: FAST_RETRY });
}

/** MiniMax plan usage limit (base_resp 2056) on the given HTTP status. */
function planLimit(status: number): Response {
  return new Response(JSON.stringify({ id: "x", choices: null, base_resp: { status_code: 2056, status_msg: "usage limit exceeded" } }), { status, headers: FAST_RETRY });
}

function unavailable(): Response {
  return new Response(JSON.stringify({ error: { message: "upstream unavailable" } }), { status: 503, headers: FAST_RETRY });
}

/** What undici throws when the socket fails, which the AI SDK retries. */
function networkFailure(): TypeError {
  return new TypeError("fetch failed", { cause: Object.assign(new Error("socket hang up"), { code: "ECONNRESET" }) });
}

/** Stub global fetch with one scripted reply per call (the last one repeats). */
function scriptFetch(...replies: Array<() => Response | Error>) {
  let call = 0;
  const fetchMock = vi.fn(async () => {
    const reply = replies[Math.min(call++, replies.length - 1)]();
    if (reply instanceof Error) throw reply;
    return reply;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The runner every pipeline stage (L1, dedup, L2, L3) gets from the factory. */
function pipelineRunner() {
  return new StandaloneLLMRunnerFactory({ config: CONFIG, logger: silent }).createRunner({ enableTools: false });
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("T5a pipeline runner: a quota reply ends the call after one fetch", () => {
  it.each([
    ["HTTP 429", rateLimited, 429, undefined],
    ["base_resp 2056 on HTTP 500", () => planLimit(500), 500, 2056],
    ["base_resp 2056 on HTTP 200", () => planLimit(200), 200, 2056],
  ] as const)("T5a %s: one fetch, LlmQuotaError, no AI SDK retry", async (_label, reply, status, providerCode) => {
    const fetchMock = scriptFetch(reply, () => completion("[]"));
    const err = await pipelineRunner().run({ prompt: "hi", taskId: "t5a" }).catch((e: unknown) => e);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(err).toBeInstanceOf(LlmQuotaError);
    expect(isLlmQuotaError(err)).toBe(true);
    expect(findLlmQuotaSignal(err)).toEqual({ status, ...(providerCode !== undefined ? { providerCode } : {}) });
  });

  it("T5a the worker defers the L1 task on that single call (no retry, no dead letter)", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const fetchMock = scriptFetch(rateLimited);
    const runner = pipelineRunner();
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, {
      executeL1: async () => { await runner.run({ prompt: "extract", taskId: "l1-extraction" }); },
      executeL2: async () => {}, executeL3: async () => {},
    }, { concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: 15 * 60_000 } as never, silent);
    await worker.start();
    await backend.enqueueTask(l1Task("L1-t5a", "proj:t5a"));

    expect(await advanceUntil(() => worker.getMetrics().quotaDeferred === 1, 60_000)).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(worker.getMetrics()).toMatchObject({ quotaDeferred: 1, quotaDeferredPending: 1, tasksDeadLettered: 0, tasksRetried: 0 });
    await worker.stop();
    await backend.destroy();
  }, 30_000);
});

describe("T5b regression: transient failures are still retried by the AI SDK", () => {
  it("T5b pipeline runner: 503 then 200 is retried and succeeds", async () => {
    const fetchMock = scriptFetch(unavailable, () => completion("ok-after-503"));
    await expect(pipelineRunner().run({ prompt: "hi", taskId: "t5b" })).resolves.toBe("ok-after-503");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("T5b pipeline runner: a network failure then 200 is retried and succeeds", async () => {
    const fetchMock = scriptFetch(networkFailure, () => completion("ok-after-reset"));
    await expect(pipelineRunner().run({ prompt: "hi", taskId: "t5b-net" })).resolves.toBe("ok-after-reset");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("T5b a runner built directly (no deferral behind it) keeps the AI SDK retry on 429", async () => {
    const fetchMock = scriptFetch(rateLimited, () => completion("ok-after-429"));
    const runner = new StandaloneLLMRunner({ config: CONFIG, logger: silent });
    await expect(runner.run({ prompt: "hi", taskId: "skill" })).resolves.toBe("ok-after-429");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("T5c batchDedup on the pipeline runner", () => {
  const memories = [{ record_id: "m-new", content: "user prefers green tea in the morning", type: "persona", priority: 50 }] as never;

  /** FTS-only store with one existing candidate, so dedup asks the LLM. */
  function ftsStore(): IMemoryStore {
    return {
      countL1: async () => 1,
      isFtsAvailable: () => true,
      searchL1Fts: async () => [{
        record_id: "m-old", content: "user likes green tea", type: "persona", priority: 50, scene_name: "",
        metadata_json: "{}", timestamp_str: "", session_key: "s", session_id: "s",
      }],
    } as unknown as IMemoryStore;
  }

  it("T5c 429: one fetch, every memory falls back to store", async () => {
    const fetchMock = scriptFetch(rateLimited, () => completion("[]"));
    const decisions = await batchDedup({ memories, config: {}, logger: silent, vectorStore: ftsStore(), llmRunner: pipelineRunner() });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(decisions).toEqual([{ record_id: "m-new", action: "store", target_ids: [] }]);
  });

  it("T5c regression 503 then 200: retried, the 200 decision is returned", async () => {
    const answer = JSON.stringify([{ record_id: "m-new", action: "skip", target_ids: ["m-old"] }]);
    const fetchMock = scriptFetch(unavailable, () => completion(answer));
    const decisions = await batchDedup({ memories, config: {}, logger: silent, vectorStore: ftsStore(), llmRunner: pipelineRunner() });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(decisions).toMatchObject([{ record_id: "m-new", action: "skip", target_ids: ["m-old"] }]);
  });
});

// ============================
// T6 deferral age
// ============================

const DEFER_MS = 15 * 60_000;

function l1Task(id: string, sessionId: string): TaskPayload {
  return { id, type: "L1", instanceId: "svc", sessionId, agentId: "default", priority: 0, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } };
}

function quota429(): APICallError {
  return new APICallError({ message: "Too Many Requests", url: "https://api.minimax.io/v1/chat/completions", requestBodyValues: {}, statusCode: 429 });
}

interface StoredEntry { id: string; fireAtMs: number; deferrals: number; firstDeferredAtMs?: number; task: TaskPayload }

function storedEntries(file: string): StoredEntry[] {
  return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, "utf-8")) as { entries: StoredEntry[] }).entries : [];
}

function deferringWorker(backend: LocalStateBackend, file: string, calls: { n: number }): PipelineWorker {
  const executor: TaskExecutor = {
    executeL1: async () => { calls.n++; throw quota429(); }, executeL2: async () => {}, executeL3: async () => {},
  };
  return new PipelineWorker(backend, executor, {
    concurrency: 1, lockRenewIntervalMs: 60_000, quotaDeferDelayMs: DEFER_MS, quotaDeferralStore: new FileQuotaDeferralStore(file),
  } as never, silent);
}

const ageOf = (w: PipelineWorker) => (w.getMetrics() as unknown as { maxDeferralAgeMs?: number }).maxDeferralAgeMs;

describe("T6 firstDeferredAtMs: the age of a deferral counts from its first deferral", () => {
  it("T6 two deferrals and one restart keep the first deferral time; maxDeferralAgeMs counts from it", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const file = path.join(tmpDir("deferral-age-"), "metadata", "pipeline-quota-deferrals.json");

    const calls1 = { n: 0 };
    const backend1 = new LocalStateBackend();
    const worker1 = deferringWorker(backend1, file, calls1);
    await worker1.start();
    expect(ageOf(worker1)).toBe(0);
    await backend1.enqueueTask(l1Task("L1-age", "proj:age"));

    // Deferral #1.
    expect(await advanceUntil(() => worker1.getMetrics().quotaDeferred === 1, 30_000)).toBe(true);
    const [first] = storedEntries(file);
    expect(first.deferrals).toBe(1);
    const firstAt = first.firstDeferredAtMs;
    expect(typeof firstAt).toBe("number");
    expect(first.fireAtMs - firstAt!).toBe(DEFER_MS);

    // The re-run fails again: deferral #2 of the same entry, first time unchanged.
    expect(await advanceUntil(() => worker1.getMetrics().quotaDeferred === 2, DEFER_MS + 60_000, 30_000)).toBe(true);
    const [second] = storedEntries(file);
    expect(second).toMatchObject({ id: first.id, deferrals: 2, firstDeferredAtMs: firstAt });
    expect(second.fireAtMs).toBeGreaterThan(first.fireAtMs);
    expect(ageOf(worker1)).toBe(Date.now() - firstAt!);
    expect(ageOf(worker1)).toBeGreaterThanOrEqual(DEFER_MS);
    await worker1.stop();
    await backend1.destroy();

    // Restart on the same file: the entry keeps its first time, and the
    // re-run (createdAt = now) failing a third time does not reset it.
    const calls2 = { n: 0 };
    const backend2 = new LocalStateBackend();
    const worker2 = deferringWorker(backend2, file, calls2);
    await worker2.start();
    expect(ageOf(worker2)).toBe(Date.now() - firstAt!);
    expect(await advanceUntil(() => worker2.getMetrics().quotaDeferred === 1, DEFER_MS + 60_000, 30_000)).toBe(true);
    expect(calls2.n).toBe(1);
    const [third] = storedEntries(file);
    expect(third).toMatchObject({ id: first.id, deferrals: 3, firstDeferredAtMs: firstAt });
    expect(ageOf(worker2)).toBe(Date.now() - firstAt!);
    expect(ageOf(worker2)).toBeGreaterThanOrEqual(2 * DEFER_MS);
    await worker2.stop();
    await backend2.destroy();
  }, 60_000);

  it("T6 a deferral file written before firstDeferredAtMs existed falls back to fireAtMs - delayMs", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const file = path.join(tmpDir("deferral-age-legacy-"), "metadata", "pipeline-quota-deferrals.json");
    const now = Date.now();
    const legacyFireAt = now + 10 * 60_000; // deferred 5 min ago with a 15 min delay
    const task = { ...l1Task("L1-legacy", "proj:legacy"), data: { instanceId: "svc", agentId: "default", quotaDeferralId: "qd-legacy" } };
    fs.mkdirSync(path.dirname(file), { recursive: true });
    // Same (task type, session scope) key the worker builds for this task.
    const key = ["L1", "svc", "", "default", "proj:legacy"].join("\u0000");
    fs.writeFileSync(file, JSON.stringify({ version: 1, entries: [{ id: "qd-legacy", key, task, fireAtMs: legacyFireAt, deferrals: 1 }] }), "utf-8");

    const calls = { n: 0 };
    const backend = new LocalStateBackend();
    const worker = deferringWorker(backend, file, calls);
    await worker.start();
    expect(ageOf(worker)).toBe(5 * 60_000);

    // Its re-run fails again: the fallback first time is kept on the new entry.
    expect(await advanceUntil(() => worker.getMetrics().quotaDeferred === 1, 11 * 60_000, 30_000)).toBe(true);
    const [entry] = storedEntries(file);
    expect(entry).toMatchObject({ id: "qd-legacy", deferrals: 2, firstDeferredAtMs: legacyFireAt - DEFER_MS });
    expect(ageOf(worker)).toBe(Date.now() - (legacyFireAt - DEFER_MS));
    await worker.stop();
    await backend.destroy();
  }, 60_000);

  it("T6 /health shows services.pipelineWorker.maxDeferralAgeMs from the oldest first deferral", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "deferral-age-gw-"));
    const dataDir = path.join(dir, "data");
    const file = path.join(dataDir, "metadata", "pipeline-quota-deferrals.json");
    const now = Date.now();
    const entries = [
      // Deferred three times, first 2 h ago, next run in 10 min.
      { id: "qd-old", key: "L1\u0000old", task: { ...l1Task("L1-old", "proj:old"), data: { quotaDeferralId: "qd-old" } }, fireAtMs: now + 10 * 60_000, deferrals: 3, firstDeferredAtMs: now - 2 * 60 * 60_000 },
      // Old file format: no firstDeferredAtMs, deferred 5 min ago (15 min default delay).
      { id: "qd-legacy", key: "L1\u0000legacy", task: { ...l1Task("L1-legacy", "proj:legacy"), data: { quotaDeferralId: "qd-legacy" } }, fireAtMs: now + 10 * 60_000, deferrals: 1 },
    ];
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ version: 1, entries }), "utf-8");
    const cfgFile = path.join(dir, "tdai-gateway.yaml");
    fs.writeFileSync(cfgFile, ["server:", "  port: 0", "  host: 127.0.0.1", "data:", `  baseDir: ${JSON.stringify(dataDir)}`, ""].join("\n"), "utf-8");
    vi.stubEnv("TDAI_GATEWAY_CONFIG", cfgFile);
    vi.stubEnv("TDAI_DATA_DIR", undefined);
    vi.stubEnv("WORKER_LLM_QUOTA_DEFER_MS", undefined);
    vi.stubEnv("TDAI_LLM_BASE_URL", CONFIG.baseUrl);
    vi.stubEnv("TDAI_LLM_MODEL", "m");
    vi.stubEnv("TDAI_LLM_API_KEY", CONFIG.apiKey);

    const gw = new TdaiGateway() as unknown as { start(): Promise<void>; stop(): Promise<void>; server: { address(): AddressInfo } };
    await gw.start();
    try {
      const res = await fetch(`http://127.0.0.1:${gw.server.address().port}/health`);
      const body = await res.json() as { services: { pipelineWorker: { maxDeferralAgeMs?: number; quotaDeferredPending: number } } };
      const pw = body.services.pipelineWorker;
      expect(pw.quotaDeferredPending).toBe(2);
      const expected = Date.now() - (now - 2 * 60 * 60_000);
      expect(typeof pw.maxDeferralAgeMs).toBe("number");
      expect(pw.maxDeferralAgeMs).toBeGreaterThanOrEqual(2 * 60 * 60_000);
      expect(pw.maxDeferralAgeMs).toBeLessThanOrEqual(expected);
    } finally {
      await gw.stop();
    }
  }, 60_000);
});
