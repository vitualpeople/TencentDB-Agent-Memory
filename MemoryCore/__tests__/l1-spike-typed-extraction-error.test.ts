/**
 * Spike, mechanism 3: a typed L1ExtractionError travels from the extractor,
 * through createL1Runner and the gateway executor, to the PipelineWorker,
 * which can tell a provider failure from an unparsable reply.
 *
 * Opt-in: extractL1Memories({ options: { throwOnFailure: true } }) and
 * createL1Runner({ throwTypedL1Errors: true }). With the flags off the
 * behaviour recorded in l1-failure-path-current-behavior.test.ts is unchanged.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { extractL1Memories } from "../src/core/record/l1-extractor.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  advanceUntil, makeCfg, makeL0Store, msg, providerError, sceneJson, scriptedLlm, silent, tmpDir,
} from "./helpers/l1-spike-harness.js";

const ERROR_MODULE = "../src/core/record/l1-extraction-error.js";
const messages = [{ id: "m1", role: "user" as const, content: msg("delta"), timestamp: 1 }];

async function extractWith(reply: () => string): Promise<unknown> {
  const { runner } = scriptedLlm(reply);
  return extractL1Memories({
    messages, sessionKey: "proj:demo", sessionId: "s", baseDir: tmpDir("l1-typed-"), config: undefined, logger: silent,
    options: { enableDedup: false, llmRunner: runner, throwOnFailure: true } as never,
  }).then(() => undefined, (err: unknown) => err);
}

describe("extractor throws L1ExtractionError when throwOnFailure is set", () => {
  it.each([
    ["401", () => { throw providerError(401); }],
    ["429", () => { throw providerError(429); }],
    ["500", () => { throw providerError(500); }],
    ["abort", () => { throw Object.assign(new Error("The operation was aborted"), { name: "AbortError" }); }],
  ])("provider failure %s -> kind provider, frozen ids = input ids", async (_label, reply) => {
    const { L1ExtractionError } = await import(ERROR_MODULE);
    const err = await extractWith(reply as () => string);
    expect(err).toBeInstanceOf(L1ExtractionError);
    expect((err as { kind: string }).kind).toBe("provider");
    expect((err as { frozenMessageIds: string[] }).frozenMessageIds).toEqual(["m1"]);
  });

  it.each([
    ["prose, no JSON array", "I could not find anything worth remembering."],
    ["broken JSON", "[{\"scene_name\": \"x\", \"memories\": [ {\"content\": "],
    ["truncated mid-array", "[{\"scene_name\":\"x\",\"message_ids\":[\"m1\"],\"memories\":[{\"content\":\"a\"}]},{\"scene_"],
    ["array of numbers", "[1]"],
    ["array of a bare object", "[{}]"],
    ["item without memories", "[{\"scene_name\":\"x\",\"message_ids\":[\"m1\"]}]"],
    ["item without message_ids", "[{\"scene_name\":\"x\",\"memories\":[]}]"],
  ])("unparsable reply (%s) -> kind parse", async (_label, reply) => {
    const { L1ExtractionError } = await import(ERROR_MODULE);
    const err = await extractWith(() => reply);
    expect(err).toBeInstanceOf(L1ExtractionError);
    expect((err as { kind: string }).kind).toBe("parse");
  });

  it.each([
    ["empty array", "[]"],
    ["valid scene", sceneJson("delta", ["m1"])],
  ])("%s is a success, not an error", async (_label, reply) => {
    await import(ERROR_MODULE);
    expect(await extractWith(() => reply)).toBeUndefined();
  });
});

describe("the worker receives the typed error from the real runner", () => {
  afterEach(() => { vi.useRealTimers(); });

  it.each([
    ["provider", () => { throw providerError(503); }],
    ["parse", () => "nothing to see here"],
  ])("dead-letter entry carries errorKind %s after 1 + 3 runs", async (kind, reply) => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const { L1ExtractionError } = await import(ERROR_MODULE);
    const dir = tmpDir("l1-typed-worker-");
    const { store } = makeL0Store([{ id: "L0-x1", sessionId: "sess-x", content: msg("echo"), timestamp: 10, recordedAtMs: 11 }]);
    const { runner, run } = scriptedLlm(reply as () => string);
    const l1 = createL1Runner({
      pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined,
      logger: silent, llmRunner: runner, throwTypedL1Errors: true,
    } as never);
    const seen: unknown[] = [];
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, {
      async executeL1(task) {
        try { await l1({ sessionKey: task.sessionId }); } catch (err) { seen.push(err); throw err; }
      },
      async executeL2() {}, async executeL3() {},
    }, { concurrency: 1, lockRenewIntervalMs: 60_000 }, silent);
    await worker.start();
    await backend.enqueueTask({ id: "L1-1", type: "L1", instanceId: "svc", sessionId: "proj:demo", priority: 0, createdAt: Date.now(), data: { instanceId: "svc" } });

    expect(await advanceUntil(() => worker.getMetrics().tasksDeadLettered === 1, 120_000)).toBe(true);

    expect(run).toHaveBeenCalledTimes(4);
    expect(seen).toHaveLength(4);
    for (const e of seen) expect(e).toBeInstanceOf(L1ExtractionError);
    const dlq = worker.getDeadLetterQueue();
    expect(dlq).toHaveLength(1);
    expect((dlq[0] as { errorKind?: string }).errorKind).toBe(kind);
    await worker.stop();
  });
});
