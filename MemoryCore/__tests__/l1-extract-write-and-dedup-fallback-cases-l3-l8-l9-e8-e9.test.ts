/**
 * Records the L1 extraction and write path as it behaves today (cases L3, L8,
 * L9, E8 and E9 of the memory-path failure table). No behaviour is changed.
 *
 *   L3  happy path: one LLM call, taskId "l1-extraction", 180 s timeout, no
 *       tools; the runner logs "[l1] L1 complete: extracted=N".
 *   L8  a memory whose type is not a known type is skipped with a warning;
 *       the rest of the scene is still stored.
 *   L9  the dedup LLM failing or answering unparsable text makes every
 *       memory a "store" (duplicates possible).
 *   E8  L1 write: an embedding failure writes metadata + FTS only; a vector
 *       of the wrong width makes SQLite roll back the whole L1 row, the
 *       writer still returns the record (JSONL already written).
 *   E9  dedup embedBatch failing: "Batch dedup failed, storing all as new".
 *
 * LLM and embedding are fakes; the store is either a fake or a real SQLite
 * VectorStore in a temp dir.
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { batchDedup } from "../src/core/record/l1-dedup.js";
import { extractL1Memories } from "../src/core/record/l1-extractor.js";
import { writeMemory } from "../src/core/record/l1-writer.js";
import { VectorStore } from "../src/core/store/sqlite.js";
import type { EmbeddingService } from "../src/core/store/embedding.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import { makeCfg, makeL0Store, msg, providerError, sceneJson, scriptedLlm, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const SESSION = "proj:l1-write";
const messages = [
  { id: "m1", role: "user" as const, content: msg("delta one"), timestamp: 1 },
  { id: "m2", role: "user" as const, content: msg("delta two"), timestamp: 2 },
];

function spyLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

const stores: VectorStore[] = [];
afterEach(() => {
  for (const s of stores.splice(0)) {
    try { s.close(); } catch { /* already closed */ }
  }
});

function realStore(dims = 4): VectorStore {
  const dir = tmpDir("l1-write-sqlite-");
  const s = new VectorStore(path.join(dir, "vectors.db"), dims);
  s.init({ provider: "openai", model: "test-embed" });
  stores.push(s);
  return s;
}

function memory(content: string) {
  return { content, type: "episodic" as const, priority: 60, source_message_ids: ["m1"], metadata: {}, scene_name: "s", record_id: `rec-${content.length}-${content.slice(0, 6)}` };
}

describe("L3 L1 extraction happy path", () => {
  it("L3 one LLM call with taskId l1-extraction, timeoutMs 180000 and no tools; extractedCount = memories in the reply", async () => {
    const { runner, run } = scriptedLlm(() => sceneJson("delta", ["m1", "m2"]));

    const res = await extractL1Memories({
      messages, sessionKey: SESSION, sessionId: "s", baseDir: tmpDir("l1-l3-"), config: undefined, logger: silent,
      options: { enableDedup: false, llmRunner: runner },
    });

    expect(res).toMatchObject({ success: true, extractedCount: 1, storedCount: 1 });
    expect(run).toHaveBeenCalledTimes(1);
    const params = run.mock.calls[0][0] as Record<string, unknown>;
    expect(params.taskId).toBe("l1-extraction");
    expect(params.timeoutMs).toBe(180_000);
    expect(params.tools).toBeUndefined();
    expect(params.enableTools).toBeUndefined();
  });

  it("L3 the runner logs \"[l1] L1 complete: extracted=N, stored=N\" at info level", async () => {
    const dir = tmpDir("l1-l3-runner-");
    const { store } = makeL0Store([
      { id: "L0-1", sessionId: "sess-1", content: msg("echo one"), timestamp: 1, recordedAtMs: 101 },
      { id: "L0-2", sessionId: "sess-1", content: msg("echo two"), timestamp: 2, recordedAtMs: 102 },
    ]);
    const { runner } = scriptedLlm(() => sceneJson("echo", ["L0-1", "L0-2"]));
    const logger = spyLogger();
    const run = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger, llmRunner: runner });

    await run({ sessionKey: SESSION });

    const infos = logger.info.mock.calls.map((c) => String(c[0]));
    expect(infos.some((l) => l.includes("[l1] L1 complete: extracted=1, stored=1 (1 group(s))"))).toBe(true);
  });
});

describe("L8 L1 memory with an invalid type", () => {
  it("L8 a memory typed \"gossip\" is skipped with a warning; the valid memory of the same scene is stored", async () => {
    const reply = JSON.stringify([{
      scene_name: "mixed",
      message_ids: ["m1", "m2"],
      memories: [
        { content: "a fact with an unknown type", type: "gossip", priority: 50, source_message_ids: ["m1"] },
        { content: "a fact with a valid type", type: "episodic", priority: 50, source_message_ids: ["m2"] },
      ],
    }]);
    const { runner } = scriptedLlm(() => reply);
    const logger = spyLogger();

    const res = await extractL1Memories({
      messages, sessionKey: SESSION, sessionId: "s", baseDir: tmpDir("l1-l8-"), config: undefined, logger,
      options: { enableDedup: false, llmRunner: runner },
    });

    expect(res.success).toBe(true);
    expect(res.extractedCount).toBe(1);
    expect(res.records.map((r) => r.content)).toEqual(["a fact with a valid type"]);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("Skipping memory with invalid type \"gossip\""));
  });
});

describe("L9 dedup LLM failure stores every memory as new", () => {
  /** FTS-only store with one existing candidate, so the dedup LLM is consulted. */
  function ftsStore() {
    return {
      countL1: async () => 0,
      isFtsAvailable: () => true,
      searchL1Fts: vi.fn(async () => [{
        record_id: "old-1", content: "an older fact about delta", type: "episodic", priority: 50, scene_name: "s",
        session_key: SESSION, session_id: "s", timestamp_str: "2026-11-01T00:00:00.000Z", metadata_json: "{}",
      }]),
    };
  }
  const mems = [memory("delta one is a new fact"), memory("delta two is another fact")];

  it.each([
    ["LLM error 500", () => { throw providerError(500); }],
    ["prose instead of a JSON array", () => "I could not decide."],
    ["broken JSON", () => "[{\"record_id\": \"x\", \"action\": "],
  ])("L9 %s -> every memory gets action store", async (_label, reply) => {
    const { runner, run } = scriptedLlm(reply);
    const logger = spyLogger();

    const decisions = await batchDedup({ memories: mems, config: undefined, logger, vectorStore: ftsStore() as never, llmRunner: runner });

    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls[0][0]).toMatchObject({ taskId: "l1-conflict-detection", timeoutMs: 180_000 });
    expect(decisions).toEqual(mems.map((m) => ({ record_id: m.record_id, action: "store", target_ids: [] })));
    expect(logger.warn).toHaveBeenCalledTimes(1);
  });

  it("L9 an LLM error logs \"Batch conflict detection failed, defaulting all to store\"", async () => {
    const { runner } = scriptedLlm(() => { throw providerError(503); });
    const logger = spyLogger();

    await batchDedup({ memories: mems, config: undefined, logger, vectorStore: ftsStore() as never, llmRunner: runner });

    expect(logger.warn.mock.calls[0][0]).toContain("Batch conflict detection failed, defaulting all to store");
  });
});

describe("E9 dedup embedBatch failure stores every memory as new", () => {
  it("E9 embedBatch throws -> \"Batch dedup failed, storing all as new\", both memories reach upsertL1, no dedup LLM call", async () => {
    const upsertL1 = vi.fn(async () => true);
    const store = {
      countL1: async () => 3,
      isFtsAvailable: () => true,
      upsertL1,
      searchL1Vector: vi.fn(async () => []),
    };
    const embedding = {
      embedBatch: vi.fn(async () => { throw new Error("Embedding API error: HTTP 503"); }),
      embed: vi.fn(async () => new Float32Array([1, 0, 0, 0])),
    } as unknown as EmbeddingService;
    const reply = JSON.stringify([{
      scene_name: "two",
      message_ids: ["m1", "m2"],
      memories: [
        { content: "delta one is a new fact", type: "episodic", priority: 50, source_message_ids: ["m1"] },
        { content: "delta two is another fact", type: "episodic", priority: 50, source_message_ids: ["m2"] },
      ],
    }]);
    const { runner, run } = scriptedLlm(() => reply);
    const logger = spyLogger();

    const res = await extractL1Memories({
      messages, sessionKey: SESSION, sessionId: "s", baseDir: tmpDir("l1-e9-"), config: undefined, logger,
      options: { enableDedup: true, llmRunner: runner, vectorStore: store as never, embeddingService: embedding },
    });

    expect(res.storedCount).toBe(2);
    expect(run).toHaveBeenCalledTimes(1); // extraction only; the dedup LLM is never reached
    expect(upsertL1).toHaveBeenCalledTimes(2);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("Batch dedup failed, storing all as new: Embedding API error: HTTP 503"));
  });
});

describe("E8 L1 write when the embedding fails or has the wrong width", () => {
  const base = { decision: { record_id: "", action: "store" as const, target_ids: [] }, sessionKey: SESSION, sessionId: "s" };

  it("E8 embed throws -> \"Embedding FAILED ... will write metadata only\"; the row is stored and found by FTS, not by vector search", async () => {
    const store = realStore(4);
    const mem = memory("echo fact stored without a vector");
    const logger = spyLogger();
    const embedding = { embed: vi.fn(async () => { throw new Error("Embedding API error: HTTP 401"); }) } as unknown as EmbeddingService;

    const rec = await writeMemory({ ...base, decision: { ...base.decision, record_id: mem.record_id }, memory: mem, baseDir: tmpDir("l1-e8-"), logger, vectorStore: store, embeddingService: embedding });

    expect(rec?.id).toBe(mem.record_id);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining(`Embedding FAILED for id=${mem.record_id}, will write metadata only`));
    expect(store.countL1()).toBe(1);
    expect(store.searchL1Fts("echo", 5).map((r) => r.record_id)).toEqual([mem.record_id]);
    expect(store.searchL1Vector(new Float32Array([1, 0, 0, 0]), 5)).toEqual([]);
  });

  it("E8 a 3-wide vector into a 4-wide store: upsertL1 rolls the whole row back and returns false; writeMemory still returns the record and its JSONL line", async () => {
    const store = realStore(4);
    const mem = memory("foxtrot fact with a short vector");
    const logger = spyLogger();
    const baseDir = tmpDir("l1-e8b-");
    const embedding = { embed: vi.fn(async () => new Float32Array([0.6, 0.8, 0])) } as unknown as EmbeddingService;
    const upsert = vi.spyOn(store, "upsertL1");

    const rec = await writeMemory({ ...base, decision: { ...base.decision, record_id: mem.record_id }, memory: mem, baseDir, logger, vectorStore: store, embeddingService: embedding });

    expect(rec?.id).toBe(mem.record_id);
    expect(upsert).toHaveReturnedWith(false);
    expect(store.countL1()).toBe(0);
    expect(store.searchL1Fts("foxtrot", 5)).toEqual([]);
    const jsonl = fs.readdirSync(path.join(baseDir, "records")).map((f) => fs.readFileSync(path.join(baseDir, "records", f), "utf-8")).join("");
    expect(jsonl).toContain(mem.record_id);
  });

  it("E8 upsertL1 throwing is caught as \"[vec-dual-write] FAILED (JSONL already written)\"", async () => {
    const mem = memory("golf fact whose store throws");
    const logger = spyLogger();
    const store = { upsertL1: vi.fn(async () => { throw new Error("database is locked"); }) };

    const rec = await writeMemory({ ...base, decision: { ...base.decision, record_id: mem.record_id }, memory: mem, baseDir: tmpDir("l1-e8c-"), logger, vectorStore: store as never });

    expect(rec?.id).toBe(mem.record_id);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining(`[vec-dual-write] FAILED (JSONL already written) id=${mem.record_id}: database is locked`));
  });
});
