/**
 * Records the embedding client and the SQLite vector tables as they behave
 * today, after the retry/validation fix (cases E2, E5 and E7 of the
 * memory-path failure table). Recording only.
 *
 *   E2  any 4xx except 429 (400, 401, 403, 404, 422) throws EmbeddingApiError
 *       on the first call, no retry.
 *   E5  a 2xx body without a `data` array throws on the first call; a 2xx
 *       body that is not JSON is treated like a network error: retried twice
 *       (500 ms, 1000 ms) and the SyntaxError is thrown after 3 calls.
 *   E7  a changed embedding dimension (or model) at boot drops and rebuilds
 *       the vec tables and returns needsReindex; L1 metadata rows survive
 *       without vectors, and no source file calls reindexAll(), so nothing
 *       re-embeds them.
 *
 * HTTP is a fake `fetch`; E7 uses a real SQLite VectorStore with sqlite-vec.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OpenAIEmbeddingService } from "../src/core/store/embedding.js";
import { VectorStore } from "../src/core/store/sqlite.js";
import type { MemoryRecord } from "../src/core/record/l1-writer.js";
import { tmpDir } from "./helpers/l1-spike-harness.js";

const DIMS = 4;

function service(): OpenAIEmbeddingService {
  return new OpenAIEmbeddingService({
    provider: "openai", baseUrl: "https://embed.invalid/v1", apiKey: "placeholder-not-a-secret", model: "test-embed", dimensions: DIMS, timeoutMs: 10_000,
  });
}

function fetchAnswering(answer: () => Response) {
  const calls: number[] = [];
  const fetchMock = vi.fn(async () => { calls.push(Date.now()); return answer(); });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, calls };
}

const stores: VectorStore[] = [];
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  for (const s of stores.splice(0)) {
    try { s.close(); } catch { /* already closed */ }
  }
});

describe("E2 embedding 4xx other than 429", () => {
  it.each([400, 401, 403, 404, 422])("E2 HTTP %i throws EmbeddingApiError on the first call, no retry", async (status) => {
    const { fetchMock } = fetchAnswering(() => new Response(JSON.stringify({ error: "nope" }), { status }));

    const err = await service().embed("hello").then(() => undefined, (e: unknown) => e as Error & { httpStatus?: number });

    expect(err?.name).toBe("EmbeddingApiError");
    expect(err?.httpStatus).toBe(status);
    expect(err?.message).toContain(`HTTP ${status}`);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("E5 embedding 2xx with an unusable body", () => {
  it("E5 a body without a data array throws \"missing 'data' array\" after one call", async () => {
    const { fetchMock } = fetchAnswering(() => new Response(JSON.stringify({ object: "list" }), { status: 200 }));

    await expect(service().embed("hello")).rejects.toThrow("missing 'data' array");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("E5 a body that is not JSON is retried like a network error: 3 calls 500 ms then 1000 ms apart, then the SyntaxError is thrown", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const { fetchMock, calls } = fetchAnswering(() => new Response("<html>bad gateway page</html>", { status: 200 }));

    const settled = service().embed("hello").then(() => undefined, (e: unknown) => e as Error);
    await vi.runAllTimersAsync();
    const err = await settled;

    expect(err).toBeInstanceOf(SyntaxError);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(calls.slice(1).map((t, i) => t - calls[i])).toEqual([500, 1_000]);
  });
});

describe("E7 embedding dimensions or model change at boot", () => {
  function open(file: string, dims: number, model = "test-embed"): { store: VectorStore; init: ReturnType<VectorStore["init"]> } {
    const store = new VectorStore(file, dims);
    stores.push(store);
    const init = store.init({ provider: "openai", model });
    return { store, init };
  }

  function record(id: string, content: string): MemoryRecord {
    const now = new Date().toISOString();
    return {
      id, content, type: "episodic", priority: 50, scene_name: "s", source_message_ids: [], metadata: {},
      timestamps: [now], createdAt: now, updatedAt: now, sessionKey: "proj:e7", sessionId: "s",
    } as MemoryRecord;
  }

  it("E7 dims 4 -> 8: needsReindex with reason \"dimensions: 4 → 8\"; the L1 row survives but has no vector any more", async () => {
    const file = path.join(tmpDir("e7-dims-"), "vectors.db");
    const first = open(file, 4);
    expect(first.init).toEqual({ needsReindex: false });
    expect(first.store.upsertL1(record("r1", "hotel fact with a vector"), new Float32Array([1, 0, 0, 0]))).toBe(true);
    expect(first.store.searchL1Vector(new Float32Array([1, 0, 0, 0]), 5).map((r) => r.record_id)).toEqual(["r1"]);
    first.store.close();
    stores.splice(stores.indexOf(first.store), 1);

    const second = open(file, 8);

    expect(second.init).toEqual({ needsReindex: true, reason: "dimensions: 4 → 8" });
    expect(second.store.countL1()).toBe(1);
    expect(second.store.searchL1Vector(new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]), 5)).toEqual([]);
    expect(second.store.searchL1Fts("hotel", 5).map((r) => r.record_id)).toEqual(["r1"]);
  });

  it("E7 a model change alone also drops the vec tables and asks for a reindex", () => {
    const file = path.join(tmpDir("e7-model-"), "vectors.db");
    const first = open(file, 4, "model-a");
    first.store.close();
    stores.splice(stores.indexOf(first.store), 1);

    expect(open(file, 4, "model-b").init).toEqual({ needsReindex: true, reason: "model: model-a → model-b" });
  });

  it("E7 no source file calls reindexAll(): needsReindex is reported but nothing re-embeds the rows", () => {
    const srcDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src");
    const callers: string[] = [];
    const walk = (d: string) => {
      for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, e.name);
        if (e.isDirectory()) walk(p);
        else if (p.endsWith(".ts") && !p.endsWith(".test.ts")) {
          const text = fs.readFileSync(p, "utf8");
          if (/\.reindexAll\s*\(/.test(text)) callers.push(path.relative(srcDir, p));
        }
      }
    };
    walk(srcDir);
    expect(callers).toEqual([]);
  });
});
