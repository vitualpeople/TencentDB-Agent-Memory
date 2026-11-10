/**
 * Records the /v2 search routes when embedding fails or no search backend
 * exists, as they behave today (cases E10, E11 and E12 of the memory-path
 * failure table). Recording only.
 *
 *   E10 atomic/search: the query is embedded with no per-call timeout option
 *       (the service default applies); a failed embed is logged and the route
 *       falls back to FTS; with FTS also empty it returns code 0 and no item.
 *   E11 conversation/search: the same, on L0.
 *   E12 no embedding service and no FTS: strategy "none", [] and code 0 on
 *       the /v2 routes (no error code reaches the hub).
 *
 * The store and the embedding service are fakes.
 */
import { describe, expect, it, vi } from "vitest";
import type { EmbeddingService } from "../src/core/store/embedding.js";
import { executeMemorySearch } from "../src/core/tools/memory-search.js";
import { executeConversationSearch } from "../src/core/tools/conversation-search.js";
import { handleAtomicSearch, handleConversationSearch, type V2RouterDeps } from "../src/gateway/v2-router.js";

const auth = { serviceId: "svc" } as Parameters<typeof handleAtomicSearch>[1];

function spyLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function failingEmbedding() {
  const embed = vi.fn(async () => { throw new Error("Embedding API error: HTTP 401 Unauthorized (body 20 bytes)"); });
  return { embed, service: { embed } as unknown as EmbeddingService };
}

const l1Hit = {
  record_id: "l1-1", content: "india fact", type: "episodic", priority: 50, scene_name: "s", score: 0.7,
  team_id: "", user_id: "", agent_id: "", task_id: "", version: 1, timestamp_start: "2026-11-01T00:00:00.000Z", timestamp_end: "2026-11-01T00:00:00.000Z",
};
const l0Hit = { record_id: "l0-1", session_key: "proj:e11", role: "user", message_text: "india message", score: 0.7, recorded_at: "2026-11-01T00:00:00.000Z" };

function store(opts: { fts: boolean; l1?: unknown[]; l0?: unknown[] }) {
  return {
    isFtsAvailable: () => opts.fts,
    getCapabilities: () => ({ nativeHybridSearch: false }),
    searchL1Fts: vi.fn(async () => opts.l1 ?? []),
    searchL1Vector: vi.fn(async () => []),
    searchL0Fts: vi.fn(async () => opts.l0 ?? []),
    searchL0Vector: vi.fn(async () => []),
  };
}

function deps(s: unknown, embedding: EmbeddingService | undefined, logger = spyLogger()): V2RouterDeps {
  return { getStore: () => s, getEmbedding: () => embedding, getStorage: () => undefined, logger, deployMode: "standalone" } as unknown as V2RouterDeps;
}

describe("E10 atomic/search when the query embed fails", () => {
  it("E10 embed(query) is called with no per-call options; the failure is logged and FTS answers", async () => {
    const { embed, service } = failingEmbedding();
    const s = store({ fts: true, l1: [l1Hit] });
    const logger = spyLogger();

    const res = await executeMemorySearch({ query: "india", limit: 5, vectorStore: s as never, embeddingService: service, logger });

    expect(embed).toHaveBeenCalledTimes(1);
    expect(embed.mock.calls[0]).toEqual(["india"]);
    expect(res.strategy).toBe("fts");
    expect(res.results.map((r) => r.id)).toEqual(["l1-1"]);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("[hybrid-vec] Embedding search failed (non-fatal)"));
    expect(s.searchL1Vector).not.toHaveBeenCalled();
  });

  it("E10 the route returns code 0 with the FTS items", async () => {
    const res = await handleAtomicSearch({ query: "india" }, auth, "req-e10", deps(store({ fts: true, l1: [l1Hit] }), failingEmbedding().service));

    expect(res.code).toBe(0);
    expect((res.data as { items: Array<{ id: string }> }).items.map((i) => i.id)).toEqual(["l1-1"]);
  });

  it("E10 embed fails and FTS finds nothing: code 0, empty items, no error signal", async () => {
    const res = await handleAtomicSearch({ query: "india" }, auth, "req-e10b", deps(store({ fts: true }), failingEmbedding().service));

    expect(res.code).toBe(0);
    expect((res.data as { items: unknown[] }).items).toEqual([]);
  });
});

describe("E11 conversation/search when the query embed fails", () => {
  it("E11 embed(query) is called with no per-call options; the failure is logged and L0 FTS answers", async () => {
    const { embed, service } = failingEmbedding();
    const s = store({ fts: true, l0: [l0Hit] });
    const logger = spyLogger();

    const res = await executeConversationSearch({ query: "india", limit: 5, vectorStore: s as never, embeddingService: service, logger });

    expect(embed.mock.calls[0]).toEqual(["india"]);
    expect(res.strategy).toBe("fts");
    expect(res.results.map((r) => r.id)).toEqual(["l0-1"]);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("[hybrid-vec] Embedding search failed (non-fatal)"));
  });

  it("E11 embed fails and FTS finds nothing: the route returns code 0 and no message", async () => {
    const res = await handleConversationSearch({ query: "india" }, auth, "req-e11", deps(store({ fts: true }), failingEmbedding().service));

    expect(res.code).toBe(0);
    expect((res.data as { messages: unknown[] }).messages).toEqual([]);
  });
});

describe("E12 no embedding service and no FTS", () => {
  it("E12 executeMemorySearch returns strategy none with an explanatory message and no result", async () => {
    const res = await executeMemorySearch({ query: "india", limit: 5, vectorStore: store({ fts: false }) as never, embeddingService: undefined, logger: spyLogger() });

    expect(res.strategy).toBe("none");
    expect(res.results).toEqual([]);
    expect(res.message).toContain("Embedding service is not configured and FTS is not available");
  });

  it("E12 executeConversationSearch also returns strategy none and no result", async () => {
    const res = await executeConversationSearch({ query: "india", limit: 5, vectorStore: store({ fts: false }) as never, embeddingService: undefined, logger: spyLogger() });

    expect(res.strategy).toBe("none");
    expect(res.results).toEqual([]);
  });

  it("E12 both /v2 routes answer code 0 with empty data", async () => {
    const atomic = await handleAtomicSearch({ query: "india" }, auth, "req-e12a", deps(store({ fts: false }), undefined));
    const conv = await handleConversationSearch({ query: "india" }, auth, "req-e12b", deps(store({ fts: false }), undefined));

    expect(atomic.code).toBe(0);
    expect((atomic.data as { items: unknown[] }).items).toEqual([]);
    expect(conv.code).toBe(0);
    expect((conv.data as { messages: unknown[] }).messages).toEqual([]);
  });
});
