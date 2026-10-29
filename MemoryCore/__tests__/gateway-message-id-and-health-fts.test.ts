import type http from "node:http";
import { describe, expect, it, vi } from "vitest";
import { TdaiGateway } from "../src/gateway/server.js";
import type { HealthResponse } from "../src/gateway/types.js";
import { handleConversationAdd, type V2RouterDeps } from "../src/gateway/v2-router.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

describe("/v2/conversation/add message ids", () => {
  it("uses the full 32 hex chars of a UUID after msg-", async () => {
    const upsertL0 = vi.fn();
    const deps = {
      getStore: () => ({ upsertL0 }),
      getEmbedding: () => undefined,
      getStorage: () => undefined,
      logger: silent,
      deployMode: "service",
    } as unknown as V2RouterDeps;

    const res = await handleConversationAdd(
      { session_id: "proj:demo", messages: [{ role: "user", content: "a" }, { role: "assistant", content: "b" }] },
      { serviceId: "svc" } as Parameters<typeof handleConversationAdd>[1],
      "req-1",
      deps,
    );

    expect(res.code).toBe(0);
    const ids = (res.data as { accepted_ids: string[] }).accepted_ids;
    expect(ids).toHaveLength(2);
    for (const id of ids) {
      expect(id).toMatch(/^msg-[0-9a-f]{32}$/);
      expect(id).toHaveLength(36);
    }
    expect(new Set(ids).size).toBe(2);
    expect(upsertL0.mock.calls.map((c) => c[0].id)).toEqual(ids);
  });
});

/** Calls the private /health handler on a gateway whose core is stubbed. */
function health(vectorStore: unknown): HealthResponse {
  const gw = Object.create(TdaiGateway.prototype) as TdaiGateway;
  Object.assign(gw, {
    startTime: Date.now(),
    core: { getVectorStore: () => vectorStore, getEmbeddingService: () => undefined },
    timerScanner: null,
    pipelineWorker: null,
    stateBackend: null,
  });
  let body = "";
  const res = { writeHead: vi.fn(), end: (s: string) => { body = s; } } as unknown as http.ServerResponse;
  (gw as unknown as { handleHealth(r: http.ServerResponse): void }).handleHealth(res);
  return JSON.parse(body) as HealthResponse;
}

describe("/health stores.ftsAvailable", () => {
  it.each([true, false])("reports the vector store's FTS5 availability (%s)", (fts) => {
    const out = health({ isFtsAvailable: () => fts });

    expect(typeof out.stores.ftsAvailable).toBe("boolean");
    expect(out.stores.ftsAvailable).toBe(fts);
  });

  it("is false when no vector store is wired", () => {
    const out = health(undefined);

    expect(out.stores.ftsAvailable).toBe(false);
    expect(out.status).toBe("degraded");
  });
});
