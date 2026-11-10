import type http from "node:http";
import { describe, expect, it, vi } from "vitest";

vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  return { ...actual, generateText: vi.fn() };
});

const { APICallError, generateText } = await import("ai");
const { StandaloneLLMRunner } = await import("../src/adapters/standalone/llm-runner.js");
const { OpenAIEmbeddingService } = await import("../src/core/store/embedding.js");
const { TdaiGateway } = await import("../src/gateway/server.js");
const { handleConversationAdd } = await import("../src/gateway/v2-router.js");
type V2RouterDeps = import("../src/gateway/v2-router.js").V2RouterDeps;

/**
 * Provider health as /health reports it. Failures are driven through the real
 * OpenAIEmbeddingService (fake fetch) and real StandaloneLLMRunner instances
 * (generateText mocked), so the test pins that every runner and embedding
 * service writes into the one telemetry sink /health reads.
 */
interface ProviderBlock {
  lastOkAt: string | null;
  lastErrorAt: string | null;
  lastErrorStatus: number | null;
  consecutiveFailures: number;
}
interface Health {
  status: string;
  stores: {
    vectorStore: boolean;
    embeddingService: boolean;
    ftsAvailable?: boolean;
    embedding?: ProviderBlock & { l0EmbeddingFailed?: number };
    llm?: ProviderBlock;
  };
}

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };
const KEY_BODY = "Incorrect API key provided: sk-abcdefghijklmnopqrstuvwxyz";

function health(): Health {
  const gw = Object.create(TdaiGateway.prototype) as InstanceType<typeof TdaiGateway>;
  Object.assign(gw, {
    startTime: Date.now(),
    core: { getVectorStore: () => ({ isFtsAvailable: () => true }), getEmbeddingService: () => ({}) },
    timerScanner: null,
    pipelineWorker: null,
    stateBackend: null,
  });
  let body = "";
  const res = { writeHead: vi.fn(), end: (s: string) => { body = s; } } as unknown as http.ServerResponse;
  (gw as unknown as { handleHealth(r: http.ServerResponse): void }).handleHealth(res);
  expect(body).not.toContain("sk-abcdefghijklmnop");
  return JSON.parse(body) as Health;
}

function embedService() {
  return new OpenAIEmbeddingService({
    provider: "openai",
    baseUrl: "https://embed.invalid/v1",
    apiKey: "test-key-not-a-secret",
    model: "test-embed",
    dimensions: 2,
  });
}

function fetchAnswering(status: number) {
  vi.stubGlobal("fetch", vi.fn(async () => status === 200
    ? new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 0] }] }), { status: 200 })
    : new Response(JSON.stringify({ error: KEY_BODY }), { status })));
}

function runner() {
  return new StandaloneLLMRunner({
    config: { baseUrl: "https://llm.invalid/v1", apiKey: "test-key-not-a-secret", model: "test-model" },
    logger: silent,
  });
}

function llm401() {
  return new APICallError({
    message: "Unauthorized",
    url: "https://llm.invalid/v1/chat/completions",
    requestBodyValues: {},
    statusCode: 401,
    responseBody: JSON.stringify({ error: KEY_BODY }),
  });
}

function llmOk() {
  vi.mocked(generateText).mockResolvedValueOnce({ text: "ok", steps: [] } as never);
}

describe("/health keeps its existing fields", () => {
  it("still carries vectorStore, embeddingService and ftsAvailable as booleans", () => {
    const out = health();

    expect(out.stores.vectorStore).toBe(true);
    expect(out.stores.embeddingService).toBe(true);
    expect(out.stores.ftsAvailable).toBe(true);
  });
});

describe("/health stores.embedding", () => {
  it("three embedding 401s make status degraded; one success resets consecutiveFailures", async () => {
    const svc = embedService();
    fetchAnswering(200);
    await svc.embed("warm");
    expect(health().stores.embedding?.consecutiveFailures).toBe(0);

    fetchAnswering(401);
    for (let i = 0; i < 3; i++) await expect(svc.embed("x")).rejects.toThrow(/HTTP 401/);

    const bad = health();
    expect(bad.status).toBe("degraded");
    expect(bad.stores.embedding?.consecutiveFailures).toBe(3);
    expect(bad.stores.embedding?.lastErrorStatus).toBe(401);
    expect(typeof bad.stores.embedding?.lastErrorAt).toBe("string");

    fetchAnswering(200);
    await svc.embed("again");
    const good = health();
    expect(good.stores.embedding?.consecutiveFailures).toBe(0);
    expect(typeof good.stores.embedding?.lastOkAt).toBe("string");
    expect(good.status).toBe("ok");
  });

  it("an L0 embedding failure still stores the record without a vector and counts l0EmbeddingFailed", async () => {
    const before = health().stores.embedding?.l0EmbeddingFailed ?? 0;
    const upsertL0 = vi.fn(() => true);
    const failing = { embed: vi.fn(async () => { throw new Error("boom"); }) };
    const deps = {
      getStore: () => ({ upsertL0 }),
      getEmbedding: () => failing,
      getStorage: () => undefined,
      logger: silent,
      deployMode: "service",
    } as unknown as V2RouterDeps;

    const res = await handleConversationAdd(
      { session_id: "proj:demo", messages: [{ role: "user", content: "kept" }] },
      { serviceId: "svc" } as Parameters<typeof handleConversationAdd>[1],
      "req-h",
      deps,
    );

    expect(res.code).toBe(0);
    expect(upsertL0).toHaveBeenCalledTimes(1);
    expect(upsertL0.mock.calls[0][1]).toBeUndefined();
    expect(health().stores.embedding?.l0EmbeddingFailed).toBe(before + 1);
  });
});

describe("/health stores.llm", () => {
  it("two separate runners write into one sink: three LLM 401s make status degraded", async () => {
    const a = runner();
    const b = runner();
    llmOk();
    await a.run({ prompt: "warm", taskId: "t0" });
    expect(health().stores.llm?.consecutiveFailures).toBe(0);

    vi.mocked(generateText).mockRejectedValueOnce(llm401());
    await expect(a.run({ prompt: "x", taskId: "t1" })).rejects.toThrow();
    vi.mocked(generateText).mockRejectedValueOnce(llm401());
    await expect(b.run({ prompt: "x", taskId: "t2" })).rejects.toThrow();
    vi.mocked(generateText).mockRejectedValueOnce(llm401());
    await expect(b.run({ prompt: "x", taskId: "t3" })).rejects.toThrow();

    const bad = health();
    expect(bad.status).toBe("degraded");
    expect(bad.stores.llm?.consecutiveFailures).toBe(3);
    expect(bad.stores.llm?.lastErrorStatus).toBe(401);

    llmOk();
    await a.run({ prompt: "again", taskId: "t4" });
    const good = health();
    expect(good.stores.llm?.consecutiveFailures).toBe(0);
    expect(typeof good.stores.llm?.lastOkAt).toBe("string");
    expect(good.status).toBe("ok");
  });

  it("a caller abort is not counted as a provider failure", async () => {
    const a = runner();
    llmOk();
    await a.run({ prompt: "warm", taskId: "t5" });
    const controller = new AbortController();
    controller.abort();
    vi.mocked(generateText).mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }));

    await expect(a.run({ prompt: "x", taskId: "t6", abortSignal: controller.signal })).rejects.toThrow();

    expect(health().stores.llm?.consecutiveFailures).toBe(0);
  });
});
