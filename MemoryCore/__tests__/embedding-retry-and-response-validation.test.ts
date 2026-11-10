import { afterEach, describe, expect, it, vi } from "vitest";
import { EmbeddingDimensionError, OpenAIEmbeddingService } from "../src/core/store/embedding.js";

/**
 * POST {baseUrl}/embeddings is answered by a scripted fake `fetch`. Each step
 * is one HTTP answer (or a hang that only ends when the call's AbortSignal
 * fires). The gaps between calls are read from the fake clock where the test
 * pins backoff, so no test sleeps for real longer than a few hundred ms.
 */
type Step =
  | { status: number; body?: unknown; headers?: Record<string, string> }
  | { hang: true };

const DIMS = 4;

function service(timeoutMs = 10_000): OpenAIEmbeddingService {
  return new OpenAIEmbeddingService({
    provider: "openai",
    baseUrl: "https://embed.invalid/v1",
    apiKey: "test-key-not-a-secret",
    model: "test-embed",
    dimensions: DIMS,
    timeoutMs,
  });
}

function vectors(count: number, dims = DIMS): unknown {
  return { data: Array.from({ length: count }, (_, i) => ({ index: i, embedding: Array.from({ length: dims }, () => 1) })) };
}

function scriptFetch(steps: Step[]): { calls: number[]; fetchMock: ReturnType<typeof vi.fn> } {
  const calls: number[] = [];
  let i = 0;
  const fetchMock = vi.fn(async (_url: string, init: { signal: AbortSignal }) => {
    calls.push(Date.now());
    const step = steps[Math.min(i, steps.length - 1)];
    i++;
    if ("hang" in step) {
      return new Promise<Response>((_resolve, reject) => {
        init.signal.addEventListener("abort", () => {
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      });
    }
    return new Response(step.body === undefined ? "" : JSON.stringify(step.body), {
      status: step.status,
      headers: step.headers,
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** Run `p` while the fake clock drains every pending timer. */
async function drain<T>(p: Promise<T>): Promise<T> {
  const settled = p.then(
    (v) => ({ ok: true as const, v }),
    (e: unknown) => ({ ok: false as const, e }),
  );
  await vi.runAllTimersAsync();
  const r = await settled;
  if (r.ok) return r.v;
  throw r.e;
}

function gaps(calls: number[]): number[] {
  return calls.slice(1).map((t, k) => t - calls[k]);
}

afterEach(() => {
  vi.useRealTimers();
});

describe("embedding retry on 429/5xx (changed: was thrown on the first failure)", () => {
  it("429 then 500 then 200 returns the vector after three calls", async () => {
    vi.useFakeTimers();
    const { fetchMock } = scriptFetch([{ status: 429 }, { status: 500 }, { status: 200, body: vectors(1) }]);

    const vec = await drain(service().embed("hello"));

    expect(vec).toBeInstanceOf(Float32Array);
    expect(vec.length).toBe(DIMS);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("waits at least 500 ms then 1000 ms between the HTTP retries", async () => {
    vi.useFakeTimers();
    const { calls } = scriptFetch([{ status: 429 }, { status: 500 }, { status: 200, body: vectors(1) }]);

    await drain(service().embed("hello"));

    const [first, second] = gaps(calls);
    expect(first).toBeGreaterThanOrEqual(500);
    expect(second).toBeGreaterThanOrEqual(1000);
  });

  it("honours Retry-After but caps the wait at 1000 ms", async () => {
    vi.useFakeTimers();
    const { calls } = scriptFetch([
      { status: 429, headers: { "Retry-After": "30" } },
      { status: 200, body: vectors(1) },
    ]);

    await drain(service().embed("hello"));

    const [first] = gaps(calls);
    expect(first).toBeGreaterThanOrEqual(1000);
    expect(first).toBeLessThan(1500);
  });

  it("gives up after two retries and throws the last HTTP error", async () => {
    vi.useFakeTimers();
    const { fetchMock } = scriptFetch([{ status: 503 }]);

    await expect(drain(service().embed("hello"))).rejects.toThrow(/HTTP 503/);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});

describe("embedding retry on timeout (changed: was thrown on the first timeout)", () => {
  it("a call that times out is retried and the retry's vector is returned", async () => {
    const { fetchMock } = scriptFetch([{ hang: true }, { status: 200, body: vectors(1) }]);

    const vec = await service(20).embed("hello");

    expect(vec.length).toBe(DIMS);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("embedding errors that must not be retried (kept)", () => {
  it.each([401, 403])("HTTP %i throws on the first call", async (status) => {
    const { fetchMock } = scriptFetch([{ status, body: { error: "bad key" } }]);

    await expect(service().embed("hello")).rejects.toThrow(new RegExp(`HTTP ${status}`));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("a body without a data array throws", async () => {
    const { fetchMock } = scriptFetch([{ status: 200, body: { object: "list" } }]);

    await expect(service().embed("hello")).rejects.toThrow(/missing 'data' array/);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("embedding response validation (changed: was accepted)", () => {
  it("an empty data array throws", async () => {
    scriptFetch([{ status: 200, body: { data: [] } }]);

    await expect(service().embed("hello")).rejects.toThrow(/expected 1 embedding/);
  });

  it("fewer embeddings than inputs throws", async () => {
    scriptFetch([{ status: 200, body: vectors(1) }]);

    await expect(service().embedBatch(["a", "b"])).rejects.toThrow(/expected 2 embedding/);
  });

  it.each([
    ["duplicate", [0, 0]],
    ["out of range", [0, 5]],
  ])("a %s index throws", async (_label, indexes) => {
    const body = { data: indexes.map((index) => ({ index, embedding: [1, 1, 1, 1] })) };
    scriptFetch([{ status: 200, body }]);

    await expect(service().embedBatch(["a", "b"])).rejects.toThrow(/index/);
  });

  it("a vector whose length differs from the configured dimensions throws EmbeddingDimensionError", async () => {
    const { fetchMock } = scriptFetch([{ status: 200, body: vectors(1, 8) }]);

    const err = await service().embed("hello").catch((e: unknown) => e);

    expect(EmbeddingDimensionError).toBeTypeOf("function");
    expect(err).toBeInstanceOf(EmbeddingDimensionError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("embedding error message carries no response body (changed: had up to 500 chars)", () => {
  it("an HTTP error message names the status and the body length only", async () => {
    const secretish = "Incorrect API key provided: sk-abcdefghijklmnopqrstuvwxyz";
    scriptFetch([{ status: 401, body: { error: secretish } }]);

    const err = (await service().embed("hello").catch((e: unknown) => e)) as Error;

    expect(err.message).toContain("HTTP 401");
    expect(err.message).not.toContain("sk-abcdefghijklmnop");
    expect(err.message).toMatch(/body \d+ bytes/);
  });
});
