/**
 * What counts as an LLM quota / rate-limit failure (utils/llm-quota-error.ts)
 * and how the standalone runner logs one: status and provider code only,
 * never the provider body (other HTTP errors keep their bounded body preview,
 * see llm-runner-api-call-error-log.test.ts).
 */
import { describe, expect, it, vi } from "vitest";

vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  return { ...actual, generateText: vi.fn() };
});

const { APICallError, RetryError, generateText } = await import("ai");
const { StandaloneLLMRunner } = await import("../src/adapters/standalone/llm-runner.js");
const { L1ExtractionError } = await import("../src/core/record/l1-extraction-error.js");
const { LlmQuotaError, describeLlmQuotaError, isLlmQuotaError } = await import("../src/utils/llm-quota-error.js");

const URL = "https://api.minimax.io/v1/chat/completions";
const SECRET = "PROVIDER-BODY-MUST-NOT-BE-LOGGED";

function apiError(statusCode: number, responseBody?: string) {
  return new APICallError({ message: "provider said no", url: URL, requestBodyValues: {}, statusCode, responseBody });
}

describe("isLlmQuotaError", () => {
  it("HTTP 429 on the error, in its cause chain, or as the AI SDK RetryError's last error", () => {
    const e429 = apiError(429);
    expect(isLlmQuotaError(e429)).toBe(true);
    expect(isLlmQuotaError(new L1ExtractionError("provider", "x", { cause: e429 }))).toBe(true);
    expect(isLlmQuotaError(new RetryError({ message: "Failed after 3 attempts", reason: "maxRetriesExceeded", errors: [apiError(503), e429] }))).toBe(true);
    expect(isLlmQuotaError(Object.assign(new Error("HTTP 429"), { status: 429 }))).toBe(true);
    expect(describeLlmQuotaError(e429)).toBe("status=429");
  });

  it("a MiniMax base_resp quota code, even on a 200 or another status", () => {
    const balance = apiError(200, JSON.stringify({ base_resp: { status_code: 1008, status_msg: SECRET } }));
    expect(isLlmQuotaError(balance)).toBe(true);
    expect(describeLlmQuotaError(balance)).toBe("status=200, provider_code=1008");
    expect(isLlmQuotaError(Object.assign(new Error("bad body"), { value: { base_resp: { status_code: 2056 } } }))).toBe(true);
  });

  it("not a quota error: other statuses, other MiniMax codes, parse failures, plain errors", () => {
    expect(isLlmQuotaError(apiError(500))).toBe(false);
    expect(isLlmQuotaError(apiError(401))).toBe(false);
    expect(isLlmQuotaError(apiError(400, JSON.stringify({ base_resp: { status_code: 2013 } })))).toBe(false);
    expect(isLlmQuotaError(new L1ExtractionError("parse", "No JSON array found"))).toBe(false);
    expect(isLlmQuotaError(new Error("timeout"))).toBe(false);
    expect(isLlmQuotaError(undefined)).toBe(false);
  });

  it("LlmQuotaError carries the signal and no body", () => {
    const err = new LlmQuotaError(apiError(429, SECRET), "L2 scene extraction");
    expect(isLlmQuotaError(err)).toBe(true);
    expect(err.message).toBe("LLM quota or rate limit (L2 scene extraction): status=429");
  });
});

describe("StandaloneLLMRunner log for a quota failure", () => {
  it("logs the status and url, not the provider message or body", async () => {
    const err = apiError(429, JSON.stringify({ base_resp: { status_code: 1002, status_msg: SECRET } }));
    vi.mocked(generateText).mockRejectedValueOnce(err);
    const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
    const runner = new StandaloneLLMRunner({ config: { baseUrl: "https://llm.invalid/v1", apiKey: "test-key", model: "test-model" }, logger });

    await expect(runner.run({ prompt: "hi", taskId: "t1" })).rejects.toBe(err);

    const line = logger.error.mock.calls[0][0] as string;
    expect(line).toContain(`LLM quota or rate limit (status=429, url=${URL})`);
    expect(line).not.toContain(SECRET);
    expect(line).not.toContain("provider said no");
  });
});
