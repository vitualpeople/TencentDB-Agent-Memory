/**
 * Records how the standalone LLM runner talks to the provider today (cases
 * L1 and L2 of the memory-path failure table). Nothing here changes
 * behaviour; each test pins what the code does now.
 *
 *   L1  any call: POST {TDAI_LLM_BASE_URL}/chat/completions with the gateway
 *       model, max_tokens 4096 and a 120 s default timeout that a caller's
 *       timeoutMs overrides. maxRetries is not passed, so the AI SDK default
 *       decides how many times a 5xx is retried.
 *   L2  401 / 403 / 429 / 5xx / abort: one "run() failed after" error line,
 *       then the same error is rethrown to the caller.
 *
 * The provider is a fake `fetch` (L1) or a mocked `generateText` (L2); no
 * request leaves the process.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const sdk = vi.hoisted(() => ({ mockGenerate: false }));
vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  const generateText = vi.fn((...args: Parameters<typeof actual.generateText>) =>
    sdk.mockGenerate ? Promise.reject(new Error("generateText mock not primed")) : actual.generateText(...args));
  return { ...actual, generateText };
});

const { APICallError, generateText } = await import("ai");
const { StandaloneLLMRunner } = await import("../src/adapters/standalone/llm-runner.js");
const { loadGatewayConfig } = await import("../src/gateway/config.js");

const BASE_URL = "https://llm.invalid/v1";

function chatCompletion(text: string): Response {
  return new Response(JSON.stringify({
    id: "chatcmpl-1",
    object: "chat.completion",
    created: 1,
    model: "gateway-model",
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
  }), { status: 200, headers: { "content-type": "application/json" } });
}

function quietLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

afterEach(() => {
  sdk.mockGenerate = false;
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("L1 standalone LLM call shape", () => {
  it("L1 the gateway config maps TDAI_LLM_* to baseUrl/model with maxTokens 4096 and timeout 120 s, and splices it into memory.llm", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "l1-cfg-"));
    const file = path.join(dir, "tdai-gateway.yaml");
    fs.writeFileSync(file, "server:\n  port: 0\n", "utf-8");
    vi.stubEnv("TDAI_GATEWAY_CONFIG", file);
    vi.stubEnv("TDAI_LLM_BASE_URL", BASE_URL);
    vi.stubEnv("TDAI_LLM_MODEL", "gateway-model");
    vi.stubEnv("TDAI_LLM_API_KEY", "placeholder-not-a-secret");
    vi.stubEnv("TDAI_LLM_MAX_TOKENS", undefined);
    vi.stubEnv("TDAI_LLM_TIMEOUT_MS", undefined);
    vi.stubEnv("TDAI_LLM_PROVIDER", undefined);

    const cfg = loadGatewayConfig();

    expect(cfg.llm).toMatchObject({ baseUrl: BASE_URL, model: "gateway-model", maxTokens: 4096, timeoutMs: 120_000, provider: "openai" });
    expect(cfg.memory.llm).toMatchObject({ enabled: true, baseUrl: BASE_URL, model: "gateway-model", maxTokens: 4096, timeoutMs: 120_000 });
  });

  it("L1 one call is POST {baseUrl}/chat/completions with the configured model and max_tokens 4096, default timeout 120 s", async () => {
    const fetchMock = vi.fn(async () => chatCompletion("ok"));
    vi.stubGlobal("fetch", fetchMock);
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    const runner = new StandaloneLLMRunner({
      config: { baseUrl: BASE_URL, apiKey: "placeholder-not-a-secret", model: "gateway-model", maxTokens: 4096, timeoutMs: 120_000 },
      logger: quietLogger(),
    });

    await expect(runner.run({ prompt: "hi", taskId: "l1-shape" })).resolves.toBe("ok");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, { method: string; body: string }];
    expect(url).toBe(`${BASE_URL}/chat/completions`);
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body) as { model: string; max_tokens: number; tools?: unknown };
    expect(body.model).toBe("gateway-model");
    expect(body.max_tokens).toBe(4096);
    expect(body.tools).toBeUndefined();
    expect(timeoutSpy).toHaveBeenCalledWith(120_000);
  });

  it("L1 a caller's timeoutMs wins over the runner default (L1 extraction passes 180 s)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => chatCompletion("ok")));
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    const runner = new StandaloneLLMRunner({
      config: { baseUrl: BASE_URL, apiKey: "placeholder-not-a-secret", model: "gateway-model", timeoutMs: 120_000 },
      logger: quietLogger(),
    });

    await runner.run({ prompt: "hi", taskId: "l1-timeout", timeoutMs: 180_000 });

    expect(timeoutSpy).toHaveBeenCalledWith(180_000);
    expect(timeoutSpy).not.toHaveBeenCalledWith(120_000);
  });

  it("L1 maxRetries is not passed: the AI SDK default sends a 500 three times (1 + 2 retries) before run() throws", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ error: { message: "upstream down" } }), {
      status: 500,
      headers: { "content-type": "application/json", "retry-after-ms": "1" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runner = new StandaloneLLMRunner({
      config: { baseUrl: BASE_URL, apiKey: "placeholder-not-a-secret", model: "gateway-model" },
      logger: quietLogger(),
    });

    await expect(runner.run({ prompt: "hi", taskId: "l1-retries" })).rejects.toThrow();

    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});

describe("L2 provider errors are logged once and rethrown unchanged", () => {
  const httpCases: Array<[number, string]> = [
    [401, "Unauthorized"],
    [403, "Forbidden"],
    [429, "Too Many Requests"],
    [500, "Internal Server Error"],
    [503, "Service Unavailable"],
  ];

  it.each(httpCases)("L2 HTTP %i: run() logs status and url, then rethrows the same APICallError", async (status, message) => {
    sdk.mockGenerate = true;
    const err = new APICallError({
      message,
      url: `${BASE_URL}/chat/completions`,
      requestBodyValues: {},
      statusCode: status,
      responseBody: "{\"error\":\"x\"}",
    });
    vi.mocked(generateText).mockRejectedValueOnce(err);
    const logger = quietLogger();
    const runner = new StandaloneLLMRunner({ config: { baseUrl: BASE_URL, apiKey: "placeholder-not-a-secret", model: "m" }, logger });

    await expect(runner.run({ prompt: "hi", taskId: `l2-${status}` })).rejects.toBe(err);

    expect(logger.error).toHaveBeenCalledTimes(1);
    const line = logger.error.mock.calls[0][0] as string;
    expect(line).toMatch(/run\(\) failed after \d+ms: /);
    expect(line).toContain(`(status=${status}, url=${BASE_URL}/chat/completions`);
  });

  it("L2 abort: a caller abort is logged and rethrown as the AbortError", async () => {
    sdk.mockGenerate = true;
    const abort = Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
    vi.mocked(generateText).mockRejectedValueOnce(abort);
    const logger = quietLogger();
    const runner = new StandaloneLLMRunner({ config: { baseUrl: BASE_URL, apiKey: "placeholder-not-a-secret", model: "m" }, logger });
    const ctrl = new AbortController();
    ctrl.abort();

    await expect(runner.run({ prompt: "hi", taskId: "l2-abort", abortSignal: ctrl.signal })).rejects.toBe(abort);

    expect(logger.error).toHaveBeenCalledTimes(1);
    expect(logger.error.mock.calls[0][0]).toMatch(/run\(\) failed after \d+ms: This operation was aborted$/);
  });
});
