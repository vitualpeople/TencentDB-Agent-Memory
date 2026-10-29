import { describe, expect, it, vi } from "vitest";

vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  return { ...actual, generateText: vi.fn() };
});

const { APICallError, generateText } = await import("ai");
const { StandaloneLLMRunner } = await import("../src/adapters/standalone/llm-runner.js");

function makeRunner() {
  const logger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
  const runner = new StandaloneLLMRunner({
    config: { baseUrl: "https://llm.invalid/v1", apiKey: "test-key", model: "test-model" },
    logger,
  });
  return { runner, logger };
}

/**
 * A provider HTTP failure reaches run() as an APICallError whose message can
 * be as terse as "Not Found". The failure log must carry the status, url and
 * (bounded) response body so the failing upstream call is identifiable.
 */
describe("StandaloneLLMRunner failure log for APICallError", () => {
  it("appends status, url and a 2048-char body preview with the remainder count", async () => {
    const body = "x".repeat(3000);
    const err = new APICallError({
      message: "Not Found",
      url: "https://llm.invalid/v1/chat/completions",
      requestBodyValues: {},
      statusCode: 404,
      responseBody: body,
    });
    vi.mocked(generateText).mockRejectedValueOnce(err);
    const { runner, logger } = makeRunner();

    await expect(runner.run({ prompt: "hi", taskId: "t1" })).rejects.toBe(err);

    expect(logger.error).toHaveBeenCalledTimes(1);
    const line = logger.error.mock.calls[0][0] as string;
    expect(line).toContain("Not Found (status=404, url=https://llm.invalid/v1/chat/completions, body=");
    expect(line).toContain(`body=${"x".repeat(2048)}...(+952))`);
    expect(line).not.toContain("x".repeat(2049));
  });

  it("omits body= when the response body is empty", async () => {
    const err = new APICallError({
      message: "Bad Gateway",
      url: "https://llm.invalid/v1/chat/completions",
      requestBodyValues: {},
      statusCode: 502,
    });
    vi.mocked(generateText).mockRejectedValueOnce(err);
    const { runner, logger } = makeRunner();

    await expect(runner.run({ prompt: "hi", taskId: "t2" })).rejects.toBe(err);

    const line = logger.error.mock.calls[0][0] as string;
    expect(line).toMatch(/Bad Gateway \(status=502, url=https:\/\/llm\.invalid\/v1\/chat\/completions\)$/);
  });

  it("keeps the plain message for errors that are not APICallError", async () => {
    vi.mocked(generateText).mockRejectedValueOnce(new Error("socket hang up"));
    const { runner, logger } = makeRunner();

    await expect(runner.run({ prompt: "hi", taskId: "t3" })).rejects.toThrow("socket hang up");

    const line = logger.error.mock.calls[0][0] as string;
    expect(line).toMatch(/: socket hang up$/);
  });
});
