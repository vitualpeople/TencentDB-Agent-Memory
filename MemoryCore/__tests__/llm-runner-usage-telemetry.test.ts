/**
 * Token telemetry of StandaloneLLMRunner.run(): every call ends with exactly
 * one INFO line
 *
 *   [standalone-runner] usage: taskId=..., outcome=ok|quota|timeout|abort|error,
 *     steps=..., inputTokens=..., outputTokens=..., totalTokens=..., ms=...
 *
 * summed over every finished step (AI SDK onStepFinish), so a multi-step tool
 * loop (L2 scene, L3 persona) reports its full cost, not only the last step.
 * A run that fails after some steps still logs what those steps used. The
 * line carries counts only: no prompt, no reply, no key. `runner.lastUsage`
 * holds the same totals for MetricTrackingRunner.
 *
 * Usage is read as AI SDK v6 `inputTokens`/`outputTokens`, falling back to
 * the older `promptTokens`/`completionTokens`.
 *
 * The provider is a fake `fetch` (OpenAI-compatible chat completions); no
 * request leaves the process.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

// generateText runs the real AI SDK unless a test scripts one call.
vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  return { ...actual, generateText: vi.fn(actual.generateText) };
});

const { generateText } = await import("ai");
const { StandaloneLLMRunner } = await import("../src/adapters/standalone/llm-runner.js");

const BASE_URL = "https://llm.invalid/v1";
const API_KEY = "placeholder-not-a-secret";
const PROMPT = "PROMPT-CONTENT-MUST-NOT-BE-LOGGED";
const REPLY = "REPLY-CONTENT-MUST-NOT-BE-LOGGED";
const USAGE_TAG = "[standalone-runner] usage:";

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

/** Step 1: the model calls the sandboxed `read` tool. */
function toolCallStep(promptTokens: number, completionTokens: number): Response {
  return json({
    id: "chatcmpl-1", object: "chat.completion", created: 1, model: "m",
    choices: [{
      index: 0, finish_reason: "tool_calls",
      message: {
        role: "assistant", content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "read", arguments: JSON.stringify({ path: "note.txt" }) } }],
      },
    }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  });
}

/** Last step: plain text. */
function textStep(promptTokens: number, completionTokens: number): Response {
  return json({
    id: "chatcmpl-2", object: "chat.completion", created: 2, model: "m",
    choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: REPLY } }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  });
}

/** A fetch that never answers until its signal aborts. */
function hangUntilAborted(init?: RequestInit): Promise<Response> {
  return new Promise((_resolve, reject) => {
    const signal = init?.signal;
    if (!signal) return;
    if (signal.aborted) { reject(signal.reason); return; }
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
}

/** Stub fetch: call N (1-based) gets `answer(N, init)`. */
function stubFetch(answer: (call: number, init?: RequestInit) => Response | Promise<Response>) {
  let calls = 0;
  const fetchMock = vi.fn(async (_url: unknown, init?: RequestInit) => answer(++calls, init));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function logger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function workspace(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "usage-telemetry-"));
  fs.writeFileSync(path.join(dir, "note.txt"), "note body", "utf-8");
  return dir;
}

function runner(log: ReturnType<typeof logger>, timeoutMs = 30_000) {
  return new StandaloneLLMRunner({
    config: { baseUrl: BASE_URL, apiKey: API_KEY, model: "m", timeoutMs },
    enableTools: true,
    logger: log,
  });
}

/** Every logged line (any level) that carries the usage tag. */
function usageLines(log: ReturnType<typeof logger>): string[] {
  return [log.debug, log.info, log.warn, log.error]
    .flatMap((fn) => fn.mock.calls.map((c) => String(c[0])))
    .filter((line) => line.includes(USAGE_TAG));
}

function field(line: string, name: string): string | undefined {
  return line.match(new RegExp(`${name}=([^,\\s]+)`))?.[1];
}

function expectNoContent(line: string): void {
  expect(line).not.toContain(PROMPT);
  expect(line).not.toContain(REPLY);
  expect(line).not.toContain(API_KEY);
  expect(line).not.toContain("note body");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(generateText).mockClear();
});

describe("StandaloneLLMRunner usage telemetry", () => {
  it("T1 two-step tool loop: one INFO usage line with outcome=ok, steps=2 and the summed tokens; lastUsage holds the totals", async () => {
    stubFetch((call) => (call === 1 ? toolCallStep(100, 20) : textStep(150, 30)));
    const log = logger();
    const r = runner(log);

    const out = await r.run({ prompt: PROMPT, taskId: "scene-extract", workspaceDir: workspace() });

    expect(out).toBe(REPLY);
    const lines = usageLines(log);
    expect(lines).toHaveLength(1);
    expect(log.info.mock.calls.map((c) => String(c[0])).filter((l) => l.includes(USAGE_TAG))).toHaveLength(1);
    const line = lines[0];
    expect(field(line, "taskId")).toBe("scene-extract");
    expect(field(line, "outcome")).toBe("ok");
    expect(field(line, "steps")).toBe("2");
    expect(field(line, "inputTokens")).toBe("250");
    expect(field(line, "outputTokens")).toBe("50");
    expect(field(line, "totalTokens")).toBe("300");
    expect(Number(field(line, "ms"))).toBeGreaterThanOrEqual(0);
    expectNoContent(line);
    expect(r.lastUsage).toEqual({ promptTokens: 250, completionTokens: 50, totalTokens: 300 });
  });

  it("T1 usage reported as promptTokens/completionTokens (pre-v6 shape) is summed over the steps too", async () => {
    vi.mocked(generateText).mockImplementationOnce((async (opts: { onStepFinish?: (s: unknown) => unknown }) => {
      await opts.onStepFinish?.({ usage: { promptTokens: 5, completionTokens: 2 } });
      await opts.onStepFinish?.({ usage: { promptTokens: 7, completionTokens: 3 } });
      return {
        text: "ok",
        steps: [{ stepNumber: 0, toolCalls: [], text: "" }, { stepNumber: 1, toolCalls: [], text: "ok" }],
        // Last step only, as the AI SDK reports `usage`.
        usage: { promptTokens: 7, completionTokens: 3 },
      };
    }) as never);
    const log = logger();
    const r = runner(log);

    await r.run({ prompt: PROMPT, taskId: "l1-extraction" });

    const lines = usageLines(log);
    expect(lines).toHaveLength(1);
    expect(field(lines[0], "steps")).toBe("2");
    expect(field(lines[0], "inputTokens")).toBe("12");
    expect(field(lines[0], "outputTokens")).toBe("5");
    expect(field(lines[0], "totalTokens")).toBe("17");
    expect(r.lastUsage).toEqual({ promptTokens: 12, completionTokens: 5, totalTokens: 17 });
  });

  it("T1b step 2 hits HTTP 429: one usage line with outcome=quota keeps step 1's tokens", async () => {
    stubFetch((call) => (call === 1
      ? toolCallStep(100, 20)
      : json({ error: { message: "rate limited" } }, 429, { "retry-after-ms": "1" })));
    const log = logger();
    const r = runner(log);

    await expect(r.run({ prompt: PROMPT, taskId: "persona-generation", workspaceDir: workspace() })).rejects.toBeDefined();

    const lines = usageLines(log);
    expect(lines).toHaveLength(1);
    expect(field(lines[0], "outcome")).toBe("quota");
    expect(field(lines[0], "steps")).toBe("1");
    expect(field(lines[0], "inputTokens")).toBe("100");
    expect(field(lines[0], "outputTokens")).toBe("20");
    expect(field(lines[0], "totalTokens")).toBe("120");
    expectNoContent(lines[0]);
    expect(r.lastUsage).toEqual({ promptTokens: 100, completionTokens: 20, totalTokens: 120 });
  }, 30_000);

  it("T1b step 2 runs past the runner timeout: one usage line with outcome=timeout keeps step 1's tokens", async () => {
    stubFetch((call, init) => (call === 1 ? toolCallStep(100, 20) : hangUntilAborted(init)));
    const log = logger();
    const r = runner(log, 300);

    await expect(r.run({ prompt: PROMPT, taskId: "scene-extract", workspaceDir: workspace() })).rejects.toBeDefined();

    const lines = usageLines(log);
    expect(lines).toHaveLength(1);
    expect(field(lines[0], "outcome")).toBe("timeout");
    expect(field(lines[0], "steps")).toBe("1");
    expect(field(lines[0], "totalTokens")).toBe("120");
    expectNoContent(lines[0]);
  }, 30_000);

  it("T1b the caller aborts during step 2 (lost lock): one usage line with outcome=abort keeps step 1's tokens", async () => {
    const caller = new AbortController();
    stubFetch((call, init) => {
      if (call === 1) return toolCallStep(100, 20);
      caller.abort(new Error("pipeline-worker: lock lost during execution"));
      return hangUntilAborted(init);
    });
    const log = logger();
    const r = runner(log);

    await expect(r.run({ prompt: PROMPT, taskId: "scene-extract", workspaceDir: workspace(), abortSignal: caller.signal })).rejects.toBeDefined();

    const lines = usageLines(log);
    expect(lines).toHaveLength(1);
    expect(field(lines[0], "outcome")).toBe("abort");
    expect(field(lines[0], "steps")).toBe("1");
    expect(field(lines[0], "inputTokens")).toBe("100");
    expect(field(lines[0], "outputTokens")).toBe("20");
    expectNoContent(lines[0]);
  }, 30_000);
});
