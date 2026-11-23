/**
 * TDAI_LLM_REASONING_EFFORT controls the `reasoning_effort` field of every
 * chat request the standalone runner sends, on every path that builds one:
 * the runner itself, the gateway host adapter, TdaiCore.resolveRuntimeLlm,
 * TdaiCore.buildSkillLlmRunner, the seed runtime and the gateway's
 * per-instance skill extractor.
 *
 *   - unset or empty: the request body has no `reasoning_effort` key, exactly
 *     as before the variable existed;
 *   - low | medium | high | xhigh | max: the body carries that value and
 *     `max_tokens` is unchanged;
 *   - anything else (including upper case): loadGatewayConfig throws, naming
 *     the variable and the accepted values.
 *
 * The provider is a fake `fetch` whose reply mirrors the shape an
 * OpenAI-compatible MiniMax endpoint returned in a live probe (top-level
 * base_resp / *_sensitive keys, message.reasoning_content, usage details).
 * No request leaves the process.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const seedCapture = vi.hoisted(() => ({ l1LlmRunner: undefined as unknown }));
vi.mock("../src/utils/pipeline-factory.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils/pipeline-factory.js")>();
  return {
    ...actual,
    // Capture the L1 runner the seed runtime built, then stop the seed run:
    // only the runner construction is under test here.
    createPipeline: vi.fn(async (opts: { l1LlmRunner?: unknown }) => {
      seedCapture.l1LlmRunner = opts.l1LlmRunner;
      throw new Error("seed stopped by test after runner construction");
    }),
  };
});

const { StandaloneLLMRunner, StandaloneLLMRunnerFactory } = await import("../src/adapters/standalone/llm-runner.js");
const { StandaloneHostAdapter } = await import("../src/adapters/standalone/host-adapter.js");
const { loadGatewayConfig } = await import("../src/gateway/config.js");
const { parseConfig, LLM_REASONING_EFFORTS } = await import("../src/config.js");
const { TdaiCore } = await import("../src/core/tdai-core.js");
const { executeSeed } = await import("../src/core/seed/seed-runtime.js");
const { TdaiGateway } = await import("../src/gateway/server.js");

const BASE_URL = "https://llm.invalid/v1";
const MODEL = "MiniMax-M3.1-Flash";
const API_KEY = "placeholder-not-a-secret";
const EFFORTS = ["low", "medium", "high", "xhigh", "max"] as const;

/** Reply shaped like the MiniMax OpenAI-compatible endpoint's chat completion. */
function minimaxCompletion(text: string): Response {
  return new Response(JSON.stringify({
    id: "chatcmpl-probe",
    object: "chat.completion",
    created: 1,
    model: MODEL,
    choices: [{
      index: 0,
      finish_reason: "stop",
      message: { role: "assistant", content: text, reasoning_content: "short reasoning" },
    }],
    usage: {
      prompt_tokens: 340,
      completion_tokens: 87,
      total_tokens: 427,
      completion_tokens_details: { reasoning_tokens: 5 },
      prompt_tokens_details: { cached_tokens: 128 },
    },
    base_resp: { status_code: 0, status_msg: "" },
    input_sensitive: false,
    input_sensitive_type: 0,
    output_sensitive: false,
    output_sensitive_int: 0,
    output_sensitive_type: 0,
  }), { status: 200, headers: { "content-type": "application/json" } });
}

type RequestBody = Record<string, unknown> & { model?: string; max_tokens?: number };

/** Stub global fetch with the fixture; returns a getter for the sent bodies. */
function captureFetch(): () => RequestBody[] {
  const fetchMock = vi.fn(async () => minimaxCompletion("[]"));
  vi.stubGlobal("fetch", fetchMock);
  return () => fetchMock.mock.calls.map((call) => {
    const init = (call as unknown as [string, { body: string }])[1];
    return JSON.parse(init.body) as RequestBody;
  });
}

function quietLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

/** Point loadGatewayConfig at an empty YAML and set the TDAI_LLM_* env. */
function stubGatewayEnv(effort: string | undefined, yaml = "server:\n  port: 0\n"): void {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "reasoning-effort-cfg-"));
  const file = path.join(dir, "tdai-gateway.yaml");
  fs.writeFileSync(file, yaml, "utf-8");
  vi.stubEnv("TDAI_GATEWAY_CONFIG", file);
  vi.stubEnv("TDAI_LLM_BASE_URL", BASE_URL);
  vi.stubEnv("TDAI_LLM_MODEL", MODEL);
  vi.stubEnv("TDAI_LLM_API_KEY", API_KEY);
  vi.stubEnv("TDAI_LLM_MAX_TOKENS", "16384");
  vi.stubEnv("TDAI_LLM_TIMEOUT_MS", undefined);
  vi.stubEnv("TDAI_LLM_PROVIDER", undefined);
  vi.stubEnv("TDAI_LLM_REASONING_EFFORT", effort);
}

/** TdaiCore built the way the gateway builds it, from loadGatewayConfig(). */
function coreFromGatewayConfig() {
  const cfg = loadGatewayConfig();
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "reasoning-effort-core-"));
  const hostAdapter = new StandaloneHostAdapter({ dataDir, llmConfig: cfg.llm, logger: quietLogger(), platform: "gateway" });
  const core = new TdaiCore({ hostAdapter, config: cfg.memory, instanceId: "default" });
  return { cfg, hostAdapter, core };
}

type PrivateCore = {
  resolveRuntimeLlm(): import("../src/adapters/standalone/llm-runner.js").StandaloneLLMConfig;
  buildSkillLlmRunner(): { run(params: { prompt: string; taskId: string }): Promise<string> } | undefined;
};

afterEach(() => {
  seedCapture.l1LlmRunner = undefined;
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("StandaloneLLMRunner request body", () => {
  it("without reasoningEffort the body has no reasoning_effort key", async () => {
    const bodies = captureFetch();
    const runner = new StandaloneLLMRunner({
      config: { baseUrl: BASE_URL, apiKey: API_KEY, model: MODEL, maxTokens: 16384 },
      logger: quietLogger(),
    });

    await runner.run({ prompt: "hi", taskId: "effort-unset" });

    expect(bodies()).toHaveLength(1);
    expect(bodies()[0]).not.toHaveProperty("reasoning_effort");
    expect(bodies()[0].max_tokens).toBe(16384);
  });

  it.each(EFFORTS)("reasoningEffort=%s is sent as reasoning_effort with max_tokens unchanged", async (effort) => {
    const bodies = captureFetch();
    const runner = new StandaloneLLMRunner({
      config: { baseUrl: BASE_URL, apiKey: API_KEY, model: MODEL, maxTokens: 16384, reasoningEffort: effort },
      logger: quietLogger(),
    });

    await expect(runner.run({ prompt: "hi", taskId: `effort-${effort}` })).resolves.toBe("[]");

    expect(bodies()).toHaveLength(1);
    expect(bodies()[0].reasoning_effort).toBe(effort);
    expect(bodies()[0].max_tokens).toBe(16384);
    expect(bodies()[0]).not.toHaveProperty("max_completion_tokens");
    expect(bodies()[0].model).toBe(MODEL);
  });
});

describe("TDAI_LLM_REASONING_EFFORT in loadGatewayConfig", () => {
  it("unset leaves cfg.llm and cfg.memory.llm without reasoningEffort", () => {
    stubGatewayEnv(undefined);
    const cfg = loadGatewayConfig();
    expect(cfg.llm.reasoningEffort).toBeUndefined();
    expect(cfg.memory.llm.reasoningEffort).toBeUndefined();
  });

  it("empty string means unset", () => {
    stubGatewayEnv("");
    const cfg = loadGatewayConfig();
    expect(cfg.llm.reasoningEffort).toBeUndefined();
    expect(cfg.memory.llm.reasoningEffort).toBeUndefined();
  });

  it.each(EFFORTS)("%s reaches cfg.llm and the memory.llm splice", (effort) => {
    stubGatewayEnv(effort);
    const cfg = loadGatewayConfig();
    expect(cfg.llm.reasoningEffort).toBe(effort);
    expect(cfg.memory.llm).toMatchObject({ enabled: true, reasoningEffort: effort, maxTokens: 16384 });
  });

  it.each(["LOW", "foo", "minimal", "none"])("%s fails startup naming the variable and the accepted values", (value) => {
    stubGatewayEnv(value);
    expect(() => loadGatewayConfig()).toThrow("TDAI_LLM_REASONING_EFFORT must be one of low|medium|high|xhigh|max");
  });

  it("YAML llm.reasoningEffort applies when the env is unset, and the env wins when both are set", () => {
    stubGatewayEnv(undefined, "server:\n  port: 0\nllm:\n  reasoningEffort: medium\n");
    expect(loadGatewayConfig().llm.reasoningEffort).toBe("medium");
    vi.stubEnv("TDAI_LLM_REASONING_EFFORT", "low");
    expect(loadGatewayConfig().llm.reasoningEffort).toBe("low");
  });

  it("an invalid YAML llm.reasoningEffort fails startup naming the key", () => {
    stubGatewayEnv(undefined, "server:\n  port: 0\nllm:\n  reasoningEffort: High\n");
    expect(() => loadGatewayConfig()).toThrow("llm.reasoningEffort must be one of low|medium|high|xhigh|max");
  });

  it("the accepted list is exactly the five lowercase values", () => {
    expect([...LLM_REASONING_EFFORTS]).toEqual([...EFFORTS]);
  });
});

describe("every runner construction path carries reasoningEffort into the request", () => {
  it("gateway host adapter factory (the default L1/L2/L3 runner) sends reasoning_effort=low", async () => {
    stubGatewayEnv("low");
    const { hostAdapter } = coreFromGatewayConfig();
    const bodies = captureFetch();

    await hostAdapter.getLLMRunnerFactory().createRunner({ enableTools: false }).run({ prompt: "hi", taskId: "host" });

    expect(bodies()[0].reasoning_effort).toBe("low");
    expect(bodies()[0].max_tokens).toBe(16384);
  });

  it("TdaiCore.resolveRuntimeLlm keeps reasoningEffort, and a factory built from it sends it", async () => {
    stubGatewayEnv("high");
    const { core } = coreFromGatewayConfig();
    const runtimeLlm = (core as unknown as PrivateCore).resolveRuntimeLlm();
    expect(runtimeLlm.reasoningEffort).toBe("high");
    const bodies = captureFetch();

    await new StandaloneLLMRunnerFactory({ config: runtimeLlm, logger: quietLogger() })
      .createRunner({ enableTools: false })
      .run({ prompt: "hi", taskId: "runtime-llm" });

    expect(bodies()[0].reasoning_effort).toBe("high");
    expect(bodies()[0].max_tokens).toBe(16384);
  });

  it("TdaiCore.resolveRuntimeLlm without the variable leaves the request without reasoning_effort", async () => {
    stubGatewayEnv(undefined);
    const { core } = coreFromGatewayConfig();
    const runtimeLlm = (core as unknown as PrivateCore).resolveRuntimeLlm();
    const bodies = captureFetch();

    await new StandaloneLLMRunnerFactory({ config: runtimeLlm, logger: quietLogger() })
      .createRunner({ enableTools: false })
      .run({ prompt: "hi", taskId: "runtime-llm-unset" });

    expect(bodies()[0]).not.toHaveProperty("reasoning_effort");
  });

  it("TdaiCore.buildSkillLlmRunner sends reasoning_effort=medium", async () => {
    stubGatewayEnv("medium");
    const { core } = coreFromGatewayConfig();
    const runner = (core as unknown as PrivateCore).buildSkillLlmRunner();
    expect(runner).toBeDefined();
    const bodies = captureFetch();

    await runner!.run({ prompt: "hi", taskId: "skill-core" });

    expect(bodies()[0].reasoning_effort).toBe("medium");
  });

  it("seed runtime builds its L1 runner with memory.llm.reasoningEffort", async () => {
    await expect(executeSeed(
      { sessions: [], totalRounds: 0, totalMessages: 0, hasTimestamps: false },
      {
        outputDir: fs.mkdtempSync(path.join(os.tmpdir(), "reasoning-effort-seed-")),
        openclawConfig: undefined,
        pluginConfig: { llm: { enabled: true, baseUrl: BASE_URL, apiKey: API_KEY, model: MODEL, maxTokens: 16384, reasoningEffort: "xhigh" } },
        logger: quietLogger(),
      },
    )).rejects.toThrow("seed stopped by test");
    expect(seedCapture.l1LlmRunner).toBeDefined();
    const bodies = captureFetch();

    await (seedCapture.l1LlmRunner as { run(p: { prompt: string; taskId: string }): Promise<string> })
      .run({ prompt: "hi", taskId: "seed" });

    expect(bodies()[0].reasoning_effort).toBe("xhigh");
    expect(bodies()[0].max_tokens).toBe(16384);
  });

  it("gateway per-instance skill extractor sends reasoning_effort=max", async () => {
    stubGatewayEnv("max");
    const cfg = loadGatewayConfig();
    const fakeGateway = {
      config: cfg,
      logger: quietLogger(),
      core: { getResolvedSkillConfig: () => undefined },
    };
    const build = (TdaiGateway.prototype as unknown as {
      buildSkillExtractorForInstance(this: unknown, skillCore: unknown, instanceId: string): Promise<unknown>;
    }).buildSkillExtractorForInstance;
    const extractor = await build.call(fakeGateway, {}, "default");
    const runner = (extractor as { runner: { run(p: { prompt: string; taskId: string }): Promise<string> } }).runner;
    const bodies = captureFetch();

    await runner.run({ prompt: "hi", taskId: "skill-extractor" });

    expect(bodies()[0].reasoning_effort).toBe("max");
  });
});

describe("memory.llm.reasoningEffort in parseConfig", () => {
  it("is undefined when absent or empty, and kept when valid", () => {
    expect(parseConfig({ llm: {} }).llm.reasoningEffort).toBeUndefined();
    expect(parseConfig({ llm: { reasoningEffort: "" } }).llm.reasoningEffort).toBeUndefined();
    expect(parseConfig({ llm: { reasoningEffort: "low" } }).llm.reasoningEffort).toBe("low");
  });

  it.each(["LOW", "foo"])("%s throws naming memory.llm.reasoningEffort", (value) => {
    expect(() => parseConfig({ llm: { reasoningEffort: value } })).toThrow("memory.llm.reasoningEffort must be one of");
  });
});
