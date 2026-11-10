/**
 * Records the L3 Jev gate as it behaves today (cases L14, L15, L16 and L17
 * of the memory-path failure table). Recording only: the deferral and the
 * boot failure are pinned, not changed.
 *
 *   L14 one call per gate, no retry; default timeout 5 s; the key comes from
 *       TDAI_JEV_API_KEY, then TYPESAFE_API_KEY (default apiKeyEnv), then
 *       OPENROUTER_API_KEY.
 *   L15 non-2xx, timeout or a choice outside the labels -> kind
 *       "unavailable", shouldGenerate false, "deferring L3 without advancing
 *       persona checkpoint"; through the L3 runner no persona LLM call is
 *       made and last_persona_time does not move.
 *   L16 state above the input limit -> deferred with input_too_large, no
 *       call.
 *   L17 triggerMode=jev with no key in any of the envs -> config parse throws,
 *       so `new TdaiGateway()` (what main() runs first) throws and the
 *       gateway does not boot.
 *
 * The decision API is a fake `fetch`; no request leaves the process.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { evaluateL3JevGate } from "../src/core/persona/jev-gate.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import type { LLMRunner } from "../src/core/types.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL3Runner } from "../src/utils/pipeline-factory.js";

const KEY_ENVS = ["TDAI_JEV_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"] as const;

function spyLogger() {
  return { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function jevEnv(keys: Partial<Record<(typeof KEY_ENVS)[number], string>> = { OPENROUTER_API_KEY: "placeholder-or" }): void {
  vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", "jev");
  vi.stubEnv("TDAI_JEV_ENDPOINT", undefined);
  vi.stubEnv("TDAI_JEV_MODEL", undefined);
  vi.stubEnv("TDAI_JEV_TIMEOUT_MS", undefined);
  for (const k of KEY_ENVS) vi.stubEnv(k, keys[k]);
}

/** Persona body + one changed scene: the gate has something to ask about. */
async function personaFixture(personaBody = "# Persona\nBase rule.\n"): Promise<string> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jev-cases-"));
  fs.mkdirSync(path.join(dir, "scene_blocks"), { recursive: true });
  fs.writeFileSync(path.join(dir, "persona.md"), personaBody, "utf-8");
  await writeSceneIndex(dir, [{ filename: "coding.md", summary: "Updated coding rule", heat: 2, created: "2026-10-29T09:00:00.000Z", updated: "2026-10-29T10:00:00.000Z" }]);
  fs.writeFileSync(path.join(dir, "scene_blocks", "coding.md"),
    ["---", "scene: coding", "created: 2026-10-29T09:00:00.000Z", "updated: 2026-10-29T10:00:00.000Z", "heat: 2", "---", "", "## Summary", "Updated coding rule.", ""].join("\n"),
    "utf-8");
  return dir;
}

function decision(choice: string): Response {
  return new Response(JSON.stringify({
    model: "typesafe/jev-test",
    answers: { memory_change: { type: "choice", choice, probabilities: { [choice]: 0.9 }, confidence: 0.9 } },
  }), { status: 200, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("L14 Jev gate call", () => {
  it.each([
    ["all three set", { TDAI_JEV_API_KEY: "k-tdai", TYPESAFE_API_KEY: "k-typesafe", OPENROUTER_API_KEY: "k-or" }, "k-tdai"],
    ["TYPESAFE + OPENROUTER", { TYPESAFE_API_KEY: "k-typesafe", OPENROUTER_API_KEY: "k-or" }, "k-typesafe"],
    ["OPENROUTER only", { OPENROUTER_API_KEY: "k-or" }, "k-or"],
  ])("L14 key order (%s) -> %s", (_label, keys, expected) => {
    jevEnv(keys);
    expect(parseConfig({}).persona.jev.apiKey).toBe(expected);
  });

  it("L14 default timeout is 5000 ms", () => {
    jevEnv();
    expect(parseConfig({}).persona.jev.timeoutMs).toBe(5_000);
  });

  it("L14 a 503 is not retried: one fetch, then kind unavailable", async () => {
    jevEnv();
    const dir = await personaFixture();
    const fetcher = vi.fn(async () => new Response("Service Unavailable", { status: 503 }));

    const res = await evaluateL3JevGate({ dataDir: dir, cfg: parseConfig({}), logger: spyLogger(), triggerReason: "below_threshold", fetcher: fetcher as unknown as typeof fetch });

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(String((fetcher.mock.calls[0] as unknown[])[0])).toBe("https://openrouter.ai/api/alpha/decisions");
    expect(res).toMatchObject({ kind: "unavailable", shouldGenerate: false, called: true });
  });
});

describe("L15 Jev unavailable defers L3", () => {
  it.each([
    ["HTTP 401", () => Promise.resolve(new Response("{\"error\":\"unauthorized\"}", { status: 401 }))],
    ["HTTP 500", () => Promise.resolve(new Response("Internal Server Error", { status: 500 }))],
    ["timeout", () => Promise.reject(Object.assign(new Error("The operation timed out."), { name: "TimeoutError" }))],
    ["choice outside the labels", () => Promise.resolve(decision("banana"))],
  ])("L15 %s -> kind unavailable, shouldGenerate false, warning \"deferring L3 without advancing persona checkpoint\"", async (_label, answer) => {
    jevEnv();
    const dir = await personaFixture();
    const fetcher = vi.fn(answer);
    const logger = spyLogger();

    const res = await evaluateL3JevGate({ dataDir: dir, cfg: parseConfig({}), logger, triggerReason: "below_threshold", fetcher: fetcher as unknown as typeof fetch });

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(res).toMatchObject({ kind: "unavailable", shouldGenerate: false });
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("deferring L3 without advancing persona checkpoint"));
  });

  it("L15 through the L3 runner: Jev 500 -> no persona LLM call, nothing thrown, last_persona_time unchanged", async () => {
    jevEnv();
    const dir = await personaFixture();
    vi.stubGlobal("fetch", vi.fn(async () => new Response("Internal Server Error", { status: 500 })));
    const run = vi.fn(async () => "should not be called");
    const l3 = createL3Runner({ pluginDataDir: dir, cfg: parseConfig({}), openclawConfig: undefined, logger: spyLogger(), llmRunner: { run } as unknown as LLMRunner });
    const before = await new CheckpointManager(dir, undefined).read();

    await expect(l3()).resolves.toBeUndefined();

    expect(run).not.toHaveBeenCalled();
    const after = await new CheckpointManager(dir, undefined).read();
    expect(after.last_persona_time).toBe(before.last_persona_time);
    expect(after.last_persona_at).toBe(before.last_persona_at);
  });
});

describe("L16 Jev input too large", () => {
  it("L16 a persona above the input limit -> deferred with reason input_too_large, no call", async () => {
    jevEnv();
    const dir = await personaFixture("# Persona\n" + "X".repeat(33_000));
    const fetcher = vi.fn();
    const logger = spyLogger();

    const res = await evaluateL3JevGate({ dataDir: dir, cfg: parseConfig({}), logger, triggerReason: "below_threshold", fetcher: fetcher as unknown as typeof fetch });

    expect(res).toMatchObject({ kind: "unavailable", shouldGenerate: false, reason: "input_too_large", called: false });
    expect(fetcher).not.toHaveBeenCalled();
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("input too large"));
  });
});

describe("L17 triggerMode=jev with no key", () => {
  it("L17 parseConfig throws: triggerMode \"jev\" requires memory.persona.jev apiKey/TYPESAFE_API_KEY", () => {
    jevEnv({});
    expect(() => parseConfig({})).toThrow('memory.persona.triggerMode="jev" requires memory.persona.jev apiKey/TYPESAFE_API_KEY');
  });

  it("L17 the gateway constructor (first step of main) throws the same error, so the gateway does not boot", () => {
    jevEnv({});
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "jev-boot-"));
    const file = path.join(dir, "tdai-gateway.yaml");
    fs.writeFileSync(file, "server:\n  port: 0\n", "utf-8");
    vi.stubEnv("TDAI_GATEWAY_CONFIG", file);

    expect(() => new TdaiGateway()).toThrow('memory.persona.triggerMode="jev" requires');
  });

  it("L17 control: the same boot with one key set does not throw at config parse", () => {
    jevEnv({ OPENROUTER_API_KEY: "placeholder-or" });
    expect(() => parseConfig({})).not.toThrow();
  });
});
