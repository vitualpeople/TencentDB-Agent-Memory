/**
 * L3 persona: the total deadline of one generation, the persona / L3 retry
 * settings and their startup validation (cases F1, F2, G). Fake LLM, fake
 * Date (vi.useFakeTimers toFake Date only), no network.
 *
 *   F1 persona.timeoutMs is the total deadline: call 1 gets at most the
 *      budget, the nudge gets what is left; spent budget means no nudge and
 *      no final write; bad values make parseConfig throw.
 *   F2 the gateway reads TDAI_WORKER_L3_MAX_RETRIES (0..10, raw string
 *      checked) and passes the TDAI_PERSONA_* variables to cfg.memory.persona.
 *   G  an L3 task runs 1 + 1 times by default, L1 and offload-l1 keep 1 + 3.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { loadGatewayConfig } from "../src/gateway/config.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import {
  MemoryBackend,
  PERSONA,
  checkpointOf,
  fsFixture,
  l3For,
  scriptedRunner,
  silent,
  storageFixture,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

const T0 = new Date("2026-11-13T07:00:00.000Z");

function freezeClock(): void {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(T0);
}

function advance(ms: number): void {
  vi.setSystemTime(new Date(Date.now() + ms));
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

describe("Case F1 total deadline of one generation", () => {
  it("F1 timeoutMs=5000: call 1 gets 5000 ms, after 2000 ms spent the nudge gets the remaining 3000 ms", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    freezeClock();
    const { runner, calls } = scriptedRunner([
      () => { advance(2_000); return "no write"; },
      (p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nNew.`); return "ok"; },
    ]);

    const outcome = await l3For({ dir, runner, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined)
      .then(() => "resolved", (e: Error) => e.message);

    expect(calls[0].timeoutMs).toBe(5_000);
    expect(calls).toHaveLength(2);
    expect(calls[1].timeoutMs).toBe(3_000);
    // Time spent in call 1 plus the nudge's budget never exceeds the total deadline.
    expect(2_000 + calls[1].timeoutMs!).toBeLessThanOrEqual(5_000);
    expect(outcome).toBe("resolved");
  });

  it("F1 no persona config: the deadline is 180000 ms", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const dir = await fsFixture();
    freezeClock();
    const { runner, calls } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nNew.`); return "ok"; }]);

    await l3For({ dir, runner }).call(undefined);

    expect(parseConfig({}).persona.timeoutMs).toBe(180_000);
    expect(calls[0].timeoutMs).toBe(180_000);
  });

  it("F1 call 1 spends the whole budget without writing: no nudge, failed, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    freezeClock();
    const { runner, calls } = scriptedRunner([() => { advance(5_000); return "no write"; }]);

    await expect(l3For({ dir, runner, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined))
      .rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(1);
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });

  it("F1 the deadline passes during post-processing (storage reads after the runner): no final write, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const { dir, backend } = await storageFixture();
    freezeClock();
    // After the model wrote persona.md, the next read of it (post-processing) takes 6 s.
    let modelDone = false;
    const slow = new MemoryBackend();
    for (const [k, v] of backend.objects) slow.objects.set(k, v);
    const getObject = slow.getObject.bind(slow);
    slow.getObject = async (key: string) => {
      if (modelDone && key === "persona.md") { modelDone = false; advance(6_000); }
      return getObject(key);
    };
    const storage = new StorageAdapter(slow);
    const modelText = `${PERSONA}\n\nWritten by the model.`;
    const { runner, calls } = scriptedRunner([async () => {
      await storage.writeFile("persona.md", modelText);
      modelDone = true;
      return "ok";
    }]);

    await expect(l3For({ dir, runner, storage, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined))
      .rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(1);
    // The model's own write stays; the final write (with navigation) never happened.
    expect(slow.text("persona.md")).toBe(modelText);
    const cp = await checkpointOf(dir, storage);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });

  it.each([
    ["TDAI_PERSONA_TIMEOUT_MS", "abc"],
    ["TDAI_PERSONA_TIMEOUT_MS", "-5"],
    ["TDAI_PERSONA_TIMEOUT_MS", "999"],
    ["TDAI_PERSONA_TIMEOUT_MS", "600001"],
    ["TDAI_PERSONA_TIMEOUT_MS", "5000ms"],
    ["TDAI_PERSONA_MAX_CHARS_CHAT", "12k"],
    ["TDAI_PERSONA_MAX_CHARS_CHAT", "199"],
    ["TDAI_PERSONA_MAX_CHARS_CODE", "50001"],
    ["TDAI_PERSONA_MAX_CHARS_CODE", "-1"],
  ])("F1 parseConfig throws at startup on %s=%s", (name, value) => {
    vi.stubEnv(name, value);
    expect(() => parseConfig({})).toThrow(name);
  });

  it.each([
    [{ timeoutMs: -1 }],
    [{ timeoutMs: 999 }],
    [{ timeoutMs: "fast" }],
    [{ maxCharsChat: 50_001 }],
    [{ maxCharsCode: 100 }],
    [{ maxCharsChat: Number.NaN }],
  ])("F1 parseConfig throws at startup on YAML persona %j", (persona) => {
    expect(() => parseConfig({ persona })).toThrow(/memory\.persona\.(timeoutMs|maxCharsChat|maxCharsCode)/);
  });

  it("F1 env wins over YAML; valid values are kept", () => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", "5000");
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CHAT", "200");
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CODE", "50000");
    const cfg = parseConfig({ persona: { timeoutMs: 9_000, maxCharsChat: 7_000, maxCharsCode: 3_000 } });
    expect(cfg.persona).toMatchObject({ timeoutMs: 5_000, maxCharsChat: 200, maxCharsCode: 50_000 });
    vi.unstubAllEnvs();
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CHAT", undefined);
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CODE", undefined);
    expect(parseConfig({}).persona).toMatchObject({ timeoutMs: 180_000, maxCharsChat: 6_000, maxCharsCode: 6_000 });
  });
});

describe("Case F2 gateway: L3 retry setting and persona env pass-through", () => {
  function gatewayConfigFile(yaml = "server:\n  port: 0\n"): void {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "l3-gw-cfg-"));
    const file = path.join(dir, "tdai-gateway.yaml");
    fs.writeFileSync(file, yaml, "utf-8");
    vi.stubEnv("TDAI_GATEWAY_CONFIG", file);
    vi.stubEnv("TDAI_WORKER_L3_MAX_RETRIES", undefined);
    vi.stubEnv("WORKER_L3_MAX_RETRIES", undefined);
  }

  it("F2 default L3 retries is 1; YAML worker.l3MaxRetries and the env override are read", () => {
    gatewayConfigFile();
    expect(loadGatewayConfig().worker.l3MaxRetries).toBe(1);
    gatewayConfigFile("worker:\n  l3MaxRetries: 4\n");
    expect(loadGatewayConfig().worker.l3MaxRetries).toBe(4);
    vi.stubEnv("TDAI_WORKER_L3_MAX_RETRIES", "0");
    expect(loadGatewayConfig().worker.l3MaxRetries).toBe(0);
  });

  it.each(["abc", "2x", "1.5", "-1", "11"])("F2 TDAI_WORKER_L3_MAX_RETRIES=%s makes loadGatewayConfig throw", (value) => {
    gatewayConfigFile();
    vi.stubEnv("TDAI_WORKER_L3_MAX_RETRIES", value);
    expect(() => loadGatewayConfig()).toThrow(/L3_MAX_RETRIES|l3MaxRetries/);
  });

  it("F2 YAML worker.l3MaxRetries outside 0..10 makes loadGatewayConfig throw", () => {
    gatewayConfigFile("worker:\n  l3MaxRetries: 12\n");
    expect(() => loadGatewayConfig()).toThrow("worker.l3MaxRetries");
  });

  it("F2 the three TDAI_PERSONA_* variables reach cfg.memory.persona through loadGatewayConfig", () => {
    gatewayConfigFile();
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", "7000");
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CHAT", "3000");
    vi.stubEnv("TDAI_PERSONA_MAX_CHARS_CODE", "2500");
    expect(loadGatewayConfig().memory.persona).toMatchObject({ timeoutMs: 7_000, maxCharsChat: 3_000, maxCharsCode: 2_500 });
  });
});

describe("Case G L3 retry budget in the worker", () => {
  async function runUntilDeadLetter(type: "L3" | "L1" | "offload-l1"): Promise<{ attempts: number; worker: PipelineWorker }> {
    let attempts = 0;
    const fail = async () => { attempts++; throw new Error(`${type} always fails`); };
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, {
      executeL1: fail, executeL2: async () => {}, executeL3: fail, executeOffloadL1: fail,
    }, { concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000, retryBaseDelayMs: 5, deadLetterRetryDelayMs: 0 }, silent);
    await worker.start();
    try {
      await backend.enqueueTask({ id: `${type}-1`, type, instanceId: "svc", sessionId: "proj:g", agentId: "default", priority: 2, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } });
      await vi.waitFor(() => { expect(worker.getMetrics().tasksDeadLettered).toBe(1); }, { timeout: 10_000, interval: 10 });
    } finally {
      await worker.stop();
    }
    return { attempts, worker };
  }

  it("G an always-failing L3 task (locked path) runs 1 + 1 times, then dead-letters", async () => {
    const { attempts, worker } = await runUntilDeadLetter("L3");
    expect(attempts).toBe(2);
    expect(worker.getMetrics()).toMatchObject({ tasksRetried: 1, tasksFailed: 2 });
  });

  it("G an always-failing L1 task still runs 1 + 3 times", async () => {
    const { attempts } = await runUntilDeadLetter("L1");
    expect(attempts).toBe(4);
  });

  it("G an always-failing offload-l1 task (lock-free path) still runs 1 + 3 times", async () => {
    const { attempts } = await runUntilDeadLetter("offload-l1");
    expect(attempts).toBe(4);
  });
});
