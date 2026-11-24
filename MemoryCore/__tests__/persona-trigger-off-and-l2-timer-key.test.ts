/**
 * Persona trigger mode "off" and the L2 timer member key.
 *
 *   T3a TDAI_PERSONA_TRIGGER_MODE=off (or YAML memory.persona.triggerMode:
 *       "off") is a valid mode: loadGatewayConfig and TdaiCore start without
 *       any Jev / OpenRouter key. Any other spelling ("OFF", "disabled")
 *       still fails at startup.
 *   T3b With "off" no L3 runs: the worker enqueues no L3 after an L2, an L3
 *       task left from before (queue or quota deferral file) finishes as a
 *       clean no-op in the gateway executor, and TdaiCore.runL3WithStore
 *       builds no L3 runner and makes no LLM / Jev request.
 *   T4  advanceL2TimerAfterL1 and armL2MaxInterval share one timer member
 *       (buildPipelineTimerMember(..., "L2_schedule", { teamId, agentId })):
 *       the max-interval timer never adds a second L2 timer, and the earlier
 *       +delay fire time is kept (setTimerIfEarlier).
 *
 * No network: the LLM endpoint is a stubbed fetch that only counts calls.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const l3Factory = vi.hoisted(() => ({ calls: 0 }));
vi.mock("../src/utils/pipeline-factory.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/utils/pipeline-factory.js")>();
  return {
    ...actual,
    // Count L3 runner constructions; the runner itself does nothing.
    createL3Runner: vi.fn(() => {
      l3Factory.calls++;
      return async () => {};
    }),
  };
});

const { loadGatewayConfig } = await import("../src/gateway/config.js");
const { parseConfig } = await import("../src/config.js");
const { TdaiCore } = await import("../src/core/tdai-core.js");
const { StandaloneHostAdapter } = await import("../src/adapters/standalone/host-adapter.js");
const { TdaiGateway } = await import("../src/gateway/server.js");
const { LocalStateBackend } = await import("../src/core/state/local-backend.js");
const { buildPipelineTimerMember } = await import("../src/core/state/timer-member.js");
const { StatefulPipelineManager } = await import("../src/utils/stateful-pipeline-manager.js");
const { armedTimers, silent } = await import("./helpers/l1-spike-harness.js");

type GatewayInternals = {
  config: { memory: { persona: { triggerMode: string } } };
  core: Record<string, unknown>;
  stateBackend: InstanceType<typeof LocalStateBackend>;
  pipelineWorker: { getMetrics(): { tasksCompleted: number } };
};

function tmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/** Gateway env with no Jev / OpenRouter key and a YAML file. */
function stubGatewayEnv(yaml: string): string {
  const dir = tmp("persona-off-cfg-");
  const file = path.join(dir, "tdai-gateway.yaml");
  fs.writeFileSync(file, yaml, "utf-8");
  vi.stubEnv("TDAI_GATEWAY_CONFIG", file);
  vi.stubEnv("TDAI_DATA_DIR", path.join(dir, "data"));
  vi.stubEnv("TDAI_LLM_BASE_URL", "https://llm.invalid/v1");
  vi.stubEnv("TDAI_LLM_MODEL", "m");
  vi.stubEnv("TDAI_LLM_API_KEY", "placeholder-not-a-secret");
  for (const name of ["TDAI_JEV_API_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "TDAI_JEV_ENDPOINT", "TDAI_JEV_MODEL", "TDAI_PERSONA_TRIGGER_MODE"]) {
    vi.stubEnv(name, undefined);
  }
  return dir;
}

/** Count every outbound request (LLM and Jev share global fetch). */
function countFetch() {
  const fetchMock = vi.fn(async () => new Response("{}", { status: 500 }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function waitFor(cond: () => boolean, maxMs = 5_000): Promise<boolean> {
  const until = Date.now() + maxMs;
  while (Date.now() < until) {
    if (cond()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return cond();
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  l3Factory.calls = 0;
});

describe("T3a persona trigger mode off is a valid startup mode", () => {
  it("T3a TDAI_PERSONA_TRIGGER_MODE=off: loadGatewayConfig and TdaiCore start with no Jev / OpenRouter key", async () => {
    stubGatewayEnv("server:\n  port: 0\n");
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", "off");

    let cfg: ReturnType<typeof loadGatewayConfig> | undefined;
    expect(() => { cfg = loadGatewayConfig(); }).not.toThrow();
    expect(cfg?.memory.persona.triggerMode).toBe("off");

    const dataDir = tmp("persona-off-core-");
    const hostAdapter = new StandaloneHostAdapter({ dataDir, llmConfig: cfg!.llm, logger: silent, platform: "gateway" });
    const core = new TdaiCore({ hostAdapter, config: cfg!.memory, instanceId: "default" });
    await expect(core.initialize()).resolves.toBeUndefined();
    await core.destroy();
  });

  it("T3a YAML memory.persona.triggerMode: \"off\" parses to off with no Jev key; \"OFF\" and \"disabled\" still throw", () => {
    stubGatewayEnv("server:\n  port: 0\nmemory:\n  persona:\n    triggerMode: \"off\"\n");

    let cfg: ReturnType<typeof loadGatewayConfig> | undefined;
    expect(() => { cfg = loadGatewayConfig(); }).not.toThrow();
    expect(cfg?.memory.persona.triggerMode).toBe("off");
    expect(parseConfig({ persona: { triggerMode: "off" } }).persona.triggerMode).toBe("off");

    // Any other spelling still fails at startup.
    for (const bad of ["OFF", "disabled"]) {
      vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", bad);
      expect(() => loadGatewayConfig()).toThrow(/triggerMode/);
      expect(() => parseConfig({ persona: { triggerMode: bad } })).toThrow(/triggerMode/);
    }
  });
});

describe("T3b with persona trigger mode off no L3 runs", () => {
  it("T3b gateway: an L2 run enqueues no L3, and a leftover L3 task ends as a no-op without runL3WithStore", async () => {
    stubGatewayEnv("server:\n  port: 0\n  host: 127.0.0.1\n");
    const gw = new TdaiGateway() as unknown as GatewayInternals & { start(): Promise<void>; stop(): Promise<void> };
    // Set after parsing so this case does not depend on the config parser.
    gw.config.memory.persona.triggerMode = "off";
    await gw.start();
    try {
      const runL2 = vi.fn(async () => ({ creditUsed: 0, skipped: false }));
      const runL3 = vi.fn(async () => ({ creditUsed: 0 }));
      gw.core.runL2WithStore = runL2;
      gw.core.runL3WithStore = runL3;
      const enqueue = vi.spyOn(gw.stateBackend, "enqueueTask");
      const base = { instanceId: "default", agentId: "default", priority: 1, data: { instanceId: "default", agentId: "default" } };

      await gw.stateBackend.enqueueTask({ ...base, id: "L2-t3b", type: "L2", sessionId: "proj:t3b", createdAt: Date.now() });
      expect(await waitFor(() => runL2.mock.calls.length === 1 && gw.pipelineWorker.getMetrics().tasksCompleted >= 1)).toBe(true);
      await new Promise((r) => setTimeout(r, 300));

      expect(enqueue.mock.calls.filter(([t]) => t.type === "L3")).toEqual([]);

      // An L3 task queued before the rollout (or re-armed from the deferral file).
      const done = gw.pipelineWorker.getMetrics().tasksCompleted;
      await gw.stateBackend.enqueueTask({ ...base, id: "L3-leftover", type: "L3", sessionId: "proj:t3b", priority: 2, createdAt: Date.now() });
      expect(await waitFor(() => gw.pipelineWorker.getMetrics().tasksCompleted === done + 1)).toBe(true);

      expect(runL3).not.toHaveBeenCalled();
    } finally {
      await gw.stop();
    }
  });

  it("T3b TdaiCore.runL3WithStore with triggerMode off builds no L3 runner and sends no LLM / Jev request", async () => {
    stubGatewayEnv("server:\n  port: 0\n");
    const fetchMock = countFetch();
    const cfg = loadGatewayConfig();
    cfg.memory.persona.triggerMode = "off" as never;
    const dataDir = tmp("persona-off-l3-");
    const hostAdapter = new StandaloneHostAdapter({ dataDir, llmConfig: cfg.llm, logger: silent, platform: "gateway" });
    const core = new TdaiCore({ hostAdapter, config: cfg.memory, instanceId: "default" });

    const result = await core.runL3WithStore({} as never, undefined, { l3TaskId: "L3-leftover", sessionId: "proj:t3b" } as never);

    expect(l3Factory.calls).toBe(0);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(result).toEqual({ creditUsed: 0 });
  });
});

describe("T4 L2 timers share one member key", () => {
  it("T4 advanceL2TimerAfterL1 (+300 s) then armL2MaxInterval (+21600 s): one L2 timer, still firing at +300 s", async () => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout"] });
    vi.setSystemTime(new Date("2026-11-24T00:00:00Z"));
    const now = Date.now();
    const backend = new LocalStateBackend({ onTimerExpired: () => {} });
    const manager = new StatefulPipelineManager({
      everyNConversations: 5,
      enableWarmup: false,
      l1: { idleTimeoutSeconds: 60 },
      l2: { delayAfterL1Seconds: 300, minIntervalSeconds: 0, maxIntervalSeconds: 21_600, sessionActiveWindowHours: 24 },
    }, backend, "svc", silent);
    const session = "proj:distributed-agent-mesh";

    await manager.advanceL2TimerAfterL1(session, "svc", "default", "default");
    await manager.armL2MaxInterval(session, "svc", "default", "default");

    const l2 = [...armedTimers(backend)].filter(([member]) => member.includes("L2_schedule"));
    expect(l2).toEqual([[buildPipelineTimerMember(session, "L2_schedule", { teamId: "default", agentId: "default" }), now + 300_000]]);
    await backend.destroy();
  });
});
