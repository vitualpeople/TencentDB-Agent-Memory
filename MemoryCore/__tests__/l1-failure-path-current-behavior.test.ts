/**
 * Records how the L1 failure path behaves today, so a later fix has a fixed
 * point to change against. Each block states one fact:
 *   (a) L1 groups of one slice interleave in recorded time, while the
 *       checkpoint keeps a single monotonic cursor per session key.
 *   (b) a timer-fired L1 task always carries triggeredBy "timer_scanner", and
 *       claimTimerL1Count skips it when conversation_count is 0.
 *   (c) the extractor turns an LLM error into an untyped success:false and an
 *       unparsable reply into [] (success:true); the runner advances the
 *       cursor past either.
 *   (d) moveToDeadLetter removes the session's L1_idle timer and nothing
 *       reads the dead-letter queue back.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

const hoisted = vi.hoisted(() => ({ captured: undefined as undefined | Record<string, never> }));
vi.mock("../src/core/state/index.js", async (importOriginal) => {
  const orig = await importOriginal<typeof import("../src/core/state/index.js")>();
  return {
    ...orig,
    createStateBackend: vi.fn(async (o: Record<string, never>) => {
      hoisted.captured = o;
      throw new Error("stop-after-state-backend");
    }),
  };
});

import { extractL1Memories } from "../src/core/record/l1-extractor.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import { buildPipelineTimerMember } from "../src/core/state/timer-member.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { PipelineWorker, claimTimerL1Count } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import {
  armedTimers, advance, gatewayHarness, makeCfg, makeL0Store, msg, providerError,
  sceneJson, scriptedLlm, silent, tmpDir, type L0Row,
} from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";

/** Two isolation groups whose rows alternate in recorded time: a1 < b1 < a2 < b2. */
function interleavedRows(): L0Row[] {
  return [
    { id: "L0-a1", sessionId: "sess-a", content: msg("alpha one"), timestamp: 1_000, recordedAtMs: 1_001 },
    { id: "L0-b1", sessionId: "sess-b", content: msg("bravo one"), timestamp: 1_100, recordedAtMs: 1_002 },
    { id: "L0-a2", sessionId: "sess-a", content: msg("alpha two"), timestamp: 1_200, recordedAtMs: 1_003 },
    { id: "L0-b2", sessionId: "sess-b", content: msg("bravo two"), timestamp: 1_300, recordedAtMs: 1_004 },
  ];
}

describe("(a) L1 groups interleave while the checkpoint has one monotonic cursor", () => {
  it("runs one extraction per group over interleaved rows and stores one scalar cursor at the slice max", async () => {
    const dir = tmpDir("l1-fact-a-");
    const { store } = makeL0Store(interleavedRows());
    const seen: string[][] = [];
    const { runner } = scriptedLlm((prompt) => {
      const ids = ["L0-a1", "L0-a2", "L0-b1", "L0-b2"].filter((id) => prompt.includes(id));
      seen.push(ids);
      return sceneJson(`group${seen.length}`, ids);
    });
    const run = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });

    await run({ sessionKey: SESSION });

    // One LLM call per group; each group spans a time range that overlaps the other.
    expect(seen).toEqual([["L0-a1", "L0-a2"], ["L0-b1", "L0-b2"]]);
    const cp = await new CheckpointManager(dir, silent).read();
    expect(Object.keys(cp.runner_states[SESSION]).sort()).toEqual(["last_captured_timestamp", "last_l1_cursor", "last_scene_name"]);
    expect(cp.runner_states[SESSION].last_l1_cursor).toBe(1_004);
  });

  it("never moves the cursor back: a lower mark is ignored", async () => {
    const dir = tmpDir("l1-fact-a2-");
    const cm = new CheckpointManager(dir, silent);
    await cm.markL1ExtractionComplete(SESSION, 0, 1_004);
    await cm.markL1ExtractionComplete(SESSION, 0, 1_002);
    expect((await cm.read()).runner_states[SESSION].last_l1_cursor).toBe(1_004);
  });
});

describe("(b) timer-fired L1 tasks are always timer_scanner and are skipped at count 0", () => {
  afterEach(() => { hoisted.captured = undefined; });

  it("the gateway's local timer callback enqueues L1_idle as triggeredBy timer_scanner", async () => {
    const { backend } = await gatewayHarness({
      TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend,
      instanceId: "svc", baseDir: tmpDir("l1-fact-b-"), runL1: async () => ({ storedCount: 0, hasMore: false, hasFullBacklog: false, profileScopes: [] }),
    });
    const member = buildPipelineTimerMember(SESSION, "L1_idle", { agentId: "default" });
    (hoisted.captured as unknown as { local: { onTimerExpired(e: unknown): void } }).local.onTimerExpired({ instanceId: "svc", member, fireAtMs: 0 });
    await Promise.resolve();

    const task = await backend.consumeTask("w", 0);
    expect(task?.type).toBe("L1");
    expect(task?.data?.triggeredBy).toBe("timer_scanner");
  });

  it("claimTimerL1Count returns true (skip) when conversation_count is 0, and the real executeL1 then never runs L1", async () => {
    const runL1 = vi.fn(async () => ({ storedCount: 0, hasMore: false, hasFullBacklog: false, profileScopes: [] }));
    const { backend, executor } = await gatewayHarness({
      TdaiGateway, captured: () => hoisted.captured as never, LocalStateBackendCtor: LocalStateBackend,
      instanceId: "svc", baseDir: tmpDir("l1-fact-b2-"), runL1,
    });
    await backend.updateSessionState("svc", SESSION, { conversation_count: 0 }, undefined, "default");
    const task: TaskPayload = {
      id: "L1-x", type: "L1", instanceId: "svc", sessionId: SESSION, agentId: "default", priority: 0, createdAt: Date.now(),
      data: { triggeredBy: "timer_scanner", instanceId: "svc", agentId: "default" },
    };

    expect(await claimTimerL1Count(backend, { ...task }, "svc", undefined, "default")).toBe(true);
    await executor.executeL1(task);
    expect(runL1).not.toHaveBeenCalled();
  });
});

describe("(c) the extractor swallows LLM and parse failures; the runner advances past them", () => {
  const base = { sessionKey: SESSION, sessionId: "s", baseDir: "", config: undefined, logger: silent };
  const messages = [{ id: "m1", role: "user" as const, content: msg("charlie"), timestamp: 1 }];

  it.each([401, 429, 500])("LLM error %s -> success:false with no error kind or message", async (status) => {
    const { runner } = scriptedLlm(() => { throw providerError(status); });
    const res = await extractL1Memories({ ...base, baseDir: tmpDir("l1-fact-c-"), messages, options: { enableDedup: false, llmRunner: runner } });
    expect(res).toEqual({ success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] });
  });

  it.each([
    ["prose, no JSON array", "Sorry, I cannot help with that."],
    ["broken JSON", "[{\"scene_name\": \"x\", \"memories\": [ {\"content\": "],
    ["array of a bare object", "[{}]"],
  ])("unparsable reply (%s) -> success:true with nothing extracted", async (_label, reply) => {
    const { runner } = scriptedLlm(() => reply);
    const res = await extractL1Memories({ ...base, baseDir: tmpDir("l1-fact-c2-"), messages, options: { enableDedup: false, llmRunner: runner } });
    expect(res.success).toBe(true);
    expect(res.extractedCount).toBe(0);
  });

  it("a group whose LLM call fails is skipped for good: the cursor still moves to the slice max", async () => {
    const dir = tmpDir("l1-fact-c3-");
    const { store, l1Records } = makeL0Store(interleavedRows());
    const { runner } = scriptedLlm((prompt) => {
      if (prompt.includes("L0-a1")) throw providerError(503);
      return sceneJson("bravo", ["L0-b1", "L0-b2"]);
    });
    const run = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });

    const out = await run({ sessionKey: SESSION });

    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-b"]);
    expect(out.processedCount).toBe(4);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(1_004);
  });
});

describe("(d) dead letter clears L1_idle and nothing reads the dead-letter queue back", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("after 1 + 3 failed runs the task is dead-lettered, no timer is left and no L1 runs again", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const fired: string[] = [];
    const backend = new LocalStateBackend({ onTimerExpired: (e) => { fired.push(e.member); } });
    let runs = 0;
    const worker = new PipelineWorker(backend, {
      async executeL1() { runs++; throw providerError(503); },
      async executeL2() {}, async executeL3() {},
    }, { concurrency: 1, pollIntervalMs: 200, lockRenewIntervalMs: 60_000 }, silent);
    const idle = buildPipelineTimerMember(SESSION, "L1_idle", { agentId: "default" });
    await backend.setTimer("svc", idle, Date.now() + 10 * 60_000);
    await worker.start();
    await backend.enqueueTask({ id: "L1-1", type: "L1", instanceId: "svc", sessionId: SESSION, agentId: "default", priority: 0, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } });

    await advance(5_000 + 15_000 + 45_000 + 5_000);
    expect(runs).toBe(4);
    expect(worker.getMetrics().tasksDeadLettered).toBe(1);
    expect(armedTimers(backend).size).toBe(0);

    await advance(2 * 60 * 60_000, 60_000);
    expect(runs).toBe(4);
    expect(fired).toEqual([]);
    await worker.stop();
  });

  it("no source file outside pipeline-worker.ts reads the dead-letter queue or wires onDeadLetter", () => {
    const srcDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../src");
    const hits: string[] = [];
    const walk = (d: string) => {
      for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, e.name);
        if (e.isDirectory()) walk(p);
        else if (p.endsWith(".ts") && !p.endsWith(".test.ts") && !p.endsWith("pipeline-worker.ts")) {
          const text = fs.readFileSync(p, "utf8");
          if (/getDeadLetterQueue\s*\(|onDeadLetter\s*:/.test(text)) hits.push(path.relative(srcDir, p));
        }
      }
    };
    walk(srcDir);
    expect(hits).toEqual([]);
  });
});
