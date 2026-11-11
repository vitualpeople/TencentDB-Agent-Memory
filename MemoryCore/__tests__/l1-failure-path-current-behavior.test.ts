/**
 * The L1 failure path, one fact per block. (a) and (b) are unchanged facts;
 * (c) and (d) used to record the swallow-and-skip behaviour and now pin the
 * fixed one:
 *   (a) L1 groups of one slice interleave in recorded time, while the
 *       checkpoint keeps a single monotonic cursor per session key.
 *   (b) a timer-fired L1 task (L1_idle) carries triggeredBy "timer_scanner",
 *       and claimTimerL1Count skips it when conversation_count is 0.
 *   (c) an LLM error or an unparsable reply is an L1ExtractionError (kind
 *       "provider" / "parse"); the runner throws it and does not move the
 *       cursor past the failed rows. A valid empty array is still a success.
 *   (d) moveToDeadLetter removes the session's L1_idle timer and, when the
 *       last failure is a typed L1ExtractionError, arms an L1_dlq timer
 *       (default 15 min) that re-runs the slice.
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

import { L1ExtractionError } from "../src/core/record/l1-extraction-error.js";
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

describe("(c) LLM and parse failures are typed errors; the runner keeps the cursor", () => {
  const base = { sessionKey: SESSION, sessionId: "s", baseDir: "", config: undefined, logger: silent };
  const messages = [{ id: "m1", role: "user" as const, content: msg("charlie"), timestamp: 1 }];
  const failures: Array<[string, "provider" | "parse", () => string]> = [
    ["LLM error 401", "provider", () => { throw providerError(401); }],
    ["LLM error 429", "provider", () => { throw providerError(429); }],
    ["LLM error 500", "provider", () => { throw providerError(500); }],
    ["abort", "provider", () => { throw Object.assign(new Error("The operation was aborted"), { name: "AbortError" }); }],
    ["timeout", "provider", () => { throw Object.assign(new Error("LLM request timed out after 120000ms"), { name: "TimeoutError" }); }],
    ["prose, no JSON array", "parse", () => "Sorry, I cannot help with that."],
    ["broken JSON", "parse", () => "[{\"scene_name\": \"x\", \"memories\": [ {\"content\": "],
    ["JSON cut mid-array", "parse", () => "[{\"scene_name\":\"x\",\"message_ids\":[\"m1\"],\"memories\":[{\"content\":\"a\"}]},{\"scene_"],
    ["array of a bare object", "parse", () => "[{}]"],
  ];
  const sessionRows = (): L0Row[] => [
    { id: "L0-c1", sessionId: "sess-c", content: msg("charlie one"), timestamp: 10, recordedAtMs: 601 },
    { id: "L0-c2", sessionId: "sess-c", content: msg("charlie two"), timestamp: 20, recordedAtMs: 602 },
  ];

  it.each(failures)("%s -> extractor throws L1ExtractionError kind %s", async (_label, kind, reply) => {
    const { runner } = scriptedLlm(reply);
    const err = await extractL1Memories({ ...base, baseDir: tmpDir("l1-fact-c-"), messages, options: { enableDedup: false, llmRunner: runner } })
      .then(() => undefined, (e: unknown) => e);
    expect(err).toBeInstanceOf(L1ExtractionError);
    expect((err as L1ExtractionError).kind).toBe(kind);
  });

  it.each(failures)("%s -> runner throws kind %s and the session cursor does not move", async (_label, kind, reply) => {
    const dir = tmpDir("l1-fact-c2-");
    await new CheckpointManager(dir, silent).markL1ExtractionComplete(SESSION, 0, 500);
    const { store, l1Records } = makeL0Store(sessionRows());
    const { runner } = scriptedLlm(reply);
    const run = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });

    const err = await run({ sessionKey: SESSION }).then(() => undefined, (e: unknown) => e);

    expect(err).toBeInstanceOf(L1ExtractionError);
    expect((err as L1ExtractionError).kind).toBe(kind);
    expect(l1Records).toHaveLength(0);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(500);
  });

  // A valid empty array and a valid scene still advance the cursor:
  // l1-success-and-empty-array-advance-cursor.test.ts (kept free of the error
  // module so it runs unchanged on the code before the fix).

  it("a group whose LLM call fails is kept: the cursor does not pass it and the runner throws", async () => {
    const dir = tmpDir("l1-fact-c4-");
    const { store, l1Records } = makeL0Store(interleavedRows());
    const { runner } = scriptedLlm((prompt) => {
      if (prompt.includes("L0-a1")) throw providerError(503);
      return sceneJson("bravo", ["L0-b1", "L0-b2"]);
    });
    const run = createL1Runner({ pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined, logger: silent, llmRunner: runner });

    await expect(run({ sessionKey: SESSION })).rejects.toBeInstanceOf(L1ExtractionError);

    expect(l1Records.map((r) => r.sessionId)).toEqual(["sess-b"]);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(0);
  });
});

describe("(d) dead letter clears L1_idle and arms a delayed L1_dlq re-run", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("after 1 + 3 failed runs the task is dead-lettered, L1_idle is gone and L1_dlq fires 15 min later by default", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const fired: string[] = [];
    const backend = new LocalStateBackend({ onTimerExpired: (e) => { fired.push(e.member); } });
    let runs = 0;
    const worker = new PipelineWorker(backend, {
      async executeL1() { runs++; throw new L1ExtractionError("provider", "LLM API call failed: HTTP 503"); },
      async executeL2() {}, async executeL3() {},
    }, { concurrency: 1, pollIntervalMs: 200, lockRenewIntervalMs: 60_000 }, silent);
    const idle = buildPipelineTimerMember(SESSION, "L1_idle", { agentId: "default" });
    const dlq = buildPipelineTimerMember(SESSION, "L1_dlq", { agentId: "default" });
    await backend.setTimer("svc", idle, Date.now() + 10 * 60_000);
    await worker.start();
    await backend.enqueueTask({ id: "L1-1", type: "L1", instanceId: "svc", sessionId: SESSION, agentId: "default", priority: 0, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } });

    await advance(5_000 + 15_000 + 45_000 + 5_000);
    expect(runs).toBe(4);
    expect(worker.getMetrics().tasksDeadLettered).toBe(1);
    const timers = armedTimers(backend);
    expect(timers.has(idle)).toBe(false);
    expect(timers.get(dlq)).toBe(worker.getDeadLetterQueue()[0].deadAt + 15 * 60_000);

    await advance(15 * 60_000, 60_000);
    expect(fired).toEqual([dlq]);
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
