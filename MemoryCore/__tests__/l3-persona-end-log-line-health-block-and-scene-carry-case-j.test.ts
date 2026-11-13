/**
 * L3 persona: the single end-of-generation log line, the persona counters on
 * /health, and case J (scenes of a failed generation reach the next one).
 * createL3Runner runs for real in a temp dir; only the LLM is a fake.
 *
 * End line (one per scope that reached the generator, emitted from the
 * runner's `finally`):
 *   persona_generation_end scope=.. attempt=.. outcome=.. reason=.. nudged=..
 *   runnerCalls=.. finalChars=.. overLimit=.. elapsedMs=..
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import {
  MemoryBackend,
  PERSONA,
  SCENE_BODY,
  SCENE_FILE,
  capturingLogger,
  checkpointOf,
  endFields,
  endLines,
  fsFixture,
  l3For,
  recordingSink,
  scriptedRunner,
  silent,
  storageFixture,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

const CTX = { l3TaskId: "L3-task-7" };
const noWrite = () => "nothing to change";
const writes = (text: string) => (p: { workspaceDir?: string }) => { writePersonaFs(p.workspaceDir!, text); return "ok"; };
const timeout = () => { throw Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" }); };

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

async function oneEndLine(run: () => Promise<unknown>, lines: Array<{ msg: string }>) {
  await run().catch(() => undefined);
  const ends = endLines(lines as never);
  expect(ends).toHaveLength(1);
  expect(ends[0]).not.toContain(PERSONA.slice(2, 20));
  return endFields(ends[0]);
}

describe("End-of-generation log line", () => {
  it.each([
    ["A nudge then write", [noWrite, writes(`${PERSONA}\n\nNew.`)], { outcome: "generated", reason: "none", nudged: "true", runnerCalls: "2", overLimit: "false" }],
    ["B nudge, still no write", [noWrite, noWrite], { outcome: "failed", reason: "no-write", nudged: "true", runnerCalls: "2", finalChars: "none" }],
    ["C timeout on call 1", [timeout], { outcome: "failed", reason: "timeout", nudged: "false", runnerCalls: "1", finalChars: "none" }],
    ["E written over the limit", [writes(`# P\n\n${"a".repeat(8_995)}`)], { outcome: "generated", reason: "none", nudged: "false", runnerCalls: "1", finalChars: "9000", overLimit: "true" }],
  ])("%s: exactly one end line with the right fields", async (_name, script, want) => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const { runner } = scriptedRunner(script as never);
    const { logger, lines } = capturingLogger();

    const f = await oneEndLine(() => l3For({ dir, runner, logger }).call(undefined, CTX), lines);

    expect(f).toMatchObject({ scope: "global", attempt: "L3-task-7", ...want });
    expect(Number(f.elapsedMs)).toBeGreaterThanOrEqual(0);
  });

  it("early \"unchanged\" (no changed scene): outcome=skipped reason=none runnerCalls=0, not counted as a generation", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const cpm = new CheckpointManager(dir, silent);
    await cpm.markPersonaGenerated(0);
    await cpm.incrementMemoriesSincePersona(60);
    const { runner, calls } = scriptedRunner([noWrite]);
    const { logger, lines } = capturingLogger();
    const sink = recordingSink();

    const f = await oneEndLine(() => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX), lines);

    expect(calls).toHaveLength(0);
    expect(f).toMatchObject({ outcome: "skipped", reason: "none", runnerCalls: "0", nudged: "false" });
    expect(sink.records.filter((r) => r.invokedRunner)).toHaveLength(0);
  });

  it("the final persona.md write throws: outcome=failed reason=write-error", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const fx = await storageFixture();
    let modelDone = false;
    const backend = new MemoryBackend();
    for (const [k, v] of fx.backend.objects) backend.objects.set(k, v);
    const put = backend.putObject.bind(backend);
    backend.putObject = async (key, content) => {
      if (modelDone && key === "persona.md") throw new Error("disk full");
      return put(key, content);
    };
    const storage = new StorageAdapter(backend);
    const { runner } = scriptedRunner([async () => { await storage.writeFile("persona.md", `${PERSONA}\n\nNew.`); modelDone = true; return "ok"; }]);
    const { logger, lines } = capturingLogger();

    const f = await oneEndLine(() => l3For({ dir: fx.dir, runner, logger, storage }).call(undefined, CTX), lines);

    expect(f).toMatchObject({ outcome: "failed", reason: "write-error", runnerCalls: "1" });
    expect((await checkpointOf(fx.dir, storage)).last_persona_time).toBe("");
  });

  it("the generator succeeds but the deadline passes at the checkpoint guard: outcome=failed reason=checkpoint-rejected", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
    // The checkpoint lock takes 10 s to acquire: past the 5 s generation deadline.
    const checkpointLock = {
      lockKey: "l3-guard",
      maxWaitMs: 60_000,
      lock: {
        acquireLock: async () => { vi.setSystemTime(new Date(Date.now() + 10_000)); return true; },
        releaseLock: async () => {},
        renewLock: async () => true,
      },
    };
    const { runner } = scriptedRunner([writes(`${PERSONA}\n\nNew.`)]);
    const { logger, lines } = capturingLogger();

    const f = await oneEndLine(() => l3For({ dir, runner, logger, checkpointLock, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined, CTX), lines);

    expect(f).toMatchObject({ outcome: "failed", reason: "checkpoint-rejected", runnerCalls: "1" });
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});

describe("/health persona block", () => {
  function health(): { persona?: Record<string, number> } {
    const gw = Object.create(TdaiGateway.prototype) as InstanceType<typeof TdaiGateway>;
    Object.assign(gw, {
      startTime: Date.now(),
      core: { getVectorStore: () => ({ isFtsAvailable: () => true }), getEmbeddingService: () => ({}) },
      timerScanner: null,
      pipelineWorker: null,
      stateBackend: null,
    });
    let body = "";
    const res = { writeHead: vi.fn(), end: (s: string) => { body = s; } } as unknown as http.ServerResponse;
    (gw as unknown as { handleHealth(r: http.ServerResponse): void }).handleHealth(res);
    return JSON.parse(body) as { persona?: Record<string, number> };
  }

  it("a nudged generation through the default sink moves /health persona counters by +1 each", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const before = health().persona;
    expect(before).toBeDefined();
    const dir = await fsFixture();

    await l3For({ dir, runner: scriptedRunner([noWrite, writes(`${PERSONA}\n\nNew.`)]).runner }).call(undefined);

    const after = health().persona;
    expect(after!.personaGenerations - before!.personaGenerations).toBe(1);
    expect(after!.personaWriteNudges - before!.personaWriteNudges).toBe(1);
    expect(after!.personaWriteNudgeSuccesses - before!.personaWriteNudgeSuccesses).toBe(1);
    expect(after!.personaOverLimit - before!.personaOverLimit).toBe(0);
  });
});

describe("Case J control: a failed generation loses no scene", () => {
  it("J generation 1 fails, generation 2 succeeds and is given every scene generation 1 was given", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    fs.writeFileSync(path.join(dir, "scene_blocks", "travel.md"), SCENE_BODY.replace("coding", "travel").replace("Coding", "Travel"), "utf-8");
    await writeSceneIndex(dir, [
      { filename: SCENE_FILE, summary: "Coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" },
      { filename: "travel.md", summary: "Travel scene", heat: 1, created: "2026-11-01T09:30:00.000Z", updated: "2026-11-01T11:00:00.000Z" },
    ]);
    const first = scriptedRunner([() => { throw Object.assign(new Error("upstream 503"), { statusCode: 503 }); }]);
    await expect(l3For({ dir, runner: first.runner }).call(undefined)).rejects.toThrow("Persona generation failed");
    const second = scriptedRunner([writes(`${PERSONA}\n\nNew.`)]);
    await l3For({ dir, runner: second.runner }).call(undefined);

    for (const name of [SCENE_FILE, "travel.md"]) {
      expect(first.calls[0].prompt).toContain(name);
      expect(second.calls[0].prompt).toContain(name);
    }
    expect((await checkpointOf(dir)).memories_since_last_persona).toBe(0);
  });
});
