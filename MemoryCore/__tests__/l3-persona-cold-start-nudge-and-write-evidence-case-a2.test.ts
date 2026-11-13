/**
 * L3 persona, case A2: the write nudge at a real cold start (no persona.md),
 * on a backend whose `stat` fails, and the counters checked after the scope
 * ended. createL3Runner + PersonaGenerator run for real in a temp dir or an
 * in-memory StorageAdapter; only the LLM is a fake. No network, no model.
 *
 *   A2a cold start: call 1 answers in text without writing, call 2 (nudged)
 *       writes persona.md through the `write` tool -> generated after 2
 *       calls; same when `stat` throws so no write marker is ever readable
 *       (the tool's own success count is the evidence).
 *   A2b the nudged call writes only whitespace / only a navigation block:
 *       failed (write-error), the nudge is counted, its success is not.
 *   A2c a written body over the limit whose checkpoint advance the guard
 *       refuses: the scope failed, so personaOverLimit stays 0.
 *   A2d persona exists, neither call writes, and `stat` fails after call 2
 *       (marker undefined) while the old body is still readable: no-write,
 *       checkpoint unchanged.
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PERSONA_WRITE_NUDGE } from "../src/core/persona/persona-generator.js";
import { PersonaGenerationSink } from "../src/core/report/persona-generation-telemetry.js";
import { generateSceneNavigation } from "../src/core/scene/scene-navigation.js";
import type { StorageAdapter } from "../src/core/storage/adapter.js";
import {
  PERSONA,
  SCENE_FILE,
  capturingLogger,
  checkpointOf,
  endFields,
  endLines,
  fsFixture,
  l3For,
  scriptedRunner,
  storageFixture,
  writePersonaFs,
  writeViaTool,
} from "./helpers/l3-persona-harness.js";

const CTX = { l3TaskId: "L3-a2" };
const COLD_BODY = "# Persona\n\nCold start body written by the nudged call.";
const textOnly = () => "Here is the persona I would write; I did not call any tool.";

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

/** Make storage.stat("persona.md") throw (all of it, or only once `broken()` is true). */
function breakPersonaStat(storage: StorageAdapter, broken: () => boolean = () => true): void {
  const orig = storage.stat.bind(storage);
  storage.stat = async (key: string) => {
    if (key === "persona.md" && broken()) throw new Error("stat is not supported by this backend");
    return orig(key);
  };
}

describe("Case A2a cold start nudge", () => {
  it("A2a local FS, no persona.md: call 1 text only, call 2 writes via the write tool -> generated after exactly 2 calls, checkpoint advanced, nudge counters +1", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture({ persona: null });
    expect(fs.existsSync(path.join(dir, "persona.md"))).toBe(false);
    const sink = new PersonaGenerationSink();
    const { runner, calls } = scriptedRunner([textOnly, async (p) => writeViaTool(p, COLD_BODY)]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX)).resolves.toBeUndefined();

    expect(calls).toHaveLength(2);
    expect(calls[1].prompt).toBe(`${calls[0].prompt}${PERSONA_WRITE_NUDGE}`);
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8").startsWith(`${COLD_BODY}\n\n`)).toBe(true);
    const cp = await checkpointOf(dir);
    expect(cp.memories_since_last_persona).toBe(0);
    expect(cp.last_persona_time).not.toBe("");
    expect(sink.snapshot()).toMatchObject({ personaGenerations: 1, personaWriteNudges: 1, personaWriteNudgeSuccesses: 1 });
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "generated", reason: "none", nudged: "true", runnerCalls: "2" });
  });

  it("A2a storage whose stat throws (no marker before or after), no persona.md: the nudged write tool call is the evidence -> generated after 2 calls", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const fx = await storageFixture({ persona: null });
    expect(fx.backend.text("persona.md")).toBeUndefined();
    breakPersonaStat(fx.storage);
    const sink = new PersonaGenerationSink();
    const { runner, calls } = scriptedRunner([textOnly, async (p) => writeViaTool(p, COLD_BODY)]);

    await expect(l3For({ dir: fx.dir, runner, storage: fx.storage, personaTelemetry: sink }).call(undefined, CTX)).resolves.toBeUndefined();

    expect(calls).toHaveLength(2);
    expect(fx.backend.text("persona.md")!.startsWith(`${COLD_BODY}\n\n`)).toBe(true);
    expect((await checkpointOf(fx.dir, fx.storage)).memories_since_last_persona).toBe(0);
    expect(sink.snapshot()).toMatchObject({ personaGenerations: 1, personaWriteNudges: 1, personaWriteNudgeSuccesses: 1 });
  });
});

describe("Case A2b the nudged call writes an empty body", () => {
  const nav = generateSceneNavigation(
    [{ filename: SCENE_FILE, summary: "Coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" }],
    undefined,
    false,
  );

  it.each([
    ["only whitespace", "   \n\t \n"],
    ["only a navigation block", `\n\n${nav}\n`],
  ])("A2b cold start, call 2 writes %s: failed reason=write-error, nudge counted, success not", async (_name, content) => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    expect(nav.length).toBeGreaterThan(0);
    const dir = await fsFixture({ persona: null });
    const sink = new PersonaGenerationSink();
    const { runner, calls } = scriptedRunner([textOnly, async (p) => writeViaTool(p, content)]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX)).rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(2);
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "write-error", nudged: "true", runnerCalls: "2" });
    expect(sink.snapshot()).toMatchObject({ personaGenerations: 1, personaWriteNudges: 1, personaWriteNudgeSuccesses: 0 });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

describe("Case A2c over the limit but the checkpoint advance is refused", () => {
  it("A2c a 9000-char body written, markPersonaGenerated refused by the guard (checkpoint-rejected): personaOverLimit stays 0", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
    // The checkpoint lock takes 10 s: past the 5 s deadline of the scope.
    const checkpointLock = {
      lockKey: "a2c",
      maxWaitMs: 60_000,
      lock: {
        acquireLock: async () => { vi.setSystemTime(new Date(Date.now() + 10_000)); return true; },
        releaseLock: async () => {},
        renewLock: async () => true,
      },
    };
    const sink = new PersonaGenerationSink();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, `# P\n\n${"a".repeat(8_995)}`); return "ok"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, checkpointLock, personaTelemetry: sink, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined, CTX))
      .rejects.toThrow();

    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "checkpoint-rejected", finalChars: "9000" });
    expect(sink.snapshot()).toMatchObject({ personaGenerations: 1, personaOverLimit: 0 });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

describe("Case A2d marker lost after the nudge, nothing written", () => {
  it("A2d persona exists, marker readable before, neither call writes, stat fails after call 2 while the old body is readable: failed reason=no-write, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const fx = await storageFixture();
    let statBroken = false;
    breakPersonaStat(fx.storage, () => statBroken);
    const { runner, calls } = scriptedRunner([textOnly, () => { statBroken = true; return "still nothing to change"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir: fx.dir, runner, logger, storage: fx.storage }).call(undefined, CTX)).rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(2);
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "no-write", nudged: "true", runnerCalls: "2" });
    expect(fx.backend.text("persona.md")).toBe(PERSONA);
    const cp = await checkpointOf(fx.dir, fx.storage);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});
