/**
 * L3 persona: the deadline is anchored at the start of the scope, and two
 * floors cut a call that could not finish anyway (cases F1b, F1-float).
 * Fake LLM, fake Date (vi.useFakeTimers toFake Date only), no network.
 *
 *   F1b-a  the scope's first steps (scene index read) spend 160 s of the
 *          180 s budget: 20 s < minRunnerMs (30 s), so runner.run is never
 *          called -> failed reason=deadline, scope end counted, not a
 *          generation.
 *   F1b-b  call 1 spends 165 s without writing: 15 s < minNudgeMs (20 s), so
 *          no nudge -> failed reason=deadline, nudgeSkipped counted.
 *   F1b-c  parseConfig validates TDAI_PERSONA_MIN_RUNNER_MS /
 *          TDAI_PERSONA_MIN_NUDGE_MS (integers 0..600000, <= timeoutMs) and
 *          derives their defaults from timeoutMs.
 *   F1-float control: a fractional timeoutMs (YAML or env) throws.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { PersonaGenerationSink } from "../src/core/report/persona-generation-telemetry.js";
import {
  capturingLogger,
  checkpointOf,
  endFields,
  endLines,
  fsFixture,
  l3For,
  scriptedRunner,
  slowSceneIndexStorage,
  storageFixture,
} from "./helpers/l3-persona-harness.js";

const CTX = { l3TaskId: "L3-f1b" };

function freezeClock(): void {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
}

function advance(ms: number): void {
  vi.setSystemTime(new Date(Date.now() + ms));
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

describe("Case F1b-a deadline anchored at the start of the scope", () => {
  it("F1b-a 160 s spent before the generator (20 s < minRunnerMs 30 s): runner never called, failed reason=deadline, scope end +1, generations 0, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const fx = await storageFixture();
    freezeClock();
    const storage = slowSceneIndexStorage(fx.backend, () => advance(160_000));
    const sink = new PersonaGenerationSink();
    const { runner, calls } = scriptedRunner([() => "unused"]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir: fx.dir, runner, storage, logger, personaTelemetry: sink }).call(undefined, CTX))
      .rejects.toThrow();

    expect(calls).toHaveLength(0);
    const ends = endLines(lines);
    expect(ends).toHaveLength(1);
    expect(endFields(ends[0])).toMatchObject({
      outcome: "failed", reason: "deadline", invokedRunner: "false", runnerCalls: "0", nudged: "false", nudgeSkipped: "false",
    });
    expect(sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 1 });
    const cp = await checkpointOf(fx.dir, storage);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});

describe("Case F1b-b no nudge without the time to finish it", () => {
  it("F1b-b call 1 spends 165 s of 180 s without writing (15 s < minNudgeMs 20 s): no nudge, failed reason=deadline, nudgeSkipped counted", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const dir = await fsFixture();
    freezeClock();
    const sink = new PersonaGenerationSink();
    const { runner, calls } = scriptedRunner([() => { advance(165_000); return "no write"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX)).rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(1);
    expect(endFields(endLines(lines)[0])).toMatchObject({
      outcome: "failed", reason: "deadline", invokedRunner: "true", runnerCalls: "1", nudged: "false", nudgeSkipped: "true",
    });
    expect(sink.snapshot()).toMatchObject({
      personaGenerations: 1, personaScopeEnds: 1, personaWriteNudges: 0, personaWriteNudgesSkipped: 1,
    });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

describe("Case F1b-c minRunnerMs / minNudgeMs startup validation", () => {
  it.each([
    ["TDAI_PERSONA_MIN_RUNNER_MS", "-1"],
    ["TDAI_PERSONA_MIN_RUNNER_MS", "1000.5"],
    ["TDAI_PERSONA_MIN_RUNNER_MS", "abc"],
    ["TDAI_PERSONA_MIN_RUNNER_MS", "600001"],
    ["TDAI_PERSONA_MIN_RUNNER_MS", "180001"],
    ["TDAI_PERSONA_MIN_NUDGE_MS", "-1"],
    ["TDAI_PERSONA_MIN_NUDGE_MS", "1000.5"],
    ["TDAI_PERSONA_MIN_NUDGE_MS", "20s"],
    ["TDAI_PERSONA_MIN_NUDGE_MS", "600001"],
    ["TDAI_PERSONA_MIN_NUDGE_MS", "180001"],
  ])("F1b-c parseConfig throws at startup on %s=%s", (name, value) => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    vi.stubEnv(name, value);
    expect(() => parseConfig({})).toThrow(name);
  });

  it.each([
    [{ minRunnerMs: -1 }],
    [{ minRunnerMs: 1000.5 }],
    [{ minRunnerMs: "30000" }],
    [{ minNudgeMs: 600_001 }],
    [{ minNudgeMs: Number.NaN }],
    [{ timeoutMs: 5_000, minRunnerMs: 5_001 }],
    [{ timeoutMs: 5_000, minNudgeMs: 6_000 }],
  ])("F1b-c parseConfig throws at startup on YAML persona %j", (persona) => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    vi.stubEnv("TDAI_PERSONA_MIN_RUNNER_MS", undefined);
    vi.stubEnv("TDAI_PERSONA_MIN_NUDGE_MS", undefined);
    expect(() => parseConfig({ persona })).toThrow(/memory\.persona\.(minRunnerMs|minNudgeMs)/);
  });

  it("F1b-c defaults derive from timeoutMs (30000 / 20000 at 180000, 833 / 555 at 5000); valid values are kept, env wins over YAML", () => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    vi.stubEnv("TDAI_PERSONA_MIN_RUNNER_MS", undefined);
    vi.stubEnv("TDAI_PERSONA_MIN_NUDGE_MS", undefined);
    expect(parseConfig({}).persona).toMatchObject({ timeoutMs: 180_000, minRunnerMs: 30_000, minNudgeMs: 20_000 });
    expect(parseConfig({ persona: { timeoutMs: 5_000 } }).persona).toMatchObject({ minRunnerMs: 833, minNudgeMs: 555 });
    expect(parseConfig({ persona: { minRunnerMs: 0, minNudgeMs: 180_000 } }).persona).toMatchObject({ minRunnerMs: 0, minNudgeMs: 180_000 });
    vi.stubEnv("TDAI_PERSONA_MIN_RUNNER_MS", "40000");
    vi.stubEnv("TDAI_PERSONA_MIN_NUDGE_MS", "0");
    expect(parseConfig({ persona: { minRunnerMs: 1, minNudgeMs: 1 } }).persona).toMatchObject({ minRunnerMs: 40_000, minNudgeMs: 0 });
  });
});

describe("Case F1-float control: a fractional timeoutMs is refused", () => {
  it("F1-float YAML timeoutMs 5000.5 makes parseConfig throw", () => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    expect(() => parseConfig({ persona: { timeoutMs: 5000.5 } })).toThrow("memory.persona.timeoutMs");
  });

  it("F1-float TDAI_PERSONA_TIMEOUT_MS=\"5000.5\" makes parseConfig throw", () => {
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", "5000.5");
    expect(() => parseConfig({})).toThrow("TDAI_PERSONA_TIMEOUT_MS");
  });
});
