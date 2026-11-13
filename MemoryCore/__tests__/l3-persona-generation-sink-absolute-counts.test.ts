/**
 * The persona generation sink injected into createL3Runner: absolute counts.
 * A generation is one that called the LLM runner at least once, so the early
 * "no changed scene" skip is not counted. Fake LLM, temp dirs, no network.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { PersonaGenerationSink } from "../src/core/report/persona-generation-telemetry.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { PERSONA, fsFixture, l3For, scriptedRunner, silent, writePersonaFs } from "./helpers/l3-persona-harness.js";

const noWrite = () => "nothing to change";
const writes = (text: string) => (p: { workspaceDir?: string }) => { writePersonaFs(p.workspaceDir!, text); return "ok"; };

afterEach(() => { vi.unstubAllEnvs(); });

describe("PersonaGenerationSink injected into createL3Runner", () => {
  it("absolute counts after two generations (two PersonaGenerator instances) and one early skip", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const sink = new PersonaGenerationSink();

    // Generation 1: no write, nudge, write -> nudge success.
    const dirA = await fsFixture();
    await l3For({ dir: dirA, runner: scriptedRunner([noWrite, writes(`${PERSONA}\n\nNew.`)]).runner, personaTelemetry: sink }).call(undefined);
    // Generation 2: one call writes a 9000-char body -> over the 6000 limit.
    const dirE = await fsFixture();
    await l3For({ dir: dirE, runner: scriptedRunner([writes(`# P\n\n${"a".repeat(8_995)}`)]).runner, personaTelemetry: sink }).call(undefined);
    // Early skip: persona exists, no scene changed since the last persona.
    const dirSkip = await fsFixture();
    const cpm = new CheckpointManager(dirSkip, silent);
    await cpm.markPersonaGenerated(0);
    await cpm.incrementMemoriesSincePersona(60);
    const skip = scriptedRunner([noWrite]);
    await l3For({ dir: dirSkip, runner: skip.runner, personaTelemetry: sink }).call(undefined);

    expect(skip.calls).toHaveLength(0);
    expect(sink.snapshot()).toEqual({
      personaGenerations: 2,
      personaWriteNudges: 1,
      personaWriteNudgeSuccesses: 1,
      personaOverLimit: 1,
      personaScopeEnds: 2,
      personaWriteNudgesSkipped: 0,
    });
  });

  it("a nudge whose second call still writes nothing is a nudge, not a success", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const sink = new PersonaGenerationSink();
    const dir = await fsFixture();

    await l3For({ dir, runner: scriptedRunner([noWrite, noWrite]).runner, personaTelemetry: sink }).call(undefined).catch(() => undefined);

    expect(sink.snapshot()).toEqual({
      personaGenerations: 1,
      personaWriteNudges: 1,
      personaWriteNudgeSuccesses: 0,
      personaOverLimit: 0,
      personaScopeEnds: 1,
      personaWriteNudgesSkipped: 0,
    });
  });

  it("scope ends count a failure that never reached the runner; a skipped nudge is counted", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const sink = new PersonaGenerationSink();

    // Scope end 1: aborted before the generator -> failed, invokedRunner=false.
    const dirAbort = await fsFixture();
    const controller = new AbortController();
    controller.abort();
    await l3For({ dir: dirAbort, runner: scriptedRunner([noWrite]).runner, personaTelemetry: sink })
      .call(undefined, { signal: controller.signal }).catch(() => undefined);
    // Scope end 2: call 1 spends 165 s of 180 s without writing -> nudge skipped.
    const dirSkip = await fsFixture();
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
    const slow = () => { vi.setSystemTime(new Date(Date.now() + 165_000)); return "no write"; };
    await l3For({ dir: dirSkip, runner: scriptedRunner([slow]).runner, personaTelemetry: sink }).call(undefined).catch(() => undefined);
    vi.useRealTimers();

    expect(sink.snapshot()).toEqual({
      personaGenerations: 1,
      personaWriteNudges: 0,
      personaWriteNudgeSuccesses: 0,
      personaOverLimit: 0,
      personaScopeEnds: 2,
      personaWriteNudgesSkipped: 1,
    });
  });
});
