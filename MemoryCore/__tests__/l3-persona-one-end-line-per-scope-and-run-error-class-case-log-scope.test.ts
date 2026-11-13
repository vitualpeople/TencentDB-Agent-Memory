/**
 * L3 persona: one end line per scope from a single try/catch/finally around
 * the whole scope body (case Log-Scope). createL3Runner runs for real; only
 * the LLM is a fake. No network.
 *
 * Pinned format (one line):
 *   [memory-tdai] [pipeline-factory] [L3] persona_generation_end scope=<scope>
 *   attempt=<l3TaskId|none> outcome=<generated|failed|skipped>
 *   rawOutcome=<generated|unchanged|failed|skipped>
 *   reason=<no-write|timeout|provider-error|aborted|deadline|checkpoint-rejected|write-error|none>
 *   invokedRunner=<bool> runnerCalls=<0..2> nudged=<bool> nudgeSkipped=<bool>
 *   finalChars=<n|none> overLimit=<bool> elapsedMs=<n>
 *
 *   a  the four "nothing to generate" exits: skipped, not counted.
 *   b  abort / deadline before the generator: failed, invokedRunner=false,
 *      counted as a scope end (never skipped).
 *   c  the model rewrites the same body after the runner was called:
 *      outcome=generated rawOutcome=unchanged.
 *   d  classifyRunError: aborted, deadline, provider-error (HTTP status),
 *      timeout, in that order.
 *   e  a normal generation: exactly one line with every pinned field.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { PersonaGenerator } from "../src/core/persona/persona-generator.js";
import { PersonaGenerationSink } from "../src/core/report/persona-generation-telemetry.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL3Runner } from "../src/utils/pipeline-factory.js";
import {
  PERSONA,
  capturingLogger,
  checkpointOf,
  endFields,
  endLines,
  fsFixture,
  l3For,
  scriptedRunner,
  silent,
  slowSceneIndexStorage,
  storageFixture,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

const CTX = { l3TaskId: "L3-log" };
const END_LINE = new RegExp(
  "^\\[memory-tdai\\] \\[pipeline-factory\\] \\[L3\\] persona_generation_end " +
  "scope=\\S+ attempt=\\S+ outcome=(generated|failed|skipped) rawOutcome=(generated|unchanged|failed|skipped) " +
  "reason=(no-write|timeout|provider-error|aborted|deadline|checkpoint-rejected|write-error|none) " +
  "invokedRunner=(true|false) runnerCalls=[0-2] nudged=(true|false) nudgeSkipped=(true|false) " +
  "finalChars=(\\d+|none) overLimit=(true|false) elapsedMs=\\d+$",
);
const SKIPPED = { outcome: "skipped", rawOutcome: "skipped", reason: "none", invokedRunner: "false", runnerCalls: "0" };

function freezeClock(): void {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
}

function advance(ms: number): void {
  vi.setSystemTime(new Date(Date.now() + ms));
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

/** Run, then return the single end line (format checked) and the sink. */
async function runOne(run: (sink: PersonaGenerationSink, logger: typeof silent) => Promise<unknown>) {
  const sink = new PersonaGenerationSink();
  const { logger, lines } = capturingLogger();
  const outcome = await run(sink, logger).then(() => "resolved", (e: Error) => e.message);
  const ends = endLines(lines);
  expect(ends).toHaveLength(1);
  expect(ends[0]).toMatch(END_LINE);
  return { fields: endFields(ends[0]), sink, outcome };
}

describe("Case Log-Scope a: the four exits with nothing to generate", () => {
  it("a1 the trigger says no (!shouldGenerate): one skipped line, nothing counted", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    await new CheckpointManager(dir, silent).markPersonaGenerated(0);
    const { runner, calls } = scriptedRunner([() => "unused"]);

    const r = await runOne((sink, logger) => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX));

    expect(calls).toHaveLength(0);
    expect(r.fields).toMatchObject(SKIPPED);
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 0 });
  });

  it("a2 no LLM runner and no OpenClaw config: one skipped line, nothing counted", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();

    const r = await runOne((sink, logger) => createL3Runner({
      pluginDataDir: dir, cfg: parseConfig({}), openclawConfig: undefined, logger, personaTelemetry: sink,
    }).call(undefined, CTX));

    expect(r.fields).toMatchObject(SKIPPED);
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 0 });
  });

  it("a3 an empty scene index: one skipped line, nothing counted, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    await writeSceneIndex(dir, []);
    const { runner, calls } = scriptedRunner([() => "unused"]);

    const r = await runOne((sink, logger) => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX));

    expect(calls).toHaveLength(0);
    expect(r.fields).toMatchObject(SKIPPED);
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 0 });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });

  it("a4 early unchanged (no changed scene) and the checkpoint advanced: one skipped line, nothing counted", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const cpm = new CheckpointManager(dir, silent);
    await cpm.markPersonaGenerated(0);
    await cpm.incrementMemoriesSincePersona(60);
    const { runner, calls } = scriptedRunner([() => "unused"]);

    const r = await runOne((sink, logger) => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX));

    expect(calls).toHaveLength(0);
    expect(r.outcome).toBe("resolved");
    expect(r.fields).toMatchObject(SKIPPED);
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 0 });
    expect((await checkpointOf(dir)).memories_since_last_persona).toBe(0);
  });
});

describe("Case Log-Scope b: abort / deadline before the runner is failed, never skipped", () => {
  it("b aborted before pullProfilesToLocal / the generator: failed rawOutcome=failed reason=aborted invokedRunner=false, scope end +1, then throws", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const controller = new AbortController();
    controller.abort();

    const r = await runOne((sink, logger) => l3For({ dir, runner: scriptedRunner([() => "unused"]).runner, logger, personaTelemetry: sink })
      .call(undefined, { ...CTX, signal: controller.signal }));

    expect(r.outcome).not.toBe("resolved");
    expect(r.fields).toMatchObject({ outcome: "failed", rawOutcome: "failed", reason: "aborted", invokedRunner: "false", runnerCalls: "0" });
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 1 });
  });

  it("b the deadline passed in the scope's first steps: failed rawOutcome=failed reason=deadline invokedRunner=false, scope end +1, then throws", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const fx = await storageFixture();
    freezeClock();
    const storage = slowSceneIndexStorage(fx.backend, () => advance(181_000));

    const r = await runOne((sink, logger) => l3For({ dir: fx.dir, runner: scriptedRunner([() => "unused"]).runner, storage, logger, personaTelemetry: sink })
      .call(undefined, CTX));

    expect(r.outcome).not.toBe("resolved");
    expect(r.fields).toMatchObject({ outcome: "failed", rawOutcome: "failed", reason: "deadline", invokedRunner: "false", runnerCalls: "0" });
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 0, personaScopeEnds: 1 });
  });
});

describe("Case Log-Scope c: unchanged after the runner is a generation", () => {
  it("c the model rewrites the same body, checkpoint advanced: outcome=generated rawOutcome=unchanged invokedRunner=true, generations +1, scope ends +1", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, PERSONA); return "same"; }]);

    const r = await runOne((sink, logger) => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX));

    expect(r.outcome).toBe("resolved");
    expect(r.fields).toMatchObject({ outcome: "generated", rawOutcome: "unchanged", reason: "none", invokedRunner: "true", runnerCalls: "1" });
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 1, personaScopeEnds: 1 });
    expect((await checkpointOf(dir)).memories_since_last_persona).toBe(0);
  });
});

describe("Case Log-Scope d: classifyRunError is deterministic", () => {
  async function reasonOf(opts: { thrower: () => never; abort?: AbortController; deadlineMs?: number }) {
    const dir = await fsFixture();
    freezeClock();
    const deadlineAt = Date.now() + (opts.deadlineMs ?? 60_000);
    const { runner } = scriptedRunner([opts.thrower]);
    const gen = new PersonaGenerator({
      dataDir: dir, config: undefined, llmRunner: runner, logger: silent, deadlineAt,
      ...(opts.abort ? { abortSignal: opts.abort.signal } : {}),
    });
    return (await gen.generateLocalPersonaDetailed("test")).reason;
  }
  const timeoutError = () => Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" });

  it("d1 the caller aborted while the runner throws TimeoutError -> aborted", async () => {
    const abort = new AbortController();
    expect(await reasonOf({ abort, thrower: () => { abort.abort(); throw timeoutError(); } })).toBe("aborted");
  });

  it("d2 the deadline has passed when the runner throws (no caller abort) -> deadline", async () => {
    expect(await reasonOf({ deadlineMs: 60_000, thrower: () => { advance(60_000); throw new Error("socket hang up"); } })).toBe("deadline");
  });

  it.each([
    [{ statusCode: 503 }, "upstream request timed out"],
    [{ status: 429 }, "rate limited, retry after the timeout"],
  ])("d3 an HTTP error %j before the deadline -> provider-error even when its message says timeout", async (fields, message) => {
    expect(await reasonOf({ thrower: () => { throw Object.assign(new Error(message), fields); } })).toBe("provider-error");
  });

  it("d4 TimeoutError before the deadline -> timeout", async () => {
    expect(await reasonOf({ thrower: () => { throw timeoutError(); } })).toBe("timeout");
  });
});

describe("Case Log-Scope e: one line for a normal generation", () => {
  it("e generated with a changed persona.md: exactly one end line, every pinned field, generations +1, scope ends +1", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nNew.`); return "ok"; }]);

    const r = await runOne((sink, logger) => l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined, CTX));

    expect(r.outcome).toBe("resolved");
    expect(r.fields).toEqual({
      scope: "global",
      attempt: "L3-log",
      outcome: "generated",
      rawOutcome: "generated",
      reason: "none",
      invokedRunner: "true",
      runnerCalls: "1",
      nudged: "false",
      nudgeSkipped: "false",
      finalChars: String(`${PERSONA}\n\nNew.`.length),
      overLimit: "false",
      elapsedMs: r.fields.elapsedMs,
    });
    expect(r.sink.snapshot()).toMatchObject({ personaGenerations: 1, personaScopeEnds: 1 });
  });
});
