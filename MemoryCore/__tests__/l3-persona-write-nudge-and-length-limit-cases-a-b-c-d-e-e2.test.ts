/**
 * L3 persona: the write nudge and the length limit (cases A, B, C, D, E, E2).
 * The real PersonaGenerator / createL3Runner run in a temp dir; only the LLM
 * is a fake. No network, no model.
 *
 *   A  the first call ends without writing persona.md, the second (nudged)
 *      call writes it: "generated" after exactly 2 calls.
 *   B  neither call writes: the L3 runner throws, both checkpoint fields stay.
 *   C  control: a timeout on call 1 is "failed" after 1 call (no nudge).
 *   D  the length limit comes from config per prompt mode, in both prompts,
 *      and an existing body over it turns on the compress instruction.
 *   E  a written body over the limit is kept, counted and logged.
 *   E2 the measure is the body after strip + escape, not the file on disk.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPersonaPrompt } from "../src/core/prompts/persona-generation.js";
import { PersonaGenerator } from "../src/core/persona/persona-generator.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import { stripSceneNavigation } from "../src/core/scene/scene-navigation.js";
import { escapeXmlTags } from "../src/utils/sanitize.js";
import {
  PERSONA,
  SCENE_FILE,
  capturingLogger,
  checkpointOf,
  fsFixture,
  l3For,
  recordingSink,
  scriptedRunner,
  silent,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

const FIXTURES = fileURLToPath(new URL("./fixtures/", import.meta.url));
// A Windows checkout may turn the fixtures' LF into CRLF; the prompts are LF.
const readFixture = (name: string) => fs.readFileSync(path.join(FIXTURES, name), "utf-8").replace(/\r\n/g, "\n");
const ORIGINAL_CHAT = readFixture("persona-system-prompt-chat-881bd21c.txt");
const ORIGINAL_CODE = readFixture("persona-system-prompt-code-881bd21c.txt");

/** A persona body of exactly n characters, no digits in it. */
function body(n: number): string {
  const head = "# Persona\n\n";
  return head + "a".repeat(n - head.length);
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

describe("Case A write nudge", () => {
  it("A the model ends call 1 without writing, call 2 carries the nudge and writes: generated after exactly 2 calls", async () => {
    const dir = await fsFixture();
    const { runner, calls } = scriptedRunner([
      () => "I reviewed the scenes; nothing to change.",
      (p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nAlso likes tests.`); return "written"; },
    ]);
    const gen = new PersonaGenerator({ dataDir: dir, config: undefined, llmRunner: runner, logger: silent });

    await expect(gen.generateLocalPersona("test")).resolves.toBe("generated");

    expect(calls).toHaveLength(2);
    // The nudge is the original user prompt plus a demand to call a write tool.
    expect(calls[1].prompt.startsWith(calls[0].prompt)).toBe(true);
    const nudge = calls[1].prompt.slice(calls[0].prompt.length);
    expect(nudge).toContain("必须调用 **write**");
    expect(nudge).toContain("**edit**");
    expect(calls[1].systemPrompt).toBe(calls[0].systemPrompt);
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toContain("Also likes tests.");
  });
});

describe("Case B nudge without a write", () => {
  it("B through createL3Runner: neither call writes -> \"Persona generation failed\", 2 calls, both checkpoint fields unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const { runner, calls } = scriptedRunner([() => "nothing to change", () => "still nothing to change"]);
    const l3 = l3For({ dir, runner });

    await expect(l3()).rejects.toThrow("[L3] Persona generation failed (scope=global)");

    expect(calls).toHaveLength(2);
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toBe(PERSONA);
  });
});

describe("Case C control: timeout is not nudged", () => {
  it("C call 1 times out -> failed after exactly 1 call", async () => {
    const dir = await fsFixture();
    const { runner, calls } = scriptedRunner([() => {
      throw Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" });
    }]);
    const gen = new PersonaGenerator({ dataDir: dir, config: undefined, llmRunner: runner, logger: silent });

    await expect(gen.generateLocalPersona("test")).resolves.toBe("failed");
    expect(calls).toHaveLength(1);
  });
});

describe("Case D length limit from config, compress mode", () => {
  async function promptsFor(cfg: Record<string, unknown>, existing: string) {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
    const dir = await fsFixture(existing);
    const { runner, calls } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, PERSONA); return "ok"; }]);
    await l3For({ dir, runner, cfg }).call(undefined);
    expect(calls.length).toBeGreaterThanOrEqual(1);
    return { system: calls[0].systemPrompt!, user: calls[0].prompt };
  }

  it("D chat default: a 14000-char persona over the 6000 limit asks for a full rewrite compressed below 6000", async () => {
    const { system, user } = await promptsFor({}, body(14_000));

    expect(user).toContain("整体重写");
    expect(user).toContain("14000 字符");
    expect(user).toContain("压缩到 6000 字符以内");
    expect(user).toContain("请控制在6000 字符内");
    expect(system).toContain("不要超过 6000 字符");
    expect(system.replace("不要超过 6000 字符", "不要超过 2000 字符")).toBe(ORIGINAL_CHAT);
  });

  it("D chat maxCharsChat=7777: both prompts carry 7777 字符, no literal 2000, system prompt otherwise unchanged", async () => {
    const { system, user } = await promptsFor({ persona: { maxCharsChat: 7777 } }, body(14_000));

    expect(system).toContain("不要超过 7777 字符");
    expect(user).toContain("请控制在7777 字符内");
    expect(user).toContain("压缩到 7777 字符以内");
    expect(system).not.toContain("2000");
    expect(user).not.toContain("2000");
    expect(system.replace("不要超过 7777 字符", "不要超过 2000 字符")).toBe(ORIGINAL_CHAT);
  });

  it("D code maxCharsCode=4321: every limit phrase says 4321 字符, no literal 1200, system prompt otherwise unchanged", async () => {
    const { system, user } = await promptsFor({ persona: { promptMode: "code", maxCharsCode: 4321 } }, body(14_000));

    expect(system.split("4321 字符").length - 1).toBe(3);
    expect(user).toContain("更新后必须压缩在 4321 字符以内");
    expect(user).toContain("压缩到 4321 字符以内");
    expect(system).not.toContain("1200");
    expect(user).not.toContain("1200");
    const restored = system
      .replace("禁止超过 4321 字符", "禁止超过 1200 字")
      .replace("全文不超过 4321 字符", "全文不超过 1200 字")
      .replace("最终内容不超过 4321 字符", "最终内容不超过 1200 字");
    expect(restored).toBe(ORIGINAL_CODE);
  });

  it("D below the limit: no compress instruction (control for the trigger)", () => {
    const existing = body(5_000);
    const { userPrompt } = buildPersonaPrompt({
      mode: "incremental", promptMode: "chat", currentTime: "t", totalProcessed: 1, sceneCount: 1,
      changedSceneCount: 1, changedScenesContent: "", existingPersona: existing,
      // The canonical measure, as the generator computes it.
      existingPersonaChars: escapeXmlTags(stripSceneNavigation(existing).trim()).length,
      personaFilePath: "", checkpointPath: "", maxCharsChat: 6_000,
    });
    expect(userPrompt).not.toContain("整体重写");
  });
});

describe("Case E written body over the limit", () => {
  it("E a 9000-char body is kept: generated, checkpoint advanced, personaOverLimit +1, one warning with the char count", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const written = body(9_000);
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, written); return "ok"; }]);
    const { logger, lines } = capturingLogger();
    const sink = recordingSink();

    await expect(l3For({ dir, runner, logger, personaTelemetry: sink }).call(undefined)).resolves.toBeUndefined();

    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8").startsWith(`${written}\n\n`)).toBe(true);
    const cp = await checkpointOf(dir);
    expect(cp.memories_since_last_persona).toBe(0);
    expect(cp.last_persona_time).not.toBe("");
    expect(sink.records.filter((r) => r.overLimit)).toHaveLength(1);
    const warnings = lines.filter((l) => l.level === "warn" && l.msg.includes("9000 chars") && l.msg.includes("6000"));
    expect(warnings).toHaveLength(1);
    expect(warnings[0].msg).not.toContain("aaaa");
  });
});

describe("Case E2 the measure is the escaped body, not the file", () => {
  it("E2 a body 1 char over the limit after escape (raw 5995 + </system>) counts once", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    // "</system>" (9 chars) escapes to "&lt;/system&gt;" (15 chars): 5995 raw -> 6001 measured.
    const written = body(5_986) + "</system>";
    expect(written.length).toBe(5_995);
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, written); return "ok"; }]);
    const sink = recordingSink();

    await l3For({ dir, runner, personaTelemetry: sink }).call(undefined);

    expect(sink.records).toHaveLength(1);
    expect(sink.records[0].overLimit).toBe(true);
  });

  it("E2 a body under the limit whose file (with navigation) is over it: not counted, next generation not in compress mode", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const written = body(5_990);
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, written); return "ok"; }]);
    const sink = recordingSink();

    await l3For({ dir, runner, personaTelemetry: sink }).call(undefined);

    const onDisk = fs.readFileSync(path.join(dir, "persona.md"), "utf-8");
    expect(onDisk.length).toBeGreaterThan(6_000);
    expect(sink.records).toHaveLength(1);
    expect(sink.records[0].overLimit).toBe(false);

    // Next generation: the scene changed again; the existing body is 5990 < 6000.
    await writeSceneIndex(dir, [{ filename: SCENE_FILE, summary: "Coding scene", heat: 2, created: "2026-11-01T09:00:00.000Z", updated: "2099-01-01T00:00:00.000Z" }]);
    const next = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, written); return "ok"; }]);
    const gen = new PersonaGenerator({ dataDir: dir, config: undefined, llmRunner: next.runner, logger: silent });
    await gen.generateLocalPersona("next");
    expect(next.calls).toHaveLength(1);
    expect(next.calls[0].prompt).not.toContain("整体重写");
  });
});
