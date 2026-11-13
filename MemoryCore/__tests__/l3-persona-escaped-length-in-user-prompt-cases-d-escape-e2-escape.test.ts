/**
 * L3 persona: one measure of the existing body's length everywhere (cases
 * D-escape, E2-escape). The measure is escapeXmlTags(stripSceneNavigation(x)
 * .trim()).length, the same object the post-write length check uses. Fake
 * LLM, temp dirs, no network.
 *
 * Raw body 5_986 chars + "</system>" = 5_995 raw; escapeXmlTags turns the
 * 9-char tag into a 15-char one, so the measured length is 6_001 > 6_000.
 *
 *   D-escape   compress mode is on, every length number the user prompt prints
 *              says 6001 (never 5995), and the embedded markdown block keeps
 *              the raw body; chat and code prompt modes.
 *   D-escape   buildPersonaPrompt refuses an existingPersona without
 *              existingPersonaChars.
 *   E2-escape  the model keeps that body: the length the end line reports
 *              (finalChars, over the limit) is the number the prompt printed.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPersonaPrompt } from "../src/core/prompts/persona-generation.js";
import { escapeXmlTags } from "../src/utils/sanitize.js";
import {
  PERSONA,
  capturingLogger,
  endFields,
  endLines,
  fsFixture,
  l3For,
  scriptedRunner,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

/** A persona body of exactly n characters, no digits in it. */
function body(n: number): string {
  const head = "# Persona\n\n";
  return head + "a".repeat(n - head.length);
}

const RAW = body(5_986) + "</system>";

afterEach(() => { vi.unstubAllEnvs(); });

/** Every "<n> 字符" length the user prompt prints about the existing body. */
function existingLengthNumbers(user: string): string[] {
  return [...user.matchAll(/完整内容（(\d+) 字符）|正文长度为 (\d+) 字符/g)].map((m) => m[1] ?? m[2]);
}

describe("Case D-escape the user prompt prints the escaped length", () => {
  it("the fixture: raw 5995 < 6000, escaped 6001 > 6000", () => {
    expect(RAW.length).toBe(5_995);
    expect(escapeXmlTags(RAW).length).toBe(6_001);
  });

  it.each([["chat"], ["code"]])("D-escape promptMode=%s: compress on, both length lines say 6001, none says 5995, the markdown block keeps the raw body", async (promptMode) => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture(RAW);
    const { runner, calls } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, PERSONA); return "ok"; }]);

    await l3For({ dir, runner, cfg: { persona: { promptMode } } }).call(undefined);

    const user = calls[0].prompt;
    expect(user).toContain("整体重写");
    expect(existingLengthNumbers(user)).toEqual(["6001", "6001"]);
    expect(user).not.toContain("5995");
    expect(user).toContain("```markdown\n" + RAW + "\n```");
  });

  it("D-escape buildPersonaPrompt throws when existingPersona is given without existingPersonaChars", () => {
    expect(() => buildPersonaPrompt({
      mode: "incremental", promptMode: "chat", currentTime: "t", totalProcessed: 1, sceneCount: 1,
      changedSceneCount: 1, changedScenesContent: "", existingPersona: RAW,
      personaFilePath: "", checkpointPath: "",
    } as unknown as Parameters<typeof buildPersonaPrompt>[0])).toThrow(/existingPersonaChars/);
  });

  it.each([["chat"], ["code"]])("D-escape buildPersonaPrompt promptMode=%s with existingPersonaChars=6001 prints 6001 in every length line", (promptMode) => {
    const { userPrompt } = buildPersonaPrompt({
      mode: "incremental", promptMode: promptMode as "chat" | "code", currentTime: "t", totalProcessed: 1, sceneCount: 1,
      changedSceneCount: 1, changedScenesContent: "", existingPersona: RAW, existingPersonaChars: 6_001,
      personaFilePath: "", checkpointPath: "",
    });
    expect(existingLengthNumbers(userPrompt)).toEqual(["6001", "6001"]);
    expect(userPrompt).not.toContain("5995");
  });
});

describe("Case E2-escape the written body and the prompt use the same measure", () => {
  it.each([["chat"], ["code"]])("E2-escape promptMode=%s: existing raw 5995 + </system>, the model keeps it: end line finalChars=6001 overLimit=true, and every prompt number is that same 6001", async (promptMode) => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture(RAW);
    const { runner, calls } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, RAW); return "ok"; }]);
    const { logger, lines } = capturingLogger();

    await l3For({ dir, runner, logger, cfg: { persona: { promptMode } } }).call(undefined);

    const end = endFields(endLines(lines)[0]);
    expect(end).toMatchObject({ finalChars: "6001", overLimit: "true" });
    expect(existingLengthNumbers(calls[0].prompt)).toEqual([end.finalChars, end.finalChars]);
  });
});
