/**
 * L3 persona: only persona.md may be written (case I, I-S, I-F). Everything
 * runs through createL3Runner + PersonaGenerator + the real
 * StandaloneLLMRunner; the provider is a fake chat/completions endpoint
 * (global fetch stub) that emits tool calls, so the runner's own AI SDK tool
 * loop executes them. No network, no model.
 *
 *   I    local FS: a write/edit to the scene index, the checkpoint, a new
 *        file in dataDir, or a scene body is refused, the target keeps every
 *        byte, the generation fails and an error is logged.
 *   I-S  the same on an in-memory StorageAdapter (scopedStore).
 *   path "./persona.md" is allowed; "scene_blocks/../persona.md" and
 *        "scene_blocks/persona.md" are refused, on FS and on storage.
 *   a forbidden call followed by a valid persona.md write still fails.
 *   I-F  the tools createL3Runner builds reach runner.run.
 *   controls: a valid write and a valid edit of persona.md (FS and storage).
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LLMRunParams, LLMRunner } from "../src/core/types.js";
import {
  PERSONA,
  SCENE_BODY,
  capturingLogger,
  checkpointOf,
  fsFixture,
  l3For,
  realRunner,
  storageFixture,
  stubChatProvider,
  type ToolCallSpec,
} from "./helpers/l3-persona-harness.js";

const NEW_PERSONA = `${PERSONA}\n\nAlso writes tests first.`;

afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

function refusals(lines: Array<{ level: string; msg: string }>): string[] {
  return lines.filter((l) => l.level === "error" && /refused (write|edit)/.test(l.msg)).map((l) => l.msg);
}

async function runFs(turns: Array<ToolCallSpec[] | null>) {
  vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
  const dir = await fsFixture();
  const { logger, lines } = capturingLogger();
  const provider = stubChatProvider(turns);
  const outcome = await l3For({ dir, runner: realRunner(logger), logger }).call(undefined)
    .then(() => "resolved", (err: Error) => err.message);
  return { dir, lines, outcome, provider };
}

async function runStorage(turns: Array<ToolCallSpec[] | null>) {
  vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
  const fx = await storageFixture();
  const { logger, lines } = capturingLogger();
  stubChatProvider(turns);
  const outcome = await l3For({ dir: fx.dir, runner: realRunner(logger), logger, storage: fx.storage }).call(undefined)
    .then(() => "resolved", (err: Error) => err.message);
  return { ...fx, lines, outcome };
}

describe("Case I local FS: only persona.md is writable", () => {
  it("I the scene index is refused: unchanged byte for byte, generation failed, error logged", async () => {
    const { lines, outcome } = await runFsKeeping(".metadata/scene_index.json",
      [[{ name: "write", args: { path: ".metadata/scene_index.json", content: "[]" } }], null]);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(1);
  });

  it("I the checkpoint is refused: unchanged byte for byte, generation failed, error logged", async () => {
    const { dir, lines, outcome } = await runFsKeeping(".metadata/recall_checkpoint.json",
      [[{ name: "write", args: { path: ".metadata/recall_checkpoint.json", content: "{}" } }], null]);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(1);
    expect((await checkpointOf(dir)).memories_since_last_persona).toBe(4);
  });

  it("I a new file in dataDir is refused: it does not exist, generation failed, error logged", async () => {
    const { dir, lines, outcome } = await runFs([[{ name: "write", args: { path: "notes.md", content: "secret plan" } }], null]);
    expect(fs.existsSync(path.join(dir, "notes.md"))).toBe(false);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(1);
  });

  it("I a scene body edit is refused: unchanged byte for byte, generation failed, error logged", async () => {
    const { dir, lines, outcome } = await runFs([[{ name: "edit", args: { path: "scene_blocks/coding.md", edits: [{ oldText: "Coding scene.", newText: "Hijacked." }] } }], null]);
    expect(fs.readFileSync(path.join(dir, "scene_blocks", "coding.md"), "utf-8")).toBe(SCENE_BODY);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(1);
    expect(refusals(lines)[0]).not.toContain("Hijacked");
  });

  /** Run, and assert the file at `rel` kept every byte. */
  async function runFsKeeping(rel: string, turns: Array<ToolCallSpec[] | null>) {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const before = fs.readFileSync(path.join(dir, rel));
    const { logger, lines } = capturingLogger();
    stubChatProvider(turns);
    const outcome = await l3For({ dir, runner: realRunner(logger), logger }).call(undefined)
      .then(() => "resolved", (err: Error) => err.message);
    expect(fs.readFileSync(path.join(dir, rel)).equals(before)).toBe(true);
    return { dir, lines, outcome };
  }
});

describe("Case I-S in-memory StorageAdapter: only persona.md is writable", () => {
  it("I-S a scene body write and a checkpoint write are refused: both unchanged, generation failed, errors logged", async () => {
    const fx0 = await storageFixture();
    const sceneBefore = fx0.backend.text("scene_blocks/coding.md");
    const { backend, lines, outcome } = await runStorage([[
      { name: "write", args: { path: "scene_blocks/coding.md", content: "gone" } },
      { name: "write", args: { path: ".metadata/checkpoint.json", content: "{}" } },
    ], null]);
    expect(backend.text("scene_blocks/coding.md")).toBe(sceneBefore);
    expect(JSON.parse(backend.text(".metadata/checkpoint.json")!).memories_since_last_persona).toBe(4);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(2);
  });
});

describe("Case I paths are compared after the resolver", () => {
  it.each([["fs"], ["storage"]])("I ./persona.md is allowed (%s)", async (mode) => {
    const turns: Array<ToolCallSpec[] | null> = [[{ name: "write", args: { path: "./persona.md", content: NEW_PERSONA } }], null];
    const r = mode === "fs" ? await runFs(turns) : await runStorage(turns);
    expect(r.outcome).toBe("resolved");
    expect(refusals(r.lines)).toHaveLength(0);
  });

  it.each([
    ["fs", "scene_blocks/../persona.md"],
    ["fs", "scene_blocks/persona.md"],
    ["storage", "scene_blocks/../persona.md"],
    ["storage", "scene_blocks/persona.md"],
  ])("I %s path %s is refused and the generation fails", async (mode, p) => {
    const turns: Array<ToolCallSpec[] | null> = [[{ name: "write", args: { path: p, content: NEW_PERSONA } }], null];
    const r = mode === "fs" ? await runFs(turns) : await runStorage(turns);
    expect(r.outcome).toContain("Persona generation failed");
    expect(refusals(r.lines)).toHaveLength(1);
  });

  it("I a forbidden write followed by a valid persona.md write in the same generation still fails", async () => {
    const { dir, lines, outcome } = await runFs([
      [{ name: "write", args: { path: "notes.md", content: "x" } }],
      [{ name: "write", args: { path: "persona.md", content: NEW_PERSONA } }],
      null,
    ]);
    expect(outcome).toContain("Persona generation failed");
    expect(refusals(lines)).toHaveLength(1);
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

describe("Case I-F the tools reach runner.run through createL3Runner + PersonaGenerator", () => {
  it("I-F every runner.run call carries the persona tool set (write/edit wrapped, read kept)", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    stubChatProvider([[{ name: "write", args: { path: "persona.md", content: NEW_PERSONA } }], null]);
    const real = realRunner();
    const seen: LLMRunParams[] = [];
    const runner: LLMRunner = { run: (p) => { seen.push(p); return real.run(p); } };

    await l3For({ dir, runner }).call(undefined);

    expect(seen).toHaveLength(1);
    expect(Object.keys(seen[0].tools ?? {}).sort()).toEqual(["edit", "read", "write"]);
  });
});

describe("Case I positive controls", () => {
  it.each([["fs"], ["storage"]])("I a valid write of persona.md is generated and advances the checkpoint (%s)", async (mode) => {
    const turns: Array<ToolCallSpec[] | null> = [[{ name: "write", args: { path: "persona.md", content: NEW_PERSONA } }], null];
    if (mode === "fs") {
      const r = await runFs(turns);
      expect(r.outcome).toBe("resolved");
      expect(fs.readFileSync(path.join(r.dir, "persona.md"), "utf-8")).toContain("Also writes tests first.");
      expect((await checkpointOf(r.dir)).memories_since_last_persona).toBe(0);
    } else {
      const r = await runStorage(turns);
      expect(r.outcome).toBe("resolved");
      expect(r.backend.text("persona.md")).toContain("Also writes tests first.");
      expect((await checkpointOf(r.dir, r.storage)).memories_since_last_persona).toBe(0);
    }
  });

  it.each([["fs"], ["storage"]])("I a valid edit of persona.md is generated and advances the checkpoint (%s)", async (mode) => {
    const turns: Array<ToolCallSpec[] | null> = [[{ name: "edit", args: { path: "persona.md", edits: [{ oldText: "concise answers.", newText: "concise answers and tests." }] } }], null];
    if (mode === "fs") {
      const r = await runFs(turns);
      expect(r.outcome).toBe("resolved");
      expect(fs.readFileSync(path.join(r.dir, "persona.md"), "utf-8")).toContain("concise answers and tests.");
      expect((await checkpointOf(r.dir)).memories_since_last_persona).toBe(0);
    } else {
      const r = await runStorage(turns);
      expect(r.outcome).toBe("resolved");
      expect(r.backend.text("persona.md")).toContain("concise answers and tests.");
      expect((await checkpointOf(r.dir, r.storage)).memories_since_last_persona).toBe(0);
    }
  });
});
