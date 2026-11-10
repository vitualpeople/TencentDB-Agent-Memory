/**
 * Records the L3 persona-generation outcomes as they behave today (cases L18
 * and L19 of the memory-path failure table). The real L3 runner and
 * PersonaGenerator run in a temp dir; only the LLM is a fake.
 *
 *   L18 the persona LLM fails: generateLocalPersona returns "failed", the L3
 *       runner throws "Persona generation failed", and through the worker
 *       the task takes the L13 path (3 retries, then an in-process dead
 *       letter). The persona checkpoint is not advanced.
 *   L19 the LLM rewrites persona.md with the same body: "unchanged", and the
 *       runner still calls markPersonaGenerated (counter reset, time set);
 *       persona.md is rewritten with a fresh scene navigation appended.
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import type { LLMRunner } from "../src/core/types.js";
import { PipelineWorker } from "../src/services/pipeline-worker.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL3Runner } from "../src/utils/pipeline-factory.js";
import { providerError, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const PERSONA = "# Persona\n\nThe user prefers concise answers.";

/** Root data dir with one scene, an optional persona, and a checkpoint that triggers L3 (first scene block). */
async function fixture(withPersona: boolean): Promise<string> {
  const dir = tmpDir("l3-cases-");
  fs.mkdirSync(path.join(dir, "scene_blocks"), { recursive: true });
  fs.writeFileSync(path.join(dir, "scene_blocks", "coding.md"),
    ["---", "scene: coding", "created: 2026-11-01T09:00:00.000Z", "updated: 2026-11-01T10:00:00.000Z", "heat: 1", "---", "", "## Summary", "Coding scene.", ""].join("\n"),
    "utf-8");
  await writeSceneIndex(dir, [{ filename: "coding.md", summary: "Coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" }]);
  if (withPersona) fs.writeFileSync(path.join(dir, "persona.md"), PERSONA, "utf-8");
  const cp = new CheckpointManager(dir, silent);
  await cp.incrementScenesProcessed();
  await cp.incrementMemoriesSincePersona(4);
  return dir;
}

function l3For(dir: string, run: (p: { workspaceDir?: string }) => Promise<string>) {
  return createL3Runner({
    pluginDataDir: dir,
    cfg: parseConfig({}),
    openclawConfig: undefined,
    logger: silent,
    llmRunner: { run: vi.fn(run) } as unknown as LLMRunner,
  });
}

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); });

describe("L18 L3 persona LLM failure", () => {
  it("L18 the runner throws \"Persona generation failed\" with a 180 s LLM call, and the persona checkpoint stays put", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fixture(true);
    const calls: Array<Record<string, unknown>> = [];
    const l3 = l3For(dir, async (p) => { calls.push(p as Record<string, unknown>); throw providerError(500); });

    await expect(l3()).rejects.toThrow("[L3] Persona generation failed (scope=global)");

    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ taskId: "persona-generation", timeoutMs: 180_000, workspaceDir: dir });
    const cp = await new CheckpointManager(dir, silent).read();
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toBe(PERSONA);
  });

  it("L18 through the worker the L3 task is run 1 + 3 times (5 s, 15 s, 45 s) and then dead-lettered", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fixture(true);
    let llmCalls = 0;
    const l3 = l3For(dir, async () => { llmCalls++; throw providerError(503); });
    const backend = new LocalStateBackend();
    const worker = new PipelineWorker(backend, {
      async executeL1() {}, async executeL2() {},
      async executeL3() { await l3(); },
    }, { concurrency: 1, pollIntervalMs: 10, lockRenewIntervalMs: 60_000, retryBaseDelayMs: 5 }, silent);
    await worker.start();
    try {
      await backend.enqueueTask({ id: "L3-1", type: "L3", instanceId: "svc", sessionId: "proj:l3", agentId: "default", priority: 2, createdAt: Date.now(), data: { instanceId: "svc", agentId: "default" } });

      await vi.waitFor(() => { expect(worker.getMetrics().tasksDeadLettered).toBe(1); }, { timeout: 10_000, interval: 10 });

      expect(llmCalls).toBe(4);
      expect(worker.getMetrics()).toMatchObject({ tasksRetried: 3, tasksFailed: 4 });
      expect(worker.getDeadLetterQueue()[0].error).toContain("Persona generation failed");
    } finally {
      await worker.stop();
    }
  });
});

describe("L19 L3 output unchanged", () => {
  it("L19 the LLM rewrites persona.md with the same body -> markPersonaGenerated: counter 0, last_persona_time set, persona body kept", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fixture(true);
    let llmCalls = 0;
    const l3 = l3For(dir, async (p) => {
      llmCalls++;
      const file = path.join(p.workspaceDir!, "persona.md");
      fs.writeFileSync(file, PERSONA, "utf-8");
      const later = new Date(Date.now() + 5_000);
      fs.utimesSync(file, later, later); // a real write moves mtime; make sure it differs
      return "persona rewritten";
    });
    const before = Date.now();

    await expect(l3()).resolves.toBeUndefined();

    expect(llmCalls).toBe(1);
    const cp = await new CheckpointManager(dir, silent).read();
    expect(cp.memories_since_last_persona).toBe(0);
    expect(Date.parse(cp.last_persona_time)).toBeGreaterThanOrEqual(before - 1_000);
    // The body is kept; the file is still rewritten with a fresh scene navigation appended.
    const written = fs.readFileSync(path.join(dir, "persona.md"), "utf-8");
    expect(written.startsWith(`${PERSONA}\n\n`)).toBe(true);
    expect(written).toContain("Scene Navigation");
  });

  it("L19 control: an LLM that does not touch persona.md is \"failed\" (not unchanged) and the checkpoint is not marked", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fixture(true);
    const l3 = l3For(dir, async () => "I decided nothing needs to change");

    await expect(l3()).rejects.toThrow("Persona generation failed");

    const cp = await new CheckpointManager(dir, silent).read();
    expect(cp.memories_since_last_persona).toBe(4);
    expect(cp.last_persona_time).toBe("");
  });
});
