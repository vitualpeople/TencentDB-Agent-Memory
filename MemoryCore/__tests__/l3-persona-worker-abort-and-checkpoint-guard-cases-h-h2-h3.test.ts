/**
 * L3 persona: the worker's abort signal (cases H, H2, H3). Fake LLM, no network.
 *
 *   H  the worker's signal goes through the real TracedTaskExecutor ->
 *      createL3Runner -> PersonaGenerator into both runner.run calls; an
 *      abort ends the generation "failed" without another call.
 *   H2 an abort after runner.run returned and before the final write: the
 *      generator returns "failed", persona.md stays byte for byte as the
 *      model left it, both checkpoint fields stay.
 *   H3 the checkpoint lock is held; the abort (or the deadline) passes while
 *      markPersonaGenerated waits for it: it throws before the write and the
 *      checkpoint stays.
 */
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TracedTaskExecutor } from "../src/core/report/traced-task-executor.js";
import type { TaskPayload } from "../src/core/state/types.js";
import type { LLMRunParams } from "../src/core/types.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import {
  PERSONA,
  checkpointOf,
  fsFixture,
  l3For,
  scriptedRunner,
  silent,
  writePersonaFs,
} from "./helpers/l3-persona-harness.js";

const TASK: TaskPayload = { id: "L3-h", type: "L3", instanceId: "svc", sessionId: "proj:h", agentId: "default", priority: 2, createdAt: 0, data: {} } as TaskPayload;

function abortError(): Error {
  return Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
}

afterEach(() => { vi.unstubAllEnvs(); });

describe("Case H worker abort signal reaches the persona LLM calls", () => {
  it("H through the real TracedTaskExecutor, both runner.run calls (first + nudge) get the worker's signal", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const { runner, calls } = scriptedRunner([
      () => "no write",
      (p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nNew.`); return "ok"; },
    ]);
    const l3 = l3For({ dir, runner });
    const traced = new TracedTaskExecutor({
      async executeL1() {}, async executeL2() {},
      async executeL3(_task: TaskPayload, signal?: AbortSignal) { await l3({ signal }); },
    });
    const controller = new AbortController();

    const outcome = await traced.executeL3(TASK, controller.signal).then(() => "resolved", (e: Error) => e.message);

    expect(calls[0].abortSignal).toBe(controller.signal);
    expect(calls).toHaveLength(2);
    expect(calls[1].abortSignal).toBe(controller.signal);
    expect(outcome).toBe("resolved");
  });

  it("H an abort during call 1 ends the generation failed with no nudge call, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const controller = new AbortController();
    const { runner, calls } = scriptedRunner([(p: LLMRunParams) => {
      controller.abort();
      if (p.abortSignal?.aborted) throw abortError();
      return "returned without writing";
    }]);
    const l3 = l3For({ dir, runner });
    const traced = new TracedTaskExecutor({
      async executeL1() {}, async executeL2() {},
      async executeL3(_task: TaskPayload, signal?: AbortSignal) { await l3({ signal }); },
    });

    await expect(traced.executeL3(TASK, controller.signal)).rejects.toThrow("Persona generation failed");

    expect(calls).toHaveLength(1);
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});

describe("Case H2 abort between runner.run and the final write", () => {
  it("H2 persona.md equals, byte for byte, the snapshot taken right after runner.run returned; checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const controller = new AbortController();
    const file = path.join(dir, "persona.md");
    let snapshot: Buffer | undefined;
    const inner = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, `${PERSONA}\n\nWritten by the model.`); return "ok"; }]);
    const runner = {
      async run(p: LLMRunParams) {
        const out = await inner.runner.run(p);
        snapshot = fs.readFileSync(file);
        controller.abort();
        return out;
      },
    };

    await expect(l3For({ dir, runner }).call(undefined, { signal: controller.signal }))
      .rejects.toThrow("Persona generation failed");

    expect(snapshot).toBeDefined();
    expect(fs.readFileSync(file).equals(snapshot!)).toBe(true);
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});

describe("Case H3 abort or deadline while waiting for the checkpoint lock", () => {
  function heldLock() {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    let waiting!: () => void;
    const entered = new Promise<void>((r) => { waiting = r; });
    const lock = {
      acquireLock: async () => { waiting(); await gate; return true; },
      releaseLock: async () => {},
      renewLock: async () => true,
    };
    return { lock, release, entered };
  }

  it("H3 abort while waiting: markPersonaGenerated throws before writeRaw, checkpoint unchanged", async () => {
    const dir = await fsFixture();
    const { lock, release, entered } = heldLock();
    const cpm = new CheckpointManager(dir, silent, undefined, { lock, lockKey: "h3", maxWaitMs: 10_000 });
    const before = fs.readFileSync(path.join(dir, ".metadata", "recall_checkpoint.json"), "utf-8");
    const controller = new AbortController();

    const pending = cpm.markPersonaGenerated(7, { signal: controller.signal } as never);
    await entered;
    controller.abort();
    release();

    await expect(pending).rejects.toThrow(/checkpoint write rejected/);
    expect(fs.readFileSync(path.join(dir, ".metadata", "recall_checkpoint.json"), "utf-8")).toBe(before);
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });

  it("H3 deadline passes while waiting: markPersonaGenerated throws before writeRaw, checkpoint unchanged", async () => {
    const dir = await fsFixture();
    const { lock, release, entered } = heldLock();
    const cpm = new CheckpointManager(dir, silent, undefined, { lock, lockKey: "h3d", maxWaitMs: 10_000 });

    const pending = cpm.markPersonaGenerated(7, { deadlineAt: Date.now() + 30 } as never);
    await entered;
    await new Promise((r) => setTimeout(r, 60));
    release();

    await expect(pending).rejects.toThrow(/checkpoint write rejected/);
    const cp = await checkpointOf(dir);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});
