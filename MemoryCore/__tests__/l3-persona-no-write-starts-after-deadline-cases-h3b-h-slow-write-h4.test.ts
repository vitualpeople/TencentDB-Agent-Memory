/**
 * L3 persona: no write starts after the scope's deadline or after the
 * worker's abort (cases H3b, H-slow-write, H4). A write that already started
 * is allowed to finish; the step after it is refused before it starts.
 * Fake LLM, fake vector store, temp dirs / in-memory storage, no network.
 *
 *   H3b  control on the checkpoint lock chain: a guarded waiter behind the
 *        file-lock holder is refused after the holder ends, the waiter after
 *        it still runs; a distributed lock acquired past the deadline is
 *        refused before writeRaw and released.
 *   H-slow-write  the final persona.md write starts before the deadline and
 *        ends after it: it completes, profile sync and provenance do not
 *        start, the checkpoint stays.
 *   H4   the check sits before every external write: pullProfilesToLocal,
 *        the generator, syncLocalProfilesToStore, the provenance log and its
 *        refs, the persona write/edit tools (also after edit's read), and
 *        the four markPersonaGenerated call sites.
 */
import { createHash } from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPersonaWriteTools } from "../src/adapters/standalone/persona-tools.js";
import { writeGenerationProvenanceBestEffort } from "../src/core/memory-generation-log/best-effort.js";
import { MemoryGenerationLogStore } from "../src/core/memory-generation-log/store.js";
import { PersonaGenerator } from "../src/core/persona/persona-generator.js";
import { listLocalProfiles } from "../src/core/profile/profile-sync.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { CheckpointManager, CheckpointWriteRejectedError } from "../src/utils/checkpoint.js";
import {
  MemoryBackend,
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

const CTX = { l3TaskId: "L3-h4" };
const NEW_BODY = `${PERSONA}\n\nWritten by the model.`;

function freezeClock(): void {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-11-13T07:00:00.000Z"));
}

function advance(ms: number): void {
  vi.setSystemTime(new Date(Date.now() + ms));
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

afterEach(() => { vi.useRealTimers(); vi.unstubAllEnvs(); vi.restoreAllMocks(); });

/** A checkpoint lock whose acquire takes `ms` of fake time (past a short deadline). */
function slowCheckpointLock(ms: number) {
  return {
    lockKey: "h4-slow",
    maxWaitMs: 60_000,
    lock: {
      acquireLock: async () => { advance(ms); return true; },
      releaseLock: async () => {},
      renewLock: async () => true,
    },
  };
}

describe("Case H3b control: the guard inside the checkpoint locks", () => {
  function gate() {
    let open!: () => void;
    const opened = new Promise<void>((r) => { open = r; });
    let signalEntered!: () => void;
    const entered = new Promise<void>((r) => { signalEntered = r; });
    return { open, opened, entered, signalEntered };
  }

  it.each([["abort"], ["deadline"]])("H3b file lock: waiter 2 (guarded) refused by %s while queued behind holder 1, no writeRaw of its own; waiter 3 still runs after it", async (kind) => {
    const dir = await fsFixture();
    const g = gate();
    const holderLock = {
      acquireLock: async () => { g.signalEntered(); await g.opened; return true; },
      releaseLock: async () => {},
      renewLock: async () => true,
    };
    const holder = new CheckpointManager(dir, silent, undefined, { lock: holderLock, lockKey: "h3b-file", maxWaitMs: 10_000 });
    const waiter2 = new CheckpointManager(dir, silent);
    const waiter3 = new CheckpointManager(dir, silent);
    const raw2 = vi.spyOn(waiter2 as unknown as { writeRaw: () => Promise<void> }, "writeRaw");
    const order: string[] = [];

    const p1 = holder.markPersonaGenerated(5).then(() => { order.push("holder"); });
    await g.entered;
    const controller = new AbortController();
    const guard = kind === "abort" ? { signal: controller.signal } : { deadlineAt: Date.now() + 20 };
    const p2 = waiter2.markPersonaGenerated(7, guard).then(() => { order.push("waiter2"); }, (e: unknown) => { order.push("waiter2-rejected"); throw e; });
    const p3 = waiter3.markPersonaGenerated(9).then(() => { order.push("waiter3"); });
    if (kind === "abort") controller.abort(); else await sleep(40);
    g.open();

    await p1;
    await expect(p2).rejects.toBeInstanceOf(CheckpointWriteRejectedError);
    await p3;
    expect(raw2).not.toHaveBeenCalled();
    expect(order).toEqual(["holder", "waiter2-rejected", "waiter3"]);
    expect((await checkpointOf(dir)).last_persona_at).toBe(9);
  });

  it("H3b distributed lock: acquireLock pending past deadlineAt then resolving true -> refused before writeRaw, the lock is released", async () => {
    const dir = await fsFixture();
    const g = gate();
    const releaseLock = vi.fn(async () => {});
    const lock = {
      acquireLock: async () => { g.signalEntered(); await g.opened; return true; },
      releaseLock,
      renewLock: async () => true,
    };
    const cpm = new CheckpointManager(dir, silent, undefined, { lock, lockKey: "h3b-dist", maxWaitMs: 10_000 });
    const raw = vi.spyOn(cpm as unknown as { writeRaw: () => Promise<void> }, "writeRaw");

    const pending = cpm.markPersonaGenerated(7, { deadlineAt: Date.now() + 20 });
    await g.entered;
    await sleep(40);
    g.open();

    await expect(pending).rejects.toBeInstanceOf(CheckpointWriteRejectedError);
    expect(raw).not.toHaveBeenCalled();
    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

/** Storage fixture whose final persona.md write (after the model's) runs `during` while it is in flight. */
async function slowFinalWriteFixture(during: () => void) {
  const fx = await storageFixture();
  const backend = new MemoryBackend();
  for (const [k, v] of fx.backend.objects) backend.objects.set(k, v);
  let modelDone = false;
  const put = backend.putObject.bind(backend);
  backend.putObject = async (key, content, ...rest) => {
    const finalWrite = modelDone && key === "persona.md";
    if (finalWrite) modelDone = false;
    await (put as (...a: unknown[]) => Promise<void>)(key, content, ...rest);
    if (finalWrite) during();
  };
  const storage = new StorageAdapter(backend);
  const runner = scriptedRunner([async () => { await storage.writeFile("persona.md", NEW_BODY); modelDone = true; return "ok"; }]);
  return { dir: fx.dir, backend, storage, ...runner };
}

describe("Case H-slow-write the final write may finish, the next step may not start", () => {
  it("H-slow-write the final persona.md write crosses the deadline: it completes (navigation appended, no .tmp), syncLocalProfilesToStore and provenance never start, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    freezeClock();
    const fx = await slowFinalWriteFixture(() => advance(6_000));
    const syncProfiles = vi.fn(async () => {});
    const vectorStore = { isDegraded: () => false, syncProfiles };
    const logWrite = vi.spyOn(MemoryGenerationLogStore.prototype, "write");
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir: fx.dir, runner: fx.runner, storage: fx.storage, vectorStore, logger, cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined, CTX))
      .rejects.toThrow();

    const persona = fx.backend.text("persona.md")!;
    expect(persona.startsWith(`${NEW_BODY}\n\n`)).toBe(true);
    expect(persona).toContain("Scene Navigation");
    expect([...fx.backend.objects.keys()].filter((k) => k.includes(".tmp"))).toEqual([]);
    expect(syncProfiles).not.toHaveBeenCalled();
    expect(logWrite).not.toHaveBeenCalled();
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "deadline", invokedRunner: "true" });
    const cp = await checkpointOf(fx.dir, fx.storage);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });

  it("H-slow-write the worker aborts while the final write is in flight (no vector store): the provenance log never starts, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const controller = new AbortController();
    const fx = await slowFinalWriteFixture(() => controller.abort());
    const logWrite = vi.spyOn(MemoryGenerationLogStore.prototype, "write");
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir: fx.dir, runner: fx.runner, storage: fx.storage, logger }).call(undefined, { ...CTX, signal: controller.signal }))
      .rejects.toThrow();

    expect(fx.backend.text("persona.md")!.startsWith(`${NEW_BODY}\n\n`)).toBe(true);
    expect(logWrite).not.toHaveBeenCalled();
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "aborted" });
    expect((await checkpointOf(fx.dir, fx.storage)).last_persona_time).toBe("");
  });
});

describe("Case H4 before pullProfilesToLocal and before the generator", () => {
  it("H4 the signal is already aborted: pullProfiles never starts, the runner is never called", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    const pullProfiles = vi.fn(async () => []);
    const controller = new AbortController();
    controller.abort();
    const { runner, calls } = scriptedRunner([() => "unused"]);

    await expect(l3For({ dir, runner, vectorStore: { isDegraded: () => false, pullProfiles } }).call(undefined, { ...CTX, signal: controller.signal }))
      .rejects.toThrow();

    expect(pullProfiles).not.toHaveBeenCalled();
    expect(calls).toHaveLength(0);
  });

  it("H4 the deadline passes in the scope's first steps (181 s of 180 s): pullProfiles never starts, the runner is never called", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const fx = await storageFixture();
    freezeClock();
    const storage = slowSceneIndexStorage(fx.backend, () => advance(181_000));
    const pullProfiles = vi.fn(async () => []);
    const { runner, calls } = scriptedRunner([() => "unused"]);

    await expect(l3For({ dir: fx.dir, runner, storage, vectorStore: { isDegraded: () => false, pullProfiles } }).call(undefined, CTX))
      .rejects.toThrow();

    expect(pullProfiles).not.toHaveBeenCalled();
    expect(calls).toHaveLength(0);
  });

  it("H4 the deadline passes during pullProfilesToLocal: the generator never starts (runner never called), checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    vi.stubEnv("TDAI_PERSONA_TIMEOUT_MS", undefined);
    const dir = await fsFixture();
    freezeClock();
    const pullProfiles = vi.fn(async () => { advance(181_000); return listLocalProfiles(dir); });
    const { runner, calls } = scriptedRunner([() => "unused"]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, vectorStore: { isDegraded: () => false, pullProfiles } }).call(undefined, CTX))
      .rejects.toThrow();

    expect(pullProfiles).toHaveBeenCalledTimes(1);
    expect(calls).toHaveLength(0);
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "deadline", invokedRunner: "false" });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });
});

describe("Case H4 provenance: the log, then the refs", () => {
  it.each([["deadline"], ["abort"]])("H4 %s reached while writeLog() runs: writeRefs() never starts", async (kind) => {
    freezeClock();
    const controller = new AbortController();
    const writeRefs = vi.fn(async () => {});
    const writeLog = vi.fn(async () => { if (kind === "abort") controller.abort(); else advance(10_000); });

    await writeGenerationProvenanceBestEffort({
      layer: "l3",
      logger: silent,
      writeLog,
      writeRefs,
      guard: { signal: controller.signal, deadlineAt: Date.now() + 5_000 },
    } as Parameters<typeof writeGenerationProvenanceBestEffort>[0]);

    expect(writeLog).toHaveBeenCalledTimes(1);
    expect(writeRefs).not.toHaveBeenCalled();
  });

  it("H4 control: without a guard (L1/L2 callers) both writes run", async () => {
    const writeLog = vi.fn(async () => {});
    const writeRefs = vi.fn(async () => {});
    await writeGenerationProvenanceBestEffort({ layer: "l2", logger: silent, writeLog, writeRefs });
    expect(writeLog).toHaveBeenCalledTimes(1);
    expect(writeRefs).toHaveBeenCalledTimes(1);
  });
});

describe("Case H4 the persona write/edit tools", () => {
  it("H4 write with an aborted signal: refused before inner.execute, persona.md unchanged, writesObserved() = 0", async () => {
    const dir = await fsFixture();
    const controller = new AbortController();
    controller.abort();
    const toolset = createPersonaWriteTools({ workspaceDir: dir, logger: silent, guard: { signal: controller.signal } } as Parameters<typeof createPersonaWriteTools>[0]);
    const write = toolset.tools.write as { execute: (a: unknown, o?: unknown) => Promise<unknown> };

    const out = String(await write.execute({ path: "persona.md", content: NEW_BODY }, {}));

    expect(out).toMatch(/error/i);
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toBe(PERSONA);
    expect((toolset as { writesObserved?: () => number }).writesObserved?.()).toBe(0);
    expect(toolset.violated()).toBe(false);
  });

  it("H4 write past deadlineAt: refused, persona.md unchanged; a write before it is counted once", async () => {
    const dir = await fsFixture();
    freezeClock();
    const toolset = createPersonaWriteTools({ workspaceDir: dir, logger: silent, guard: { deadlineAt: Date.now() + 1_000 } } as Parameters<typeof createPersonaWriteTools>[0]);
    const write = toolset.tools.write as { execute: (a: unknown, o?: unknown) => Promise<unknown> };

    expect(String(await write.execute({ path: "persona.md", content: NEW_BODY }, {}))).toContain("success");
    advance(1_000);
    expect(String(await write.execute({ path: "persona.md", content: "late" }, {}))).toMatch(/error/i);

    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toBe(NEW_BODY);
    expect((toolset as { writesObserved?: () => number }).writesObserved?.()).toBe(1);
  });

  it("H4 storage edit: the abort lands while edit reads persona.md -> storage.writeFile never starts, writesObserved() = 0", async () => {
    const fx = await storageFixture();
    const controller = new AbortController();
    const getObject = fx.backend.getObject.bind(fx.backend);
    fx.backend.getObject = async (key: string) => {
      const o = await getObject(key);
      if (key === "persona.md") controller.abort();
      return o;
    };
    const putSpy = vi.spyOn(fx.backend, "putObject");
    const toolset = createPersonaWriteTools({ workspaceDir: fx.dir, storage: fx.storage, logger: silent, guard: { signal: controller.signal } } as Parameters<typeof createPersonaWriteTools>[0]);
    const edit = toolset.tools.edit as { execute: (a: unknown, o?: unknown) => Promise<unknown> };

    const out = String(await edit.execute({ path: "persona.md", edits: [{ oldText: "concise answers.", newText: "long answers." }] }, {}));

    expect(out).toMatch(/error/i);
    expect(putSpy).not.toHaveBeenCalled();
    expect(fx.backend.text("persona.md")).toBe(PERSONA);
    expect((toolset as { writesObserved?: () => number }).writesObserved?.()).toBe(0);
  });

  it("H4 local FS edit: the abort lands while edit reads persona.md -> fs writeFile never starts, writesObserved() = 0", async () => {
    const dir = await fsFixture();
    const controller = new AbortController();
    const readFile = fsp.readFile.bind(fsp);
    vi.spyOn(fsp, "readFile").mockImplementation((async (...args: Parameters<typeof fsp.readFile>) => {
      const out = await readFile(...args);
      if (String(args[0]).endsWith("persona.md")) controller.abort();
      return out;
    }) as typeof fsp.readFile);
    const writeSpy = vi.spyOn(fsp, "writeFile");
    const toolset = createPersonaWriteTools({ workspaceDir: dir, logger: silent, guard: { signal: controller.signal } } as Parameters<typeof createPersonaWriteTools>[0]);
    const edit = toolset.tools.edit as { execute: (a: unknown, o?: unknown) => Promise<unknown> };

    const out = String(await edit.execute({ path: "persona.md", edits: [{ oldText: "concise answers.", newText: "long answers." }] }, {}));

    expect(controller.signal.aborted).toBe(true);
    expect(out).toMatch(/error/i);
    expect(writeSpy).not.toHaveBeenCalled();
    expect(fs.readFileSync(path.join(dir, "persona.md"), "utf-8")).toBe(PERSONA);
    expect((toolset as { writesObserved?: () => number }).writesObserved?.()).toBe(0);
  });
});

describe("Case H4 the four markPersonaGenerated call sites", () => {
  it("H4 site 1 (generator unchanged): the lock is acquired past the deadline -> refused, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    freezeClock();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, PERSONA); return "same body"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, checkpointLock: slowCheckpointLock(10_000), cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined, CTX))
      .rejects.toThrow();

    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "checkpoint-rejected" });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });

  it("H4 site 2 (generated, no changed profile): refused past the deadline, no provenance, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    // The pulled baseline already carries the md5 of the body the model will
    // write (its content fails the md5 check, so the local copy is kept): after
    // the generation no profile differs from the baseline.
    const md5 = (t: string) => createHash("md5").update(t).digest("hex");
    const pullProfiles = vi.fn(async () => (await listLocalProfiles(dir)).map((r) =>
      r.type === "l3" ? { ...r, content: "stale remote copy", contentMd5: md5(NEW_BODY) } : r));
    const syncProfiles = vi.fn(async () => {});
    const logWrite = vi.spyOn(MemoryGenerationLogStore.prototype, "write");
    freezeClock();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, NEW_BODY); return "ok"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({
      dir, runner, logger, checkpointLock: slowCheckpointLock(10_000), cfg: { persona: { timeoutMs: 5_000 } },
      vectorStore: { isDegraded: () => false, pullProfiles, syncProfiles },
    }).call(undefined, CTX)).rejects.toThrow();

    expect(syncProfiles).not.toHaveBeenCalled();
    expect(logWrite).not.toHaveBeenCalled();
    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "checkpoint-rejected" });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });

  it("H4 site 3 (generated, after provenance): refused past the deadline, checkpoint unchanged", async () => {
    vi.stubEnv("TDAI_PERSONA_TRIGGER_MODE", undefined);
    const dir = await fsFixture();
    freezeClock();
    const { runner } = scriptedRunner([(p) => { writePersonaFs(p.workspaceDir!, NEW_BODY); return "ok"; }]);
    const { logger, lines } = capturingLogger();

    await expect(l3For({ dir, runner, logger, checkpointLock: slowCheckpointLock(10_000), cfg: { persona: { timeoutMs: 5_000 } } }).call(undefined, CTX))
      .rejects.toThrow();

    expect(endFields(endLines(lines)[0])).toMatchObject({ outcome: "failed", reason: "checkpoint-rejected" });
    expect((await checkpointOf(dir)).last_persona_time).toBe("");
  });

  it("H4 site 4 (PersonaGenerator.generate): the abort lands after the final write -> markPersonaGenerated refused, checkpoint unchanged, generate() returns false", async () => {
    const fx = await storageFixture();
    const controller = new AbortController();
    const backend = new MemoryBackend();
    for (const [k, v] of fx.backend.objects) backend.objects.set(k, v);
    let modelDone = false;
    const put = backend.putObject.bind(backend);
    backend.putObject = async (key, content, ...rest) => {
      await (put as (...a: unknown[]) => Promise<void>)(key, content, ...rest);
      if (modelDone && key === "persona.md") controller.abort();
    };
    const storage = new StorageAdapter(backend);
    const { runner } = scriptedRunner([async () => { await storage.writeFile("persona.md", NEW_BODY); modelDone = true; return "ok"; }]);
    const gen = new PersonaGenerator({ dataDir: fx.dir, config: undefined, llmRunner: runner, logger: silent, storage, abortSignal: controller.signal });

    await expect(gen.generate("test")).resolves.toBe(false);

    expect(backend.text("persona.md")!.startsWith(`${NEW_BODY}\n\n`)).toBe(true);
    const cp = await checkpointOf(fx.dir, storage);
    expect(cp.last_persona_time).toBe("");
    expect(cp.memories_since_last_persona).toBe(4);
  });
});
