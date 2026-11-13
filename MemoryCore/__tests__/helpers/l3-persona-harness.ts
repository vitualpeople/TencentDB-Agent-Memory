/**
 * Shared fakes for the L3 persona tests: a data dir (local FS or an in-memory
 * StorageAdapter) with one scene, an optional persona and a checkpoint that
 * triggers L3; a logger that keeps every line; a scripted LLMRunner; and a
 * fake OpenAI-compatible provider (global fetch stub) so the real
 * StandaloneLLMRunner runs its own tool loop without any network.
 *
 * Only modules that already exist on the base commit are imported here, so a
 * test that fails on the base commit fails on behaviour, not on an import.
 */
import fs from "node:fs";
import path from "node:path";
import { vi } from "vitest";
import { parseConfig } from "../../src/config.js";
import { writeSceneIndex } from "../../src/core/scene/scene-index.js";
import { StorageAdapter } from "../../src/core/storage/adapter.js";
import type { IStorageBackend, ListResult, StorageObject } from "../../src/core/storage/types.js";
import type { LLMRunner, LLMRunParams } from "../../src/core/types.js";
import { StandaloneLLMRunner } from "../../src/adapters/standalone/llm-runner.js";
import { ProviderHealthSink } from "../../src/core/report/provider-health-telemetry.js";
import { CheckpointManager } from "../../src/utils/checkpoint.js";
import { createL3Runner } from "../../src/utils/pipeline-factory.js";
import { silent, tmpDir } from "./l1-spike-harness.js";

export { silent, tmpDir };

export const PERSONA = "# Persona\n\nThe user prefers concise answers.";
export const SCENE_FILE = "coding.md";
export const SCENE_BODY = ["---", "scene: coding", "created: 2026-11-01T09:00:00.000Z", "updated: 2026-11-01T10:00:00.000Z", "heat: 1", "---", "", "## Summary", "Coding scene.", ""].join("\n");
const SCENE_ENTRY = { filename: SCENE_FILE, summary: "Coding scene", heat: 1, created: "2026-11-01T09:00:00.000Z", updated: "2026-11-01T10:00:00.000Z" };

/** In-memory object store; every put gets a strictly newer lastModified. */
export class MemoryBackend implements IStorageBackend {
  readonly type = "cos" as const;
  readonly objects = new Map<string, { content: Buffer; lastModified: Date }>();
  private tick = Date.parse("2026-11-01T00:00:00.000Z");
  async putObject(key: string, content: string | Buffer): Promise<void> {
    this.tick += 1_000;
    this.objects.set(key, { content: Buffer.from(content), lastModified: new Date(this.tick) });
  }
  async appendObject(key: string, content: string | Buffer): Promise<void> {
    const prev = this.objects.get(key)?.content ?? Buffer.alloc(0);
    await this.putObject(key, Buffer.concat([prev, Buffer.from(content)]));
  }
  async getObject(key: string): Promise<StorageObject | null> {
    const o = this.objects.get(key);
    return o ? { key, content: o.content, lastModified: o.lastModified, size: o.content.length } : null;
  }
  async exists(key: string): Promise<boolean> { return this.objects.has(key); }
  async listObjects(prefix: string, opts?: { recursive?: boolean }): Promise<ListResult> {
    const entries = [...this.objects.entries()]
      .filter(([k]) => k.startsWith(prefix) && (opts?.recursive || !k.slice(prefix.length).includes("/")))
      .map(([k, o]) => ({ key: k, size: o.content.length, lastModified: o.lastModified, isDirectory: false }));
    return { entries, total: entries.length };
  }
  async deleteObject(key: string): Promise<void> { this.objects.delete(key); }
  async deleteByPrefix(prefix: string): Promise<number> {
    let n = 0;
    for (const k of [...this.objects.keys()]) if (k.startsWith(prefix)) { this.objects.delete(k); n++; }
    return n;
  }
  text(key: string): string | undefined { return this.objects.get(key)?.content.toString("utf-8"); }
}

/** Local data dir with one scene, an optional persona, and a checkpoint that triggers L3 (first scene block). */
export async function fsFixture(persona: string | undefined = PERSONA): Promise<string> {
  const dir = tmpDir("l3-persona-");
  fs.mkdirSync(path.join(dir, "scene_blocks"), { recursive: true });
  fs.writeFileSync(path.join(dir, "scene_blocks", SCENE_FILE), SCENE_BODY, "utf-8");
  await writeSceneIndex(dir, [SCENE_ENTRY]);
  if (persona !== undefined) fs.writeFileSync(path.join(dir, "persona.md"), persona, "utf-8");
  const cp = new CheckpointManager(dir, silent);
  await cp.incrementScenesProcessed();
  await cp.incrementMemoriesSincePersona(4);
  return dir;
}

/** Same fixture in an in-memory StorageAdapter (service mode). */
export async function storageFixture(persona: string | undefined = PERSONA): Promise<{ dir: string; storage: StorageAdapter; backend: MemoryBackend }> {
  const dir = tmpDir("l3-persona-storage-");
  const backend = new MemoryBackend();
  const storage = new StorageAdapter(backend);
  await storage.writeFile(`scene_blocks/${SCENE_FILE}`, SCENE_BODY);
  await writeSceneIndex(dir, [SCENE_ENTRY], storage);
  if (persona !== undefined) await storage.writeFile("persona.md", persona);
  const cp = new CheckpointManager(dir, silent, storage);
  await cp.incrementScenesProcessed();
  await cp.incrementMemoriesSincePersona(4);
  return { dir, storage, backend };
}

export interface LogLine { level: "debug" | "info" | "warn" | "error"; msg: string }

export function capturingLogger(): { logger: typeof silent; lines: LogLine[] } {
  const lines: LogLine[] = [];
  const at = (level: LogLine["level"]) => (msg: string) => { lines.push({ level, msg }); };
  return { logger: { debug: at("debug"), info: at("info"), warn: at("warn"), error: at("error") }, lines };
}

/** The end-of-generation lines of a run. */
export function endLines(lines: LogLine[]): string[] {
  return lines.filter((l) => l.msg.includes("persona_generation_end")).map((l) => l.msg);
}

/** key=value fields of one end line. */
export function endFields(line: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of line.matchAll(/(\w+)=(\S+)/g)) out[m[1]] = m[2];
  return out;
}

/** Scripted LLMRunner: call i runs script[i] (the last entry repeats). */
export function scriptedRunner(script: Array<(p: LLMRunParams) => Promise<string> | string>) {
  const calls: LLMRunParams[] = [];
  const runner = {
    run: vi.fn(async (p: LLMRunParams) => {
      calls.push(p);
      const step = script[Math.min(calls.length - 1, script.length - 1)];
      return step(p);
    }),
  };
  return { runner: runner as unknown as LLMRunner, calls };
}

/** Write persona.md the way a tool call would, with an mtime that surely moved. */
export function writePersonaFs(dir: string, content: string): void {
  const file = path.join(dir, "persona.md");
  fs.writeFileSync(file, content, "utf-8");
  const later = new Date(Date.now() + 5_000);
  fs.utimesSync(file, later, later);
}

export function l3For(opts: {
  dir: string;
  runner: LLMRunner;
  cfg?: Record<string, unknown>;
  storage?: StorageAdapter;
  logger?: typeof silent;
  personaTelemetry?: unknown;
  checkpointLock?: unknown;
}) {
  return createL3Runner({
    pluginDataDir: opts.dir,
    cfg: parseConfig(opts.cfg ?? {}),
    openclawConfig: undefined,
    logger: opts.logger ?? silent,
    llmRunner: opts.runner,
    storage: opts.storage,
    ...(opts.personaTelemetry ? { personaTelemetry: opts.personaTelemetry } : {}),
    ...(opts.checkpointLock ? { checkpointLock: opts.checkpointLock } : {}),
  } as Parameters<typeof createL3Runner>[0]);
}

export async function checkpointOf(dir: string, storage?: StorageAdapter) {
  return new CheckpointManager(dir, silent, storage).read();
}

/** A sink that only records what it is given (duck-typed, no import of the real sink). */
export function recordingSink() {
  const records: Array<{ invokedRunner: boolean; nudged: boolean; nudgeSucceeded: boolean; overLimit: boolean }> = [];
  return { records, record: (r: (typeof records)[number]) => { records.push(r); } };
}

// ── Fake OpenAI-compatible provider for the real StandaloneLLMRunner ──

export interface ToolCallSpec { name: "write" | "edit" | "read"; args: Record<string, unknown> }

/**
 * Stub global fetch with a chat/completions endpoint. Each request takes the
 * next turn: a list of tool calls, or `null` for a final text answer. When the
 * script runs out every request answers with text.
 */
export function stubChatProvider(turns: Array<ToolCallSpec[] | null>) {
  const requests: Array<Record<string, unknown>> = [];
  let n = 0;
  vi.stubGlobal("fetch", vi.fn(async (_url: unknown, init?: { body?: unknown }) => {
    requests.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
    const turn = n < turns.length ? turns[n] : null;
    n++;
    const message = turn
      ? {
          role: "assistant",
          content: null,
          tool_calls: turn.map((c, i) => ({
            id: `call_${n}_${i}`,
            type: "function",
            function: { name: c.name, arguments: JSON.stringify(c.args) },
          })),
        }
      : { role: "assistant", content: "done" };
    const body = {
      id: `chatcmpl-${n}`,
      object: "chat.completion",
      created: 0,
      model: "test-model",
      choices: [{ index: 0, message, finish_reason: turn ? "tool_calls" : "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  }));
  return { requests };
}

/** Real StandaloneLLMRunner with tools on, its own health sink, no key. */
export function realRunner(logger: typeof silent = silent): LLMRunner {
  return new StandaloneLLMRunner({
    config: { baseUrl: "https://llm.invalid/v1", apiKey: "test-key-not-a-secret", model: "test-model" },
    enableTools: true,
    logger,
    health: new ProviderHealthSink(),
  });
}
