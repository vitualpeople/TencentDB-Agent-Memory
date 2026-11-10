/**
 * Shared fakes for the L1 failure-path tests: an in-memory L0/L1 store that
 * implements only what createL1Runner and writeMemory touch, a scripted LLM
 * runner, a minimal plugin config, and a way to reach the real gateway
 * TaskExecutor and the real local timer-scanner callback without booting the
 * whole gateway.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { vi } from "vitest";
import type { LLMRunner } from "../../src/core/types.js";
import type { LocalStateBackend } from "../../src/core/state/local-backend.js";
import type { TaskExecutor } from "../../src/services/pipeline-worker.js";

export const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

export interface L0Row {
  id: string;
  sessionId: string;
  userId?: string;
  agentId?: string;
  teamId?: string;
  role?: "user" | "assistant";
  content: string;
  timestamp: number;
  recordedAtMs: number;
}

export function tmpDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/**
 * Fake IMemoryStore: queryL0GroupedBySessionId mirrors the SQLite contract
 * (rows with recorded_at_ms > cursor, oldest first, LIMIT, grouped by the
 * isolation tuple + sessionId); upsertL1 records every L1 write.
 */
export function makeL0Store(rows: L0Row[]) {
  const l1Records: Array<{ id: string; content: string; sessionId?: string }> = [];
  const queries: Array<number | undefined> = [];
  const store = {
    isDegraded: () => false,
    queryL0GroupedBySessionId(_sessionKey: string, after?: number, limit = 50) {
      queries.push(after);
      const picked = rows
        .filter((r) => r.recordedAtMs > (after ?? 0))
        .sort((a, b) => a.recordedAtMs - b.recordedAtMs)
        .slice(0, limit);
      const groups = new Map<string, { sessionId: string; teamId?: string; userId: string; agentId: string; messages: Array<{ id: string; role: string; content: string; timestamp: number; recordedAtMs: number }> }>();
      for (const r of picked) {
        const key = `${r.userId ?? "u"}\u0000${r.agentId ?? "a"}\u0000${r.sessionId}`;
        let g = groups.get(key);
        if (!g) {
          g = { sessionId: r.sessionId, teamId: r.teamId, userId: r.userId ?? "u", agentId: r.agentId ?? "a", messages: [] };
          groups.set(key, g);
        }
        g.messages.push({ id: r.id, role: r.role ?? "user", content: r.content, timestamp: r.timestamp, recordedAtMs: r.recordedAtMs });
      }
      return [...groups.values()];
    },
    async upsertL1(record: { id: string; content: string; sessionId?: string }) {
      l1Records.push(record);
      return true;
    },
  };
  return { store, l1Records, queries };
}

/** Plugin config fields createL1Runner reads; dedup off so writes are 1:1. */
export function makeCfg() {
  return {
    extraction: { enableDedup: false, maxMemoriesPerSession: 10, model: undefined, promptMode: "chat" },
    embedding: { conflictRecallTopK: 5, captureTimeoutMs: 1000, timeoutMs: 1000 },
  } as never;
}

/** One-scene extraction response whose single memory names `tag`. */
export function sceneJson(tag: string, messageIds: string[]): string {
  return JSON.stringify([
    {
      scene_name: `scene-${tag}`,
      message_ids: messageIds,
      memories: [{ content: `remembered fact from ${tag}`, type: "episodic", priority: 60, source_message_ids: messageIds }],
    },
  ]);
}

/** Error shaped like the standalone runner's HTTP failure (has a status). */
export function providerError(status: number): Error {
  return Object.assign(new Error(`LLM API call failed: HTTP ${status}`), { status });
}

/** Scripted LLMRunner: `answer(prompt, call#)` returns text or throws. */
export function scriptedLlm(answer: (prompt: string, call: number) => string | Promise<string>) {
  let calls = 0;
  const run = vi.fn(async (p: { prompt: string }) => answer(p.prompt, ++calls));
  return { runner: { run } as unknown as LLMRunner, run };
}

/** Sentence long enough to pass the L1 quality gate. */
export function msg(tag: string): string {
  return `The user said that ${tag} matters for the quarterly portfolio review next week.`;
}

/** Captures the `local` options handed to createStateBackend by startIntegratedServices. */
export const STOP_AFTER_BACKEND = "stop-after-state-backend";

/**
 * Builds a TdaiGateway with only the fields the timer callback and
 * buildTaskExecutor read. `runL1` stands in for core.runL1WithStore.
 */
export async function gatewayHarness(opts: {
  TdaiGateway: { prototype: object };
  captured: () => { local?: { onTimerExpired?: (e: { instanceId: string; member: string; fireAtMs: number }) => void } } | undefined;
  LocalStateBackendCtor: new (o?: { onTimerExpired?: (e: never) => void }) => LocalStateBackend;
  instanceId: string;
  baseDir: string;
  /** `slice` is what executeL1 hands runL1WithStore for a frozen slice (bound or skip). */
  runL1: (sessionKey: string, slice?: { sliceUpperBoundMs?: number; skipThroughMs?: number }) => Promise<{ storedCount: number; hasMore: boolean; hasFullBacklog: boolean; profileScopes: string[] }>;
}) {
  const gw = Object.create(opts.TdaiGateway.prototype) as Record<string, unknown>;
  Object.assign(gw, {
    config: { deployMode: "standalone", instanceId: opts.instanceId, stateBackend: "local", data: { baseDir: opts.baseDir }, redis: {} },
    logger: silent,
    core: {
      setInstanceId: () => {},
      runL1WithStore: async (sessionKey: string, _store: unknown, _embedding: unknown, _storage: unknown, _lock: unknown, slice?: { sliceUpperBoundMs?: number; skipThroughMs?: number }) =>
        ({ ...(await opts.runL1(sessionKey, slice)), creditUsed: 0 }),
    },
    configProvider: {},
    storePool: { mode: "sqlite", getStore: async () => ({ store: {}, embedding: undefined }) },
  });
  await (gw as { startIntegratedServices(): Promise<void> }).startIntegratedServices().catch((err: Error) => {
    if (err.message !== STOP_AFTER_BACKEND) throw err;
  });
  const onTimerExpired = opts.captured()?.local?.onTimerExpired;
  if (!onTimerExpired) throw new Error("timer callback not captured");
  const backend = new opts.LocalStateBackendCtor({ onTimerExpired: onTimerExpired as never });
  gw.stateBackend = backend;
  const executor = (gw as { buildTaskExecutor(): TaskExecutor }).buildTaskExecutor();
  return { gw, backend, executor, onTimerExpired };
}

/** Timers currently armed on a LocalStateBackend (member -> fireAtMs). */
export function armedTimers(backend: LocalStateBackend): Map<string, number> {
  const out = new Map<string, number>();
  for (const t of (backend as unknown as { timers: Map<string, { member: string; fireAtMs: number }> }).timers.values()) {
    out.set(t.member, t.fireAtMs);
  }
  return out;
}

/** Real setTimeout, captured before any test installs fake timers. */
const realSetTimeout = globalThis.setTimeout;
/** Let real I/O (checkpoint and JSONL fs writes) finish while the fake clock stands still. */
async function settleRealIo(): Promise<void> {
  for (let i = 0; i < 3; i++) await new Promise((r) => realSetTimeout(r, 2));
}

/** Advance fake timers in steps, letting real I/O settle between steps. */
export async function advance(ms: number, step = 500): Promise<void> {
  for (let done = 0; done < ms; done += step) {
    await vi.advanceTimersByTimeAsync(Math.min(step, ms - done));
    await settleRealIo();
  }
}

/** Advance fake time until `cond()` holds or `maxMs` elapses. */
export async function advanceUntil(cond: () => boolean, maxMs: number, step = 500): Promise<boolean> {
  for (let done = 0; done < maxMs; done += step) {
    await settleRealIo();
    if (cond()) return true;
    await vi.advanceTimersByTimeAsync(step);
  }
  await settleRealIo();
  return cond();
}
