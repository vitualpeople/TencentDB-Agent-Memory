/**
 * Delayed re-run of pipeline tasks that failed on an LLM quota / rate limit.
 *
 * PipelineWorker hands a task here instead of retrying it: the task is
 * written to a QuotaDeferralStore, then enqueued again once `delayMs` has
 * passed. Its retry count is left as it was, so a quota window never walks a
 * task into the dead letter. The entry stays in the store until the re-run
 * settles (settle()), so a restart between "enqueued" and "done" re-arms it;
 * the local backend's queue is in RAM, so the restart cannot also replay it.
 */
import type { TaskPayload } from "../core/state/types.js";
import { InMemoryQuotaDeferralStore, type QuotaDeferralEntry, type QuotaDeferralStore } from "./quota-deferral-store.js";
import { extractSessionForFilter, type SessionFilter } from "../utils/session-filter.js";

interface Logger {
  info: (message: string) => void;
  warn: (message: string) => void;
}

/** Default wait before a quota-deferred task runs again (15 min). */
export const DEFAULT_QUOTA_DEFER_DELAY_MS = 15 * 60_000;

/** `data` key carrying the deferral id on a re-run task. */
export const QUOTA_DEFERRAL_ID_KEY = "quotaDeferralId";

const TAG = "[quota-deferral]";

export interface QuotaDeferResult {
  /** The entry that will re-run the scope. */
  entry: QuotaDeferralEntry;
  /** True when an earlier deferral of the same scope already covers this task. */
  merged: boolean;
}

export class QuotaDeferralScheduler {
  private readonly store: QuotaDeferralStore;
  private readonly byKey = new Map<string, QuotaDeferralEntry>();
  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();
  private stopped = false;

  /**
   * @param sessionFilter - sessions excluded from the pipeline
   *   (memory.capture.excludeAgents): their stored deferrals are removed at
   *   start instead of re-armed, so work deferred before the filter existed
   *   never runs.
   */
  constructor(
    private readonly delayMs: number,
    private readonly enqueue: (task: TaskPayload) => Promise<void>,
    private readonly logger: Logger,
    store?: QuotaDeferralStore,
    private readonly sessionFilter?: Pick<SessionFilter, "shouldSkip">,
  ) {
    this.store = store ?? new InMemoryQuotaDeferralStore();
  }

  /** Re-arm every deferral left in the store (e.g. by a previous process). */
  async start(): Promise<void> {
    this.stopped = false;
    let entries: QuotaDeferralEntry[];
    try {
      entries = await this.store.list();
    } catch (err) {
      this.logger.warn(`${TAG} Failed to load deferred tasks: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    let restored = 0;
    for (const entry of entries) {
      if (this.sessionFilter?.shouldSkip(extractSessionForFilter(entry.task.sessionId))) {
        try {
          await this.store.remove(entry.id);
          this.logger.info(`${TAG} Dropped deferred ${entry.task.type} ${entry.id} [${entry.task.sessionId}]: session excluded by memory.capture.excludeAgents`);
        } catch (err) {
          this.logger.warn(`${TAG} Failed to remove excluded deferred task ${entry.id}: ${err instanceof Error ? err.message : String(err)}`);
        }
        continue;
      }
      const restoredEntry = { ...entry, firstDeferredAtMs: this.firstDeferredAt(entry) };
      this.byKey.set(restoredEntry.key, restoredEntry);
      this.arm(restoredEntry);
      restored++;
    }
    if (restored > 0) this.logger.info(`${TAG} Restored ${restored} quota-deferred task(s)`);
  }

  stop(): void {
    this.stopped = true;
    for (const t of this.timers.values()) clearTimeout(t);
    this.timers.clear();
  }

  /** Deferred tasks not settled yet. */
  pendingCount(): number {
    return this.byKey.size;
  }

  /** Age of the oldest unsettled deferral, from its first deferral (0 when none). */
  maxDeferralAgeMs(now = Date.now()): number {
    let oldest = 0;
    for (const entry of this.byKey.values()) {
      oldest = Math.max(oldest, now - this.firstDeferredAt(entry));
    }
    return oldest;
  }

  /**
   * Defer `task` for its scope `key`. A re-run of an existing entry that hits
   * the limit again re-arms that entry; any other task of a scope that is
   * already deferred is merged into it (the caller ACKs and drops it).
   */
  async defer(task: TaskPayload, key: string): Promise<QuotaDeferResult> {
    const existing = this.byKey.get(key);
    const ownId = task.data?.[QUOTA_DEFERRAL_ID_KEY];
    if (existing && existing.id !== ownId) return { entry: existing, merged: true };

    const now = Date.now();
    const id = existing?.id ?? `qd-${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    const entry: QuotaDeferralEntry = {
      id,
      key,
      task: stripRuntimeFields(task, id),
      fireAtMs: now + this.delayMs,
      deferrals: (existing?.deferrals ?? 0) + 1,
      // A re-run's createdAt is its enqueue time; the scope's wait starts here.
      firstDeferredAtMs: existing ? this.firstDeferredAt(existing) : now,
    };
    await this.store.put(entry);
    this.byKey.set(key, entry);
    this.arm(entry);
    return { entry, merged: false };
  }

  /** The re-run of a deferral finished (any outcome but another deferral): forget it. */
  async settle(task: TaskPayload): Promise<void> {
    const id = task.data?.[QUOTA_DEFERRAL_ID_KEY];
    if (typeof id !== "string") return;
    for (const [key, entry] of this.byKey) {
      if (entry.id !== id) continue;
      this.byKey.delete(key);
      const t = this.timers.get(id);
      if (t) clearTimeout(t);
      this.timers.delete(id);
    }
    try {
      await this.store.remove(id);
    } catch (err) {
      this.logger.warn(`${TAG} Failed to remove deferred task ${id}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /** Stored first deferral time, or `fireAtMs - delayMs` for an entry from an older file. */
  private firstDeferredAt(entry: QuotaDeferralEntry): number {
    return typeof entry.firstDeferredAtMs === "number" && Number.isFinite(entry.firstDeferredAtMs)
      ? entry.firstDeferredAtMs
      : entry.fireAtMs - this.delayMs;
  }

  private arm(entry: QuotaDeferralEntry): void {
    if (this.stopped) return;
    const prev = this.timers.get(entry.id);
    if (prev) clearTimeout(prev);
    const t = setTimeout(() => {
      this.timers.delete(entry.id);
      if (this.stopped) return;
      const run: TaskPayload = { ...entry.task, id: `${entry.task.id}-qd${entry.deferrals}`, createdAt: Date.now() };
      this.enqueue(run).catch((err) => {
        // The entry stays in the store; the next start re-arms it.
        this.logger.warn(`${TAG} Failed to enqueue deferred task ${entry.id}: ${err instanceof Error ? err.message : String(err)}`);
      });
    }, Math.max(0, entry.fireAtMs - Date.now()));
    t.unref?.();
    this.timers.set(entry.id, t);
  }
}

/** Copy of a task without the worker/executor `_` fields, tagged with its deferral id. */
function stripRuntimeFields(task: TaskPayload, id: string): TaskPayload {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(task)) {
    if (!k.startsWith("_")) out[k] = v;
  }
  const base = out as unknown as TaskPayload;
  // Keep the id of the first deferred task so re-run ids do not grow.
  const firstId = typeof task.data?.[QUOTA_DEFERRAL_ID_KEY] === "string" ? task.id.replace(/-qd\d+$/, "") : task.id;
  return { ...base, id: firstId, data: { ...(task.data ?? {}), [QUOTA_DEFERRAL_ID_KEY]: id } };
}
