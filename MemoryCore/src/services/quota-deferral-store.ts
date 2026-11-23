/**
 * Persistence of pipeline tasks deferred after an LLM quota / rate-limit
 * failure (see PipelineWorker.deferForQuota).
 *
 * The local state backend keeps its queue and timers in RAM, so a deferred
 * task held only there would be lost on a container restart. The worker
 * writes each deferral here before it ACKs the failed message, removes it
 * once the re-run settles, and re-arms every entry it finds at start.
 *
 * One entry per (task type, session scope): a second quota failure of the
 * same scope while one is deferred adds nothing, the deferred run reads the
 * session's cursor when it fires and covers the newer rows too.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { TaskPayload } from "../core/state/types.js";

export interface QuotaDeferralEntry {
  /** Stable id; the re-run task carries it in `data.quotaDeferralId`. */
  id: string;
  /** (task type, session scope) key: at most one entry per key. */
  key: string;
  /** Task to enqueue again (runtime `_` fields stripped). */
  task: TaskPayload;
  /** When the task may run again (epoch ms). */
  fireAtMs: number;
  /** Consecutive quota deferrals of this scope, this one included. */
  deferrals: number;
}

export interface QuotaDeferralStore {
  list(): Promise<QuotaDeferralEntry[]>;
  /** Insert or replace by `id`. */
  put(entry: QuotaDeferralEntry): Promise<void>;
  remove(id: string): Promise<void>;
}

/** RAM-only store: the default when no file is configured (lost on restart). */
export class InMemoryQuotaDeferralStore implements QuotaDeferralStore {
  private entries = new Map<string, QuotaDeferralEntry>();

  async list(): Promise<QuotaDeferralEntry[]> {
    return [...this.entries.values()];
  }

  async put(entry: QuotaDeferralEntry): Promise<void> {
    this.entries.set(entry.id, entry);
  }

  async remove(id: string): Promise<void> {
    this.entries.delete(id);
  }
}

/**
 * JSON file store, rewritten whole through a temp file + rename so a crash
 * mid-write leaves the previous version. Writes are serialised in-process.
 */
export class FileQuotaDeferralStore implements QuotaDeferralStore {
  private chain: Promise<unknown> = Promise.resolve();

  constructor(private readonly filePath: string) {}

  async list(): Promise<QuotaDeferralEntry[]> {
    await this.chain.catch(() => {});
    return this.read();
  }

  put(entry: QuotaDeferralEntry): Promise<void> {
    return this.mutate((entries) => {
      const kept = entries.filter((e) => e.id !== entry.id);
      kept.push(entry);
      return kept;
    });
  }

  remove(id: string): Promise<void> {
    return this.mutate((entries) => entries.filter((e) => e.id !== id));
  }

  private mutate(change: (entries: QuotaDeferralEntry[]) => QuotaDeferralEntry[]): Promise<void> {
    const next = this.chain.catch(() => {}).then(async () => {
      const entries = change(await this.read());
      await fs.mkdir(path.dirname(this.filePath), { recursive: true });
      const tmp = `${this.filePath}.${process.pid}.tmp`;
      await fs.writeFile(tmp, JSON.stringify({ version: 1, entries }, null, 2), "utf-8");
      await fs.rename(tmp, this.filePath);
    });
    this.chain = next;
    return next;
  }

  private async read(): Promise<QuotaDeferralEntry[]> {
    let raw: string;
    try {
      raw = await fs.readFile(this.filePath, "utf-8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw err;
    }
    const parsed = JSON.parse(raw) as { entries?: unknown };
    return Array.isArray(parsed.entries) ? (parsed.entries as QuotaDeferralEntry[]).filter(isEntry) : [];
  }
}

function isEntry(e: unknown): e is QuotaDeferralEntry {
  if (!e || typeof e !== "object") return false;
  const x = e as Record<string, unknown>;
  return typeof x.id === "string" && typeof x.key === "string" && typeof x.fireAtMs === "number" &&
    !!x.task && typeof x.task === "object";
}
