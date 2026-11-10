import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { VectorStore } from "../src/core/store/sqlite.js";
import type { L0Record } from "../src/core/store/types.js";
import { handleConversationAdd, type V2RouterDeps } from "../src/gateway/v2-router.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };
const auth = { serviceId: "svc" } as Parameters<typeof handleConversationAdd>[1];
const twoMessages = {
  session_id: "proj:demo",
  messages: [{ role: "user", content: "first" }, { role: "assistant", content: "second" }],
};

function deps(store: unknown, notifyPipeline = vi.fn(async () => {})): V2RouterDeps {
  return {
    getStore: () => store,
    getEmbedding: () => undefined,
    getStorage: () => undefined,
    logger: silent,
    deployMode: "service",
    notifyPipeline,
  } as unknown as V2RouterDeps;
}

type AddData = { accepted_ids: string[]; total_count: number };

describe("conversation/add with a store that has no atomic write (per-row fallback)", () => {
  it("an id whose upsertL0 returned false is not accepted nor counted", async () => {
    const upsertL0 = vi.fn().mockReturnValueOnce(true).mockReturnValueOnce(false);

    const res = await handleConversationAdd(twoMessages, auth, "req-1", deps({ upsertL0 }));

    expect(res.code).toBe(0);
    const data = res.data as AddData;
    expect(data.total_count).toBe(1);
    expect(data.accepted_ids).toEqual([upsertL0.mock.calls[0][0].id]);
  });

  it("returns an error envelope when no id was stored", async () => {
    const upsertL0 = vi.fn(() => false);
    const notify = vi.fn(async () => {});

    const res = await handleConversationAdd(twoMessages, auth, "req-2", deps({ upsertL0 }, notify));

    expect(res.code).toBe(500);
    expect(res.data).toBeUndefined();
    expect(notify).not.toHaveBeenCalled();
  });
});

describe("conversation/add with a store that has upsertL0Atomic", () => {
  it("writes every record in one call and never falls back to upsertL0", async () => {
    const upsertL0 = vi.fn(() => true);
    const upsertL0Atomic = vi.fn();

    const res = await handleConversationAdd(twoMessages, auth, "req-3", deps({ upsertL0, upsertL0Atomic }));

    expect(res.code).toBe(0);
    expect(upsertL0Atomic).toHaveBeenCalledTimes(1);
    const [records, embeddings] = upsertL0Atomic.mock.calls[0] as [L0Record[], unknown[]];
    expect(records.map((r) => r.messageText)).toEqual(["first", "second"]);
    expect(embeddings).toHaveLength(2);
    expect(upsertL0).not.toHaveBeenCalled();
    expect((res.data as AddData).accepted_ids).toEqual(records.map((r) => r.id));
  });

  it("a failed atomic write returns an error envelope and does not notify the pipeline", async () => {
    const upsertL0Atomic = vi.fn(() => { throw new Error("disk full"); });
    const notify = vi.fn(async () => {});

    const res = await handleConversationAdd(twoMessages, auth, "req-4", deps({ upsertL0: vi.fn(), upsertL0Atomic }, notify));

    expect(res.code).toBe(500);
    expect(res.message).not.toContain("disk full");
    expect(notify).not.toHaveBeenCalled();
  });
});

/**
 * A real SQLite VectorStore with dimensions=0 (metadata + FTS, no sqlite-vec
 * needed). The prepared metadata statement is wrapped so its Nth call throws,
 * which is a failure in the middle of the write of one request.
 */
describe("conversation/add on SQLite is all-or-nothing", () => {
  let dir: string | undefined;
  let store: VectorStore | undefined;

  afterEach(() => {
    store?.close();
    if (dir) rmSync(dir, { recursive: true, force: true });
    store = undefined;
    dir = undefined;
  });

  function openStore(failOnCall: number): VectorStore {
    dir = mkdtempSync(path.join(os.tmpdir(), "tdai-l0-atomic-"));
    store = new VectorStore(path.join(dir, "vectors.db"), 0);
    store.init();
    const internals = store as unknown as { stmtL0UpsertMeta: { run: (...a: unknown[]) => unknown } };
    const real = internals.stmtL0UpsertMeta;
    let calls = 0;
    internals.stmtL0UpsertMeta = {
      run: (...a: unknown[]) => {
        calls++;
        if (calls === failOnCall) throw new Error("SQLITE_FULL: simulated");
        return real.run(...a);
      },
    };
    return store;
  }

  it("the second message failing leaves no record of the request and returns an error envelope", async () => {
    const s = openStore(2);

    const res = await handleConversationAdd(twoMessages, auth, "req-5", deps(s));

    expect(res.code).not.toBe(0);
    expect(s.countL0()).toBe(0);
  });

  it("upsertL0Atomic throws and rolls back every record when one fails mid-transaction", () => {
    const s = openStore(3);
    const records: L0Record[] = ["a", "b", "c"].map((t, i) => ({
      id: `msg-${i}`,
      sessionKey: "proj:demo",
      sessionId: "proj:demo",
      role: "user",
      messageText: t,
      recordedAt: new Date(1_760_000_000_000 + i).toISOString(),
      timestamp: 1_760_000_000_000 + i,
    }));

    expect(typeof s.upsertL0Atomic).toBe("function");
    expect(() => s.upsertL0Atomic(records, [undefined, undefined, undefined])).toThrow(/SQLITE_FULL/);
    expect(s.countL0()).toBe(0);
  });

  it("upsertL0Atomic commits every record when nothing fails", () => {
    const s = openStore(0);
    const records: L0Record[] = ["a", "b"].map((t, i) => ({
      id: `msg-ok-${i}`,
      sessionKey: "proj:demo",
      sessionId: "proj:demo",
      role: "user",
      messageText: t,
      recordedAt: new Date(1_760_000_000_000 + i).toISOString(),
      timestamp: 1_760_000_000_000 + i,
    }));

    expect(typeof s.upsertL0Atomic).toBe("function");
    s.upsertL0Atomic(records, [undefined, undefined]);
    expect(s.countL0()).toBe(2);
  });
});
