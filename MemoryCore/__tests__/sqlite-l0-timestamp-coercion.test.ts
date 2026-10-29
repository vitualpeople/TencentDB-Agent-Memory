import { describe, expect, it } from "vitest";
import { VectorStore } from "../src/core/store/sqlite.js";

/**
 * Databases created before l0_conversations.timestamp was an INTEGER column
 * hold epoch-ms strings there. queryL0ForL1 must hand L1 a number, as
 * L0QueryRow declares: the L1 prompt builder calls new Date(m.timestamp)
 * .toISOString(), which throws on a string and empties the L1 batch.
 *
 * The prepared statement is stubbed so the test pins the row mapping without
 * depending on the sqlite-vec extension being loadable on the test host.
 */
function storeReturning(rows: Array<Record<string, unknown>>): VectorStore {
  const store = Object.create(VectorStore.prototype) as VectorStore;
  Object.assign(store, {
    degraded: false,
    logger: undefined,
    stmtL0QueryAll: { all: () => rows },
    stmtL0QueryAfter: { all: () => rows },
  });
  return store;
}

const baseRow = {
  record_id: "msg-1",
  session_key: "proj:demo",
  session_id: "proj:demo",
  team_id: "default",
  task_id: "",
  user_id: "default",
  agent_id: "default",
  role: "user",
  message_text: "hello",
  recorded_at: "2026-10-29T06:00:00.000Z",
};

describe("VectorStore.queryL0ForL1 timestamp coercion", () => {
  it.each([
    ["epoch-ms string", "1761717600000", 1761717600000],
    ["number", 1761717600000, 1761717600000],
    ["bigint", 1761717600000n, 1761717600000],
    ["null", null, 0],
    ["non-numeric string", "not-a-time", 0],
  ])("maps a %s timestamp to a number", (_label, raw, expected) => {
    const [row] = storeReturning([{ ...baseRow, timestamp: raw }]).queryL0ForL1("proj:demo");

    expect(typeof row.timestamp).toBe("number");
    expect(row.timestamp).toBe(expected);
  });

  it("a string timestamp survives the L1 prompt's Date formatting", () => {
    const [row] = storeReturning([{ ...baseRow, timestamp: "1761717600000" }]).queryL0ForL1("proj:demo", 1);

    expect(new Date(row.timestamp).toISOString()).toBe("2025-10-29T06:00:00.000Z");
  });
});
