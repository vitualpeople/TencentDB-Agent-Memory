/**
 * A successful L1 run moves the session cursor to the slice max, and so does
 * a valid empty array `[]` ("the LLM found no memory" is not a failure).
 *
 * Deliberately imports nothing added by the failure-path fix, so the same
 * file runs unchanged on the code before it and must pass on both.
 */
import { describe, expect, it } from "vitest";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { createL1Runner } from "../src/utils/pipeline-factory.js";
import { makeCfg, makeL0Store, msg, sceneJson, scriptedLlm, silent, tmpDir } from "./helpers/l1-spike-harness.js";

const SESSION = "proj:demo";

describe("L1 success paths advance the cursor", () => {
  it.each([
    ["a valid empty array", "[]", 0],
    ["a valid empty array in a markdown fence", "```json\n[]\n```", 0],
    ["a valid scene", sceneJson("charlie", ["L0-c1", "L0-c2"]), 1],
  ])("%s -> success, cursor moves to the slice max", async (_label, reply, stored) => {
    const dir = tmpDir("l1-success-");
    await new CheckpointManager(dir, silent).markL1ExtractionComplete(SESSION, 0, 500);
    const { store, l1Records } = makeL0Store([
      { id: "L0-c1", sessionId: "sess-c", content: msg("charlie one"), timestamp: 10, recordedAtMs: 601 },
      { id: "L0-c2", sessionId: "sess-c", content: msg("charlie two"), timestamp: 20, recordedAtMs: 602 },
    ]);
    const { runner, run: llmCalls } = scriptedLlm(() => reply);
    const run = createL1Runner({
      pluginDataDir: dir, cfg: makeCfg(), openclawConfig: undefined, vectorStore: store as never, embeddingService: undefined,
      logger: silent, llmRunner: runner,
    });

    const out = await run({ sessionKey: SESSION });

    expect(llmCalls).toHaveBeenCalledTimes(1);
    expect(out.processedCount).toBe(2);
    expect(l1Records).toHaveLength(stored);
    expect((await new CheckpointManager(dir, silent).read()).runner_states[SESSION].last_l1_cursor).toBe(602);
  });
});
