/**
 * L2 scene extraction must count the memories it processed into the persona
 * counters of the profile scope it wrote, and only there.
 *
 * L1 already adds every extracted memory to the root checkpoint
 * (markL1ExtractionComplete). L2 writes scenes under profiles/<scope>/, and
 * the L3 runner reads the persona trigger from that same scoped checkpoint, so
 * a scope whose counter never moves never gets a persona refresh.
 *
 * Every L1 record resolves to a `team:<t>|agent:<a>` scope (missing ids fall
 * back to `default`), so both the default bucket and an explicit team/agent
 * land under profiles/. Both branches of the runner are covered: no changed
 * scene profile, and changed profiles with generation provenance.
 */
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const extractorBehaviour = vi.hoisted(() => ({
  memoriesProcessed: 0,
  writeSceneFile: false,
}));

vi.mock("../src/core/scene/scene-extractor.js", () => ({
  // Stands in for the LLM-backed extractor: no model call, optionally writes
  // one scene block through the scoped storage it was handed.
  SceneExtractor: class {
    private readonly storage: { writeFile(key: string, content: string): Promise<void> } | undefined;
    constructor(opts: { storage?: { writeFile(key: string, content: string): Promise<void> } }) {
      this.storage = opts.storage;
    }
    async extract(memories: unknown[]) {
      if (extractorBehaviour.writeSceneFile && this.storage) {
        await this.storage.writeFile("scene_blocks/fake-scene.md", `# fake scene\n\n${memories.length} memories\n`);
      }
      return { success: true, memoriesProcessed: extractorBehaviour.memoriesProcessed, emptyExtraction: false };
    }
  },
}));

import { createL2Runner } from "../src/utils/pipeline-factory.js";
import { CheckpointManager } from "../src/utils/checkpoint.js";
import { StorageAdapter } from "../src/core/storage/adapter.js";
import { LocalStorageBackend } from "../src/core/storage/local-backend.js";
import type { IMemoryStore, L1RecordRow } from "../src/core/store/types.js";
import type { MemoryTdaiConfig } from "../src/config.js";

const silent = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

function l1Row(id: string, updated: string, scope: { team?: string; agent?: string }): L1RecordRow {
  return {
    record_id: id,
    content: `memory ${id}`,
    type: "episodic",
    priority: 50,
    scene_name: "",
    session_key: "proj:l2-scope-test",
    session_id: "proj:l2-scope-test",
    team_id: scope.team ?? "",
    task_id: "",
    user_id: "",
    agent_id: scope.agent ?? "",
    version: 1,
    timestamp_str: updated,
    timestamp_start: updated,
    timestamp_end: updated,
    created_time: updated,
    updated_time: updated,
    metadata_json: "{}",
  };
}

function fakeStore(rows: L1RecordRow[]): IMemoryStore {
  return {
    isDegraded: () => false,
    queryL1Records: async () => rows,
  } as unknown as IMemoryStore;
}

const cfg = {
  persona: { model: "fake-model", promptMode: undefined, maxScenes: 10, sceneBackupCount: 0 },
} as unknown as MemoryTdaiConfig;

const scopes = [
  { label: "default team/agent bucket", ids: {}, dir: "team%3Adefault%7Cagent%3Adefault" },
  { label: "explicit team/agent", ids: { team: "team-a", agent: "agent-x" }, dir: "team%3Ateam-a%7Cagent%3Aagent-x" },
];

const branches = [
  { label: "no changed scene profile", writeSceneFile: false },
  { label: "changed scene profile with provenance", writeSceneFile: true },
];

describe("L2 runner counts processed memories into its profile scope checkpoint", () => {
  let dataDir: string;
  let storage: StorageAdapter;

  beforeEach(async () => {
    dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "l2-scope-checkpoint-"));
    storage = new StorageAdapter(new LocalStorageBackend(dataDir));
  });

  afterEach(async () => {
    await fs.rm(dataDir, { recursive: true, force: true });
  });

  for (const scope of scopes) {
    for (const branch of branches) {
      it(`${scope.label}, ${branch.label}: scope counters rise by memoriesProcessed, root persona counters stay at the L1 total`, async () => {
        extractorBehaviour.memoriesProcessed = 3;
        extractorBehaviour.writeSceneFile = branch.writeSceneFile;

        // L1 has already counted 5 memories into the root checkpoint.
        const root = new CheckpointManager(dataDir, silent, storage);
        await root.markL1ExtractionComplete("proj:l2-scope-test", 5);

        const rows = [
          l1Row("r1", "2026-11-01T10:00:00.000Z", scope.ids),
          l1Row("r2", "2026-11-01T12:00:00.000Z", scope.ids),
          l1Row("r3", "2026-11-01T11:00:00.000Z", scope.ids),
        ];
        const runner = createL2Runner({
          pluginDataDir: dataDir,
          cfg,
          openclawConfig: {},
          vectorStore: fakeStore(rows),
          logger: silent,
          instanceId: "test",
          storage,
        });

        const result = await runner("proj:l2-scope-test");
        expect(result).toMatchObject({ latestCursor: "2026-11-01T12:00:00.000Z" });

        const scopedStorage = new StorageAdapter(new LocalStorageBackend(path.join(dataDir, "profiles", scope.dir)));
        const scoped = new CheckpointManager(path.join(dataDir, "profiles", scope.dir), silent, scopedStorage);
        const scopedState = await scoped.read();
        expect(scopedState.scenes_processed).toBe(1);
        expect(scopedState.memories_since_last_persona).toBe(3);
        expect(scopedState.total_memories_extracted).toBe(3);

        const rootState = await root.read();
        expect(rootState.memories_since_last_persona).toBe(5);
        expect(rootState.total_memories_extracted).toBe(5);

        // A persona run resets the trigger counter but keeps the lifetime total.
        await scoped.markPersonaGenerated(scopedState.total_processed);
        const afterPersona = await scoped.read();
        expect(afterPersona.memories_since_last_persona).toBe(0);
        expect(afterPersona.total_memories_extracted).toBe(3);
      });
    }
  }

  it("a second L2 run on the same scope accumulates on top of the first", async () => {
    extractorBehaviour.memoriesProcessed = 2;
    extractorBehaviour.writeSceneFile = false;
    const ids = { team: "team-a", agent: "agent-x" };
    const runner = createL2Runner({
      pluginDataDir: dataDir,
      cfg,
      openclawConfig: {},
      vectorStore: fakeStore([l1Row("r1", "2026-11-01T10:00:00.000Z", ids), l1Row("r2", "2026-11-01T11:00:00.000Z", ids)]),
      logger: silent,
      storage,
    });

    await runner("proj:l2-scope-test");
    await runner("proj:l2-scope-test", "2026-11-01T09:00:00.000Z");

    const dir = path.join(dataDir, "profiles", "team%3Ateam-a%7Cagent%3Aagent-x");
    const state = await new CheckpointManager(dir, silent, new StorageAdapter(new LocalStorageBackend(dir))).read();
    expect(state.scenes_processed).toBe(2);
    expect(state.memories_since_last_persona).toBe(4);
    expect(state.total_memories_extracted).toBe(4);
  });
});

describe("CheckpointManager.incrementMemoriesSincePersona", () => {
  let dataDir: string;

  beforeEach(async () => {
    dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "checkpoint-increment-"));
  });

  afterEach(async () => {
    await fs.rm(dataDir, { recursive: true, force: true });
  });

  it("adds to both the persona trigger counter and the lifetime total, and ignores non-positive counts", async () => {
    const cp = new CheckpointManager(dataDir, silent, new StorageAdapter(new LocalStorageBackend(dataDir)));
    await cp.incrementMemoriesSincePersona(4);
    await cp.incrementMemoriesSincePersona(0);
    await cp.incrementMemoriesSincePersona(-2);
    const state = await cp.read();
    expect(state.memories_since_last_persona).toBe(4);
    expect(state.total_memories_extracted).toBe(4);
  });
});
