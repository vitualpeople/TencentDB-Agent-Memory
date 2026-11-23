/**
 * memory.capture.excludeAgents keeps a session out of the L1/L2/L3 pipeline
 * while its L0 rows are still stored and readable.
 *
 *   T2a SessionFilter patterns: a leading `^` (and a trailing `$`) anchor the
 *       match; a pattern with no `^` (OpenClaw's `bench-judge-*`) still
 *       matches anywhere, as before.
 *   T2b Real gateway: with excludeAgents ["^plan-hash:*", "^deploy:*"],
 *       POST /v2/conversation/add for plan-hash:abc stores L0 (query returns
 *       it) but enqueues no L1/L2/L3 task and arms no pipeline timer; a
 *       proj:* session in the same gateway still gets its L1.
 *   T2c Rollout recovery on the local backend: a quota deferral in
 *       pipeline-quota-deferrals.json and a task already queued for
 *       plan-hash:abc are dropped (deferral removed, task ACKed, 0 LLM
 *       calls); the same for proj:distributed-agent-mesh still runs.
 *   T2d L2 keys "profile:<scope>|session:<encoded session>" are filtered on
 *       the decoded source session.
 *
 * No network: LLM work is replaced by counting executors / stubbed core
 * methods.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as sessionFilterModule from "../src/utils/session-filter.js";
import { SessionFilter } from "../src/utils/session-filter.js";
import { TdaiGateway } from "../src/gateway/server.js";
import { LocalStateBackend } from "../src/core/state/local-backend.js";
import type { TaskPayload } from "../src/core/state/types.js";
import { PipelineWorker, type TaskExecutor } from "../src/services/pipeline-worker.js";
import { FileQuotaDeferralStore } from "../src/services/quota-deferral-store.js";
import { QUOTA_DEFERRAL_ID_KEY } from "../src/services/quota-deferral-scheduler.js";
import { silent } from "./helpers/l1-spike-harness.js";

const EXCLUDE = ["^plan-hash:*", "^deploy:*"];
const FILTERED = "plan-hash:abc";
const KEPT = "proj:distributed-agent-mesh";
const INSTANCE = "default";

function tmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

async function waitFor(cond: () => boolean, maxMs = 5_000): Promise<boolean> {
  const until = Date.now() + maxMs;
  while (Date.now() < until) {
    if (cond()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return cond();
}

/** Executor that only records which session each L1/L2/L3 run was for. */
function countingExecutor() {
  const runs: Array<{ type: string; sessionId: string }> = [];
  const record = async (t: TaskPayload) => { runs.push({ type: t.type, sessionId: t.sessionId }); };
  const executor: TaskExecutor = { executeL1: record, executeL2: record, executeL3: record };
  return { executor, runs };
}

function task(type: "L1" | "L2" | "L3", sessionId: string, id: string): TaskPayload {
  return {
    id, type, instanceId: INSTANCE, sessionId, agentId: "default", priority: type === "L1" ? 0 : 1,
    createdAt: Date.now(), data: { instanceId: INSTANCE, agentId: "default" },
  };
}

function worker(backend: LocalStateBackend, executor: TaskExecutor, extra: Record<string, unknown> = {}) {
  return new PipelineWorker(backend, executor, {
    concurrency: 1, pollIntervalMs: 20, lockRenewIntervalMs: 60_000, sessionFilter: new SessionFilter(EXCLUDE), ...extra,
  } as never, silent);
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("T2a SessionFilter anchors", () => {
  it("T2a ^ anchors the start (and $ the end); patterns without ^ still match anywhere", () => {
    const f = new SessionFilter(EXCLUDE);
    expect(f.shouldSkip("plan-hash:abc")).toBe(true);
    expect(f.shouldSkip("deploy:byd-v3")).toBe(true);
    for (const kept of ["proj:plan-hash:foo", "proj:distributed-agent-mesh", "telegram:1794957247", "vnstock:watchlist"]) {
      expect(f.shouldSkip(kept)).toBe(false);
    }

    const exact = new SessionFilter(["^deploy:byd$"]);
    expect(exact.shouldSkip("deploy:byd")).toBe(true);
    expect(exact.shouldSkip("deploy:byd-v3")).toBe(false);

    // OpenClaw-style pattern with no anchor keeps matching inside the key.
    const legacy = new SessionFilter(["bench-judge-*"]);
    expect(legacy.shouldSkip("agent:main:bench-judge-42")).toBe(true);
    expect(legacy.shouldSkip("agent:main:chat")).toBe(false);
    // Built-in rules stay on.
    expect(f.shouldSkip("temp:slug-generator")).toBe(true);
  });
});

describe("T2b gateway: filtered sessions keep L0 but never enter the pipeline", () => {
  it("T2b /v2/conversation/add for plan-hash:abc stores L0 and enqueues no L1/L2/L3; proj:* still gets L1", async () => {
    const dir = tmp("session-filter-gw-");
    const file = path.join(dir, "tdai-gateway.yaml");
    fs.writeFileSync(file, [
      "server:", "  port: 0", "  host: 127.0.0.1",
      "data:", `  baseDir: ${JSON.stringify(path.join(dir, "data"))}`,
      "memory:", "  capture:", "    excludeAgents:", ...EXCLUDE.map((p) => `      - ${JSON.stringify(p)}`),
      "",
    ].join("\n"), "utf-8");
    vi.stubEnv("TDAI_GATEWAY_CONFIG", file);
    vi.stubEnv("TDAI_DATA_DIR", undefined);
    vi.stubEnv("TDAI_LLM_BASE_URL", "https://llm.invalid/v1");
    vi.stubEnv("TDAI_LLM_MODEL", "m");
    vi.stubEnv("TDAI_LLM_API_KEY", "placeholder-not-a-secret");

    const gw = new TdaiGateway() as unknown as {
      start(): Promise<void>; stop(): Promise<void>;
      server: { address(): AddressInfo };
      core: Record<string, unknown>;
      stateBackend: LocalStateBackend;
    };
    await gw.start();
    try {
      const runL1 = vi.fn(async () => ({ storedCount: 0, creditUsed: 0, hasMore: false, hasFullBacklog: false, profileScopes: [] }));
      gw.core.runL1WithStore = runL1;
      const enqueue = vi.spyOn(gw.stateBackend, "enqueueTask");
      const setTimer = vi.spyOn(gw.stateBackend, "setTimer");
      const url = `http://127.0.0.1:${gw.server.address().port}`;
      const post = async (route: string, body: unknown) => {
        const res = await fetch(`${url}${route}`, {
          method: "POST",
          headers: { "content-type": "application/json", "x-tdai-service-id": INSTANCE, authorization: "Bearer t" },
          body: JSON.stringify(body),
        });
        return { status: res.status, body: await res.json() as { code: number; data?: { messages?: unknown[] } } };
      };

      for (const session of [FILTERED, KEPT]) {
        const added = await post("/v2/conversation/add", {
          session_id: session,
          messages: [{ role: "user", content: `question for ${session}` }, { role: "assistant", content: "answer" }],
        });
        expect(added.status).toBe(200);
        expect(added.body.code).toBe(0);
      }

      const queried = await post("/v2/conversation/query", { session_id: FILTERED });
      expect(queried.body.code).toBe(0);
      expect(queried.body.data?.messages).toHaveLength(2);

      const pipelineTasks = (session: string) => enqueue.mock.calls
        .map(([t]) => t)
        .filter((t) => ["L1", "L2", "L3", "flush"].includes(t.type) && (t.sessionId === session || t.sessionId.includes(encodeURIComponent(session))));
      expect(pipelineTasks(FILTERED)).toEqual([]);
      expect(setTimer.mock.calls.filter(([, member]) => member.includes(encodeURIComponent(FILTERED)) || member.includes(FILTERED))).toEqual([]);
      // The kept session went through the same gateway and got its L1.
      expect(pipelineTasks(KEPT).map((t) => t.type)).toContain("L1");

      await waitFor(() => runL1.mock.calls.length >= 1);
      expect(runL1.mock.calls.map((c) => (c as unknown[])[0])).not.toContain(FILTERED);
    } finally {
      await gw.stop();
    }
  }, 60_000);
});

describe("T2c rollout recovery: work left from before the filter existed", () => {
  it("T2c a deferral in pipeline-quota-deferrals.json and a queued task for plan-hash:abc are dropped with 0 LLM calls; proj:* runs", async () => {
    const file = path.join(tmp("session-filter-deferrals-"), "metadata", "pipeline-quota-deferrals.json");
    const store = new FileQuotaDeferralStore(file);
    for (const session of [FILTERED, KEPT]) {
      const id = `qd-${session.replace(/[^a-z]/g, "")}`;
      await store.put({
        id,
        key: `L1\u0000${INSTANCE}\u0000\u0000default\u0000${session}`,
        task: { ...task("L1", session, `L1-deferred-${session}`), data: { instanceId: INSTANCE, agentId: "default", [QUOTA_DEFERRAL_ID_KEY]: id } },
        fireAtMs: Date.now() + 200,
        deferrals: 1,
      });
    }
    const backend = new LocalStateBackend();
    await backend.enqueueTask(task("L1", FILTERED, "L1-queued-filtered"));
    await backend.enqueueTask(task("L1", KEPT, "L1-queued-kept"));
    const { executor, runs } = countingExecutor();
    const w = worker(backend, executor, { quotaDeferralStore: new FileQuotaDeferralStore(file) });

    await w.start();
    try {
      // The deferral store forgets plan-hash:abc as soon as the scheduler starts.
      const ids = (await new FileQuotaDeferralStore(file).list()).map((e) => e.task.sessionId);
      expect(ids).not.toContain(FILTERED);

      expect(await waitFor(() => runs.filter((r) => r.sessionId === KEPT).length === 2)).toBe(true);
      await new Promise((r) => setTimeout(r, 300));

      const llmCalls = runs.filter((r) => r.sessionId === FILTERED).length;
      expect(llmCalls).toBe(0);
      expect(await backend.listQueuedTasks()).toEqual([]);
      expect(await new FileQuotaDeferralStore(file).list()).toEqual([]);
      expect(w.getMetrics()).toMatchObject({ tasksFailed: 0, tasksDeadLettered: 0, quotaDeferredPending: 0 });
    } finally {
      await w.stop();
    }
  });
});

describe("T2d L2 profile keys are filtered on their decoded source session", () => {
  it("T2d profile:...|session:plan-hash%3Aabc is dropped (0 LLM calls); profile:...|session:proj%3Adistributed-agent-mesh runs", async () => {
    const filteredKeys = [
      "profile:team%3Adefault%7Cagent%3Adefault|session:plan-hash%3Aabc",
      "profile:team:default|agent:default|session:plan-hash%3Aabc",
    ];
    const keptKey = "profile:team%3Adefault%7Cagent%3Adefault|session:proj%3Adistributed-agent-mesh";
    const backend = new LocalStateBackend();
    for (const [i, key] of filteredKeys.entries()) await backend.enqueueTask(task("L2", key, `L2-filtered-${i}`));
    await backend.enqueueTask(task("L2", keptKey, "L2-kept"));
    const { executor, runs } = countingExecutor();
    const w = worker(backend, executor);

    await w.start();
    try {
      expect(await waitFor(() => runs.some((r) => r.sessionId === keptKey))).toBe(true);
      await new Promise((r) => setTimeout(r, 300));

      const llmCalls = runs.filter((r) => filteredKeys.includes(r.sessionId)).length;
      expect(llmCalls).toBe(0);
      expect(runs.filter((r) => r.type === "L2").map((r) => r.sessionId)).toEqual([keptKey]);
      expect(await backend.listQueuedTasks()).toEqual([]);
      expect(w.getMetrics()).toMatchObject({ tasksFailed: 0, tasksDeadLettered: 0 });
    } finally {
      await w.stop();
    }

    // The helper the worker filters with: source session of a profile key, decoded.
    const extract = (sessionFilterModule as unknown as { extractSessionForFilter?: (key: string) => string }).extractSessionForFilter;
    expect(typeof extract).toBe("function");
    expect(extract?.("profile:team%3Adefault%7Cagent%3Adefault|session:plan-hash%3Aabc")).toBe(FILTERED);
    expect(extract?.("profile:team:default|agent:default|session:plan-hash%3Aabc")).toBe(FILTERED);
    expect(extract?.("profile:team:default|agent:default")).toBe("profile:team:default|agent:default");
    expect(extract?.(KEPT)).toBe(KEPT);
  });
});
