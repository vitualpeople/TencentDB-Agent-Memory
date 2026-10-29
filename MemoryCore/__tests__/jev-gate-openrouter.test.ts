import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parseConfig } from "../src/config.js";
import { JevClient } from "../src/core/decision/providers/jev/client.js";
import { evaluateL3JevGate } from "../src/core/persona/jev-gate.js";
import { writeSceneIndex } from "../src/core/scene/scene-index.js";

describe("JEV gate + OpenRouter Decisions API integration", () => {
  const savedEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...savedEnv };
    vi.restoreAllMocks();
  });

  it("resolves OpenRouter defaults from TDAI_PERSONA_TRIGGER_MODE=jev and OPENROUTER_API_KEY", () => {
    process.env.TDAI_PERSONA_TRIGGER_MODE = "jev";
    process.env.OPENROUTER_API_KEY = "sk-or-test-env-key";
    delete process.env.TDAI_JEV_ENDPOINT;
    delete process.env.TDAI_JEV_MODEL;
    delete process.env.TDAI_JEV_API_KEY;

    const cfg = parseConfig({});
    expect(cfg.persona.triggerMode).toBe("jev");
    expect(cfg.persona.jev.endpoint).toBe("https://openrouter.ai/api/alpha/decisions");
    expect(cfg.persona.jev.model).toBe("~typesafe/jev-latest");
    expect(cfg.persona.jev.apiKey).toBe("sk-or-test-env-key");
    expect(cfg.persona.jev.timeoutMs).toBe(5000);
  });

  it("parses both OpenRouter choice field and fallback value field in JevClient", async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          model: "typesafe/jev-1.14-20261014",
          answers: {
            q_choice: {
              type: "choice",
              choice: "conflict",
              probabilities: { conflict: 0.9, none: 0.1 },
              confidence: 0.88,
            },
            q_value: {
              type: "choice",
              value: "boundary_change",
              probabilities: { boundary_change: 0.8, none: 0.2 },
              confidence: 0.75,
            },
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const client = new JevClient(
      {
        endpoint: "https://openrouter.ai/api/alpha/decisions",
        apiKey: "sk-or-test",
        model: "~typesafe/jev-latest",
        timeoutMs: 5000,
      },
      fetcher as unknown as typeof fetch,
    );

    const ans = await client.askChoices("state text", [
      { id: "q_choice", instructions: "inst 1", criteria: { conflict: "c", none: "n" } },
      { id: "q_value", instructions: "inst 2", criteria: { boundary_change: "b", none: "n" } },
    ]);

    expect(ans.get("q_choice")?.choice).toBe("conflict");
    expect(ans.get("q_value")?.choice).toBe("boundary_change");
  });

  it.each([
    ["conflict", "immediate", true],
    ["boundary_change", "immediate", true],
    ["new_rule", "immediate", true],
    ["reinforcement", "deferred", false],
    ["none", "deferred", false],
  ] as const)("evaluates 5-level label %s -> kind=%s shouldGenerate=%s", async (label, expectedKind, expectedShould) => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "jev-gate-test-"));
    try {
      fs.mkdirSync(path.join(tmpDir, "scene_blocks"), { recursive: true });
      fs.writeFileSync(path.join(tmpDir, "persona.md"), "# Persona\nAlways use TypeScript strict mode.\n", "utf-8");
      fs.writeFileSync(
        path.join(tmpDir, ".checkpoint.json"),
        JSON.stringify({
          last_l1_id: "",
          last_scene_time: "2026-10-29T10:00:00.000Z",
          last_persona_time: "2026-10-20T00:00:00.000Z",
          total_processed: 5,
          scenes_processed: 1,
          memories_since_persona: 5,
        }),
        "utf-8",
      );
      await writeSceneIndex(tmpDir, [
        {
          filename: "coding.md",
          summary: "Updated coding rule",
          heat: 2,
          updated: "2026-10-29T10:00:00.000Z",
        },
      ]);
      fs.writeFileSync(
        path.join(tmpDir, "scene_blocks", "coding.md"),
        [
          "---",
          "scene: coding",
          "created: 2026-10-29T09:00:00.000Z",
          "updated: 2026-10-29T10:00:00.000Z",
          "heat: 2",
          "---",
          "",
          "## Summary",
          "Updated coding rule.",
          "",
        ].join("\n"),
        "utf-8",
      );

      process.env.TDAI_PERSONA_TRIGGER_MODE = "jev";
      process.env.OPENROUTER_API_KEY = "sk-or-test";
      const cfg = parseConfig({});

      const fetcher = vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            model: "typesafe/jev-1.14-20261014",
            answers: {
              memory_change: {
                type: "choice",
                choice: label,
                probabilities: { [label]: 0.92 },
                confidence: 0.9,
              },
            },
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );

      const res = await evaluateL3JevGate({
        dataDir: tmpDir,
        cfg,
        logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
        triggerReason: "memories_since_persona=5 < 50",
        fetcher: fetcher as unknown as typeof fetch,
      });

      expect(res.kind).toBe(expectedKind);
      expect(res.shouldGenerate).toBe(expectedShould);
      expect(res.label).toBe(label);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("fails safe to kind=unavailable on HTTP 500 or oversized state (>32000 chars)", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "jev-gate-fail-"));
    try {
      fs.mkdirSync(path.join(tmpDir, "scene_blocks"), { recursive: true });
      fs.writeFileSync(path.join(tmpDir, "persona.md"), "# Persona\nBase rule.\n", "utf-8");
      fs.writeFileSync(
        path.join(tmpDir, ".checkpoint.json"),
        JSON.stringify({
          last_l1_id: "",
          last_scene_time: "2026-10-29T10:00:00.000Z",
          last_persona_time: "2026-10-20T00:00:00.000Z",
          total_processed: 5,
          scenes_processed: 1,
          memories_since_persona: 5,
        }),
        "utf-8",
      );
      await writeSceneIndex(tmpDir, [
        {
          filename: "coding.md",
          summary: "Updated coding rule",
          heat: 2,
          updated: "2026-10-29T10:00:00.000Z",
        },
      ]);
      fs.writeFileSync(
        path.join(tmpDir, "scene_blocks", "coding.md"),
        [
          "---",
          "scene: coding",
          "created: 2026-10-29T09:00:00.000Z",
          "updated: 2026-10-29T10:00:00.000Z",
          "heat: 2",
          "---",
          "",
          "## Summary",
          "Updated coding rule.",
          "",
        ].join("\n"),
        "utf-8",
      );

      process.env.TDAI_PERSONA_TRIGGER_MODE = "jev";
      process.env.OPENROUTER_API_KEY = "sk-or-test";
      const cfg = parseConfig({});

      const fetcher500 = vi.fn().mockResolvedValue(new Response("Internal Server Error", { status: 500 }));
      const res500 = await evaluateL3JevGate({
        dataDir: tmpDir,
        cfg,
        logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
        triggerReason: "below_threshold",
        fetcher: fetcher500 as unknown as typeof fetch,
      });
      expect(res500.kind).toBe("unavailable");
      expect(res500.shouldGenerate).toBe(false);

      fs.writeFileSync(path.join(tmpDir, "persona.md"), "# Persona\n" + "X".repeat(33000), "utf-8");
      const fetcherNever = vi.fn();
      const resHuge = await evaluateL3JevGate({
        dataDir: tmpDir,
        cfg,
        logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
        triggerReason: "below_threshold",
        fetcher: fetcherNever as unknown as typeof fetch,
      });
      expect(resHuge.kind).toBe("unavailable");
      expect(resHuge.shouldGenerate).toBe(false);
      expect(fetcherNever).not.toHaveBeenCalled();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
