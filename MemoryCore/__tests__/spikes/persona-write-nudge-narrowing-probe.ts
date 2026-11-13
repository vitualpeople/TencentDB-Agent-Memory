/**
 * Spike probe (not a test, not wired into any runner): how far can the L3
 * write nudge be narrowed? Compares, on one throwaway data dir, three ways to
 * send the nudge prompt with the same persona-only tool set
 * (createPersonaWriteTools, the allowlist wrapper production uses):
 *
 *   default      generateText + tools, stopWhen stepCountIs(20) (production)
 *   steps-1/2    the same with stopWhen stepCountIs(1) / stepCountIs(2)
 *                (what LLMRunParams.maxIterations would give)
 *   forced-write 1 step with toolChoice { type: "tool", toolName: "write" },
 *                called on generateText directly. LLMRunParams has no
 *                toolChoice field (core/types.ts), so this is deliberately not
 *                wired into StandaloneLLMRunner here.
 *
 * Per variant it prints: steps used, tool calls by name, whether persona.md
 * was written (writesObserved()), and the latency. Never prints content.
 *
 * Model: always "orcarouter/auto" (never a named model). It runs only when
 * both PERSONA_PROBE_BASE_URL and PERSONA_PROBE_API_KEY are set in the
 * environment; it never reads a key from a file and spends nothing otherwise.
 *
 *   PERSONA_PROBE_BASE_URL=... PERSONA_PROBE_API_KEY=... \
 *     pnpm exec tsx __tests__/spikes/persona-write-nudge-narrowing-probe.ts
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, stepCountIs, type ToolSet } from "ai";
import { createPersonaWriteTools } from "../../src/adapters/standalone/persona-tools.js";
import { PERSONA_WRITE_NUDGE } from "../../src/core/persona/persona-generator.js";

const MODEL = "orcarouter/auto";
const PROMPT =
  "Update the user persona from this scene: the user prefers short answers and writes tests first. " +
  "The current persona.md is: '# Persona\\n\\nThe user prefers concise answers.'" +
  PERSONA_WRITE_NUDGE;

interface Variant {
  name: string;
  steps: number;
  forceWrite: boolean;
}

const VARIANTS: Variant[] = [
  { name: "default", steps: 20, forceWrite: false },
  { name: "steps-1", steps: 1, forceWrite: false },
  { name: "steps-2", steps: 2, forceWrite: false },
  { name: "forced-write", steps: 1, forceWrite: true },
];

async function runVariant(v: Variant, baseURL: string, apiKey: string): Promise<Record<string, unknown>> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "persona-probe-"));
  fs.writeFileSync(path.join(dir, "persona.md"), "# Persona\n\nThe user prefers concise answers.", "utf-8");
  const toolset = createPersonaWriteTools({ workspaceDir: dir });
  const provider = createOpenAI({ baseURL, apiKey });
  const started = Date.now();
  try {
    const result = await generateText({
      model: provider.chat(MODEL),
      prompt: PROMPT,
      tools: toolset.tools as ToolSet,
      stopWhen: stepCountIs(v.steps),
      ...(v.forceWrite ? { toolChoice: { type: "tool" as const, toolName: "write" } } : {}),
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(60_000),
    });
    const toolCalls = result.steps.flatMap((s) => s.toolCalls.map((c) => c.toolName));
    return {
      variant: v.name,
      stepsUsed: result.steps.length,
      toolCalls,
      wrotePersona: toolset.writesObserved() > 0,
      violated: toolset.violated(),
      latencyMs: Date.now() - started,
    };
  } catch (err) {
    return { variant: v.name, error: err instanceof Error ? err.name : "error", latencyMs: Date.now() - started };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  const baseURL = process.env.PERSONA_PROBE_BASE_URL?.trim();
  const apiKey = process.env.PERSONA_PROBE_API_KEY?.trim();
  if (!baseURL || !apiKey) {
    console.log("persona nudge probe: PERSONA_PROBE_BASE_URL / PERSONA_PROBE_API_KEY not set; nothing called (result TBD)");
    return;
  }
  for (const v of VARIANTS) console.log(JSON.stringify(await runVariant(v, baseURL, apiKey)));
}

void main();
