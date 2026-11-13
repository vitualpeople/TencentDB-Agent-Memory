/**
 * PersonaGenerator: generates or updates user persona using the four-layer
 * deep scan model via CleanContextRunner.
 *
 * One generation ("lần sinh") is bounded by a total deadline (`deadlineAt`):
 * the first runner call and the optional write nudge share it. A caller
 * abort signal reaches every runner call and blocks the final persona write.
 */

import type { MemoryPromptMode } from "../../config.js";
import { CleanContextRunner } from "../../utils/clean-context-runner.js";
import { CheckpointManager } from "../../utils/checkpoint.js";
import { readSceneIndex } from "../scene/scene-index.js";
import { generateSceneNavigation, stripSceneNavigation } from "../scene/scene-navigation.js";
import { buildPersonaPrompt, personaMaxCharsFor } from "../prompts/persona-generation.js";
import { BackupManager } from "../../utils/backup.js";
import { escapeXmlTags } from "../../utils/sanitize.js";
import { report } from "../report/reporter.js";
import { reportL3LatencyMetrics } from "../report/metric-tracking-l3-latency.js";
import type { LLMRunner, Logger, TraceContext } from "../types.js";
import { buildTraceParams } from "../types.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";
import type { ResolvedMemoryPrompt } from "../memory-prompt/types.js";
import { composeMemorySystemPrompt } from "../memory-prompt/composer.js";

const TAG = "[memory-tdai] [persona]";

/** Deadline of one generation when the caller does not set `deadlineAt`. */
export const DEFAULT_PERSONA_TIMEOUT_MS = 180_000;

/**
 * Appended to the original user prompt for the single nudge call, sent when
 * the first call ended without writing persona.md.
 */
export const PERSONA_WRITE_NUDGE =
  `\n\n---\n\n⚠️ **提醒**：你上一轮结束时没有调用 **write** 或 **edit** 工具，persona.md 没有被更新。` +
  `本轮必须调用 **write**（整体写入）或 **edit**（局部修改）工具把最终内容写入 \`persona.md\`；` +
  `只在回复里输出文字不算完成。`;

/** Closed set of reasons one generation ended with. `none` = no failure. */
export type PersonaGenerationReason =
  | "no-write"
  | "timeout"
  | "provider-error"
  | "aborted"
  | "deadline"
  | "checkpoint-rejected"
  | "write-error"
  | "none";

/** Raw outcome; `skipped` = no changed scene, the runner was never called. */
export type PersonaGenerationOutcome = "generated" | "unchanged" | "failed" | "skipped";

export interface PersonaGenerationDetail {
  outcome: PersonaGenerationOutcome;
  reason: PersonaGenerationReason;
  /** A second, nudging runner call was made. */
  nudged: boolean;
  /** The nudge call changed persona.md (marker moved on call 2). */
  nudgeWrote: boolean;
  /** At least one runner call was made (definition of a counted generation). */
  invokedRunner: boolean;
  /** Runner calls made (0, 1 or 2). */
  runnerCalls: number;
  /** Persona body length written (after strip + escape); undefined when nothing was written. */
  finalChars?: number;
  /** finalChars is above the limit of the active prompt mode. */
  overLimit: boolean;
}

/** Minimal tool-set contract (see adapters/standalone/persona-tools.ts). */
export interface PersonaToolset {
  tools: Record<string, unknown>;
  violated(): boolean;
}

export class PersonaGenerator {
  private dataDir: string;
  private runner: LLMRunner;
  private logger: Logger | undefined;
  private backupCount: number;
  private promptMode: MemoryPromptMode;
  private memoryPrompt: ResolvedMemoryPrompt | undefined;
  private instanceId: string | undefined;
  private storage: StorageAdapter | undefined;
  private traceContext: TraceContext | undefined;
  private toolset: PersonaToolset | undefined;
  private abortSignal: AbortSignal | undefined;
  private deadlineAt: number | undefined;
  private maxChars: number;
  private maxCharsChat: number | undefined;
  private maxCharsCode: number | undefined;

  constructor(opts: {
    dataDir: string;
    config: unknown;
    model?: string;
    /** Prompt family for L3 generation (default: chat). */
    promptMode?: MemoryPromptMode;
    /** Resolved custom strategy. Undefined preserves the current system prompt exactly. */
    memoryPrompt?: ResolvedMemoryPrompt;
    backupCount?: number;
    logger?: Logger;
    /** Plugin instance ID for metric reporting (optional) */
    instanceId?: string;
    /**
     * Host-neutral LLM runner. When provided, used instead of creating
     * a CleanContextRunner (decouples from OpenClaw runtime).
     * Must be configured with `enableTools: true`.
     */
    llmRunner?: LLMRunner;
    /** StorageAdapter for file operations (COS/local). Falls back to fs when absent. */
    storage?: StorageAdapter;
    /** langfuse 上报身份四元组（team/user/agent/session），填充 trace 顶级字段。 */
    traceContext?: TraceContext;
    /**
     * Persona-only write tools for every runner call of this generation; a
     * write outside persona.md flags a violation and fails the generation.
     */
    tools?: PersonaToolset;
    /** Caller abort (worker lost its lock, shutdown); reaches every runner call. */
    abortSignal?: AbortSignal;
    /** Absolute ms epoch: total deadline of the generation, nudge included. */
    deadlineAt?: number;
    /** Persona body length limits per prompt mode (characters, default 6000). */
    maxCharsChat?: number;
    maxCharsCode?: number;
  }) {
    this.dataDir = opts.dataDir;
    this.logger = opts.logger;
    this.promptMode = opts.promptMode ?? "chat";
    this.memoryPrompt = opts.memoryPrompt;
    this.backupCount = opts.backupCount ?? 3;
    this.instanceId = opts.instanceId;
    this.storage = opts.storage;
    this.traceContext = opts.traceContext;
    this.toolset = opts.tools;
    this.abortSignal = opts.abortSignal;
    this.deadlineAt = opts.deadlineAt;
    this.maxCharsChat = opts.maxCharsChat;
    this.maxCharsCode = opts.maxCharsCode;
    this.maxChars = personaMaxCharsFor(this.promptMode, opts);
    // Use injected LLMRunner if available, otherwise fall back to CleanContextRunner
    this.runner = opts.llmRunner ?? new CleanContextRunner({
      config: opts.config,
      modelRef: opts.model,
      enableTools: true,
      logger: opts.logger,
    });
    this.logger?.debug?.(`${TAG} Generator created: model=${opts.model ?? "(default)"}, promptMode=${this.promptMode}, dataDir=${opts.dataDir}`);
  }

  /**
   * Execute local persona generation without advancing checkpoint.
   */
  async generateLocalPersona(triggerReason?: string): Promise<"generated" | "unchanged" | "failed"> {
    const detail = await this.generateLocalPersonaDetailed(triggerReason);
    return detail.outcome === "skipped" ? "unchanged" : detail.outcome;
  }

  /**
   * Same as generateLocalPersona, with the detail createL3Runner logs and
   * counts once per generation.
   */
  async generateLocalPersonaDetailed(triggerReason?: string): Promise<PersonaGenerationDetail> {
    const startMs = Date.now();
    const deadlineAt = this.deadlineAt ?? startMs + DEFAULT_PERSONA_TIMEOUT_MS;
    let runnerCalls = 0;
    let nudged = false;
    let nudgeWrote = false;
    const done = (
      outcome: PersonaGenerationOutcome,
      reason: PersonaGenerationReason,
      extra: { finalChars?: number; overLimit?: boolean } = {},
    ): PersonaGenerationDetail => ({
      outcome,
      reason,
      nudged,
      nudgeWrote,
      invokedRunner: runnerCalls > 0,
      runnerCalls,
      finalChars: extra.finalChars,
      overLimit: extra.overLimit ?? false,
    });
    this.logger?.debug?.(`${TAG} Starting generation: reason="${triggerReason ?? "none"}"`);

    const cpManager = new CheckpointManager(this.dataDir, this.logger, this.storage);
    const cp = await cpManager.read();
    this.logger?.debug?.(`${TAG} Checkpoint: total_processed=${cp.total_processed}, last_persona_at=${cp.last_persona_at}`);

    const targetFile = StoragePaths.persona;
    const targetLabel = this.promptMode === "code" ? "team operating doctrine" : "persona";

    // 1. Read existing L3 document (strip navigation)
    let existingPersona: string | undefined;
    try {
      let raw: string | null;
      if (this.storage) {
        raw = await this.storage.readFile(targetFile);
      } else {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        raw = await fs.default.readFile(path.default.join(this.dataDir, targetFile), "utf-8");
      }
      if (raw) {
        existingPersona = stripSceneNavigation(raw).trim() || undefined;
      }
      this.logger?.debug?.(`${TAG} Existing ${targetLabel}: ${existingPersona ? `${existingPersona.length} chars` : "empty"}`);
    } catch {
      this.logger?.debug?.(`${TAG} No existing ${targetFile} file`);
    }

    // 2. Load scene index + identify changed scenes
    const index = await readSceneIndex(this.dataDir, this.storage);
    const changedScenes = index.filter((e) => {
      if (!cp.last_persona_time) return true;
      const updatedMs = new Date(e.updated).getTime();
      const personaMs = new Date(cp.last_persona_time).getTime();
      // If either date is unparseable (NaN), treat as changed (conservative)
      if (Number.isNaN(updatedMs) || Number.isNaN(personaMs)) return true;
      return updatedMs > personaMs;
    });
    this.logger?.debug?.(`${TAG} Scene index: ${index.length} total, ${changedScenes.length} changed since last persona`);

    // 3. Read changed scene contents (full raw content including META, matching Python reference)
    const changedSceneContents: string[] = [];
    for (const entry of changedScenes) {
      try {
        let raw: string | null;
        if (this.storage) {
          raw = await this.storage.readFile(`${StoragePaths.sceneBlocksDir}${entry.filename}`);
        } else {
          const fs = await import("node:fs/promises");
          const path = await import("node:path");
          raw = await fs.default.readFile(path.default.join(this.dataDir, "scene_blocks", entry.filename), "utf-8");
        }
        if (!raw) continue;
        changedSceneContents.push(
          `### [${changedSceneContents.length + 1}] ${entry.filename}\n\n\`\`\`markdown\n${raw}\n\`\`\``,
        );
      } catch {
        this.logger?.warn(`${TAG} Could not read scene block: ${entry.filename}`);
      }
    }

    if (changedSceneContents.length === 0 && existingPersona) {
      this.logger?.debug?.(`${TAG} No scene changes and persona exists, skipping generation`);
      return done("skipped", "none");
    }

    // 4. Determine mode
    const mode = existingPersona ? "incremental" : "first";
    this.logger?.debug?.(`${TAG} Generation mode: ${mode}, ${changedSceneContents.length} scene blocks to process`);

    // 5. Build changed scenes section with guidance (matching Python reference format)
    let changedScenesContent: string;
    if (changedSceneContents.length > 0) {
      changedScenesContent =
        `\n\n## 📄 变化场景完整内容\n\n` +
        `*自上次 Persona 更新后，以下 ${changedSceneContents.length} 个场景发生了变化。工程已为你预加载完整内容：*\n\n` +
        changedSceneContents.join("\n\n") +
        `\n\n---\n\n` +
        `⚠️ **重点分析变化场景**：上述场景是自上次更新后的**新增/修改内容**，请**重点分析**这些场景中的新信息。\n`;
    } else {
      changedScenesContent = `\n\n⚠️ **无变化场景**：所有场景均已在上次 Persona 更新中分析过，本次可直接读取所有场景进行全局审视。\n`;
    }

    // 6. Build prompt
    const personaFilePath = this.storage
      ? targetFile
      : await (async () => { const path = await import("node:path"); return path.default.join(this.dataDir, targetFile); })();
    const checkpointPath = this.storage
      ? StoragePaths.checkpoint
      : await (async () => { const path = await import("node:path"); return path.default.join(this.dataDir, ".metadata", "recall_checkpoint.json"); })();

    const { systemPrompt: baseSystemPrompt, userPrompt } = buildPersonaPrompt({
      mode,
      promptMode: this.promptMode,
      currentTime: new Date().toISOString(),
      totalProcessed: cp.total_processed,
      sceneCount: index.length,
      changedSceneCount: changedScenes.length,
      changedScenesContent,
      existingPersona,
      triggerInfo: triggerReason,
      personaFilePath,
      checkpointPath,
      maxCharsChat: this.maxCharsChat,
      maxCharsCode: this.maxCharsCode,
      // Same object the post-write check measures: body after strip + escape.
      existingPersonaChars: existingPersona ? escapeXmlTags(existingPersona).length : undefined,
    });
    const systemPrompt = composeMemorySystemPrompt(baseSystemPrompt, this.memoryPrompt);

    // 7. Backup before LLM run (LLM writes persona.md via tools)
    const bm = new BackupManager(this.storage
      ? undefined  // COS mode: BackupManager not used (TODO: adapt BackupManager for StorageAdapter)
      : await (async () => { const path = await import("node:path"); return path.default.join(this.dataDir, ".backup"); })()
    );
    if (!this.storage) {
      const path = await import("node:path");
      await bm.backupFile(path.default.join(this.dataDir, targetFile), "persona", `offset${cp.total_processed}`, this.backupCount);
    }
    const personaWriteMarkerBefore = await readPersonaWriteMarker(
      this.dataDir,
      targetFile,
      this.storage,
    );

    // 8. Run LLM agent (sandboxed to dataDir, tools enabled — LLM writes target L3 file directly)
    // langfuse trace 语义：L3 persona 生成有独立 name / 顶级 user/session 列 / 可筛选 tags。
    const traceParams = buildTraceParams("memory.persona-generate", this.traceContext);
    const runOnce = async (prompt: string): Promise<PersonaGenerationReason | undefined> => {
      if (this.abortSignal?.aborted) return "aborted";
      const remainingMs = deadlineAt - Date.now();
      if (remainingMs <= 0) return "deadline";
      runnerCalls++;
      try {
        this.logger?.debug?.(`${TAG} Calling LLM for ${targetFile} generation (call=${runnerCalls}, timeout=${remainingMs}ms, tools=enabled, workspaceDir=${this.dataDir})...`);
        await this.runner.run({
          systemPrompt,
          prompt,
          taskId: "persona-generation",
          timeoutMs: remainingMs,
          // maxTokens omitted → core uses the resolved model's maxTokens from catalog
          workspaceDir: this.dataDir,
          // Service mode: LLM tools read/write via StorageAdapter (COS) instead of local FS
          storage: this.storage,
          storagePrefix: this.storage ? "" : undefined,
          ...(this.toolset ? { tools: this.toolset.tools } : {}),
          ...(this.abortSignal ? { abortSignal: this.abortSignal } : {}),
          ...traceParams,
        });
        this.logger?.debug?.(`${TAG} LLM runner completed (call=${runnerCalls})`);
        return undefined;
      } catch (err) {
        const elapsedMs = Date.now() - startMs;
        this.logger?.error(`${TAG} Persona generation failed after ${elapsedMs}ms: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
        return classifyRunError(err, this.abortSignal);
      }
    };

    const firstFailure = await runOnce(userPrompt);
    if (firstFailure) return done("failed", firstFailure);
    if (this.toolset?.violated()) {
      this.logger?.error(`${TAG} LLM tried to write outside ${targetFile}; generation failed`);
      return done("failed", "write-error");
    }
    let personaWriteMarkerAfter = await readPersonaWriteMarker(
      this.dataDir,
      targetFile,
      this.storage,
    );

    // 9. Read LLM-written persona.md and apply post-processing
    const readPersona = async (): Promise<string | undefined> => {
      try {
        let raw: string | null;
        if (this.storage) {
          raw = await this.storage.readFile(targetFile);
        } else {
          const fs = await import("node:fs/promises");
          raw = await fs.default.readFile(personaFilePath, "utf-8");
        }
        return raw || undefined;
      } catch {
        return undefined;
      }
    };
    let personaText = await readPersona();
    if (personaText === undefined) {
      // LLM failed to write persona.md — treat as failure
      this.logger?.error(`${TAG} LLM did not write ${targetFile} — file not found after runner completed`);
      return done("failed", "no-write");
    }
    if (
      existingPersona &&
      personaWriteMarkerBefore &&
      personaWriteMarkerAfter === personaWriteMarkerBefore
    ) {
      // The model ended without write/edit: one nudge, inside the same deadline.
      this.logger?.warn(`${TAG} LLM completed without writing ${targetFile}; nudging once to call write/edit`);
      if (this.abortSignal?.aborted) return done("failed", "aborted");
      if (deadlineAt - Date.now() <= 0) {
        this.logger?.error(`${TAG} No time left for the write nudge; preserving checkpoint for retry`);
        return done("failed", "deadline");
      }
      nudged = true;
      const nudgeFailure = await runOnce(`${userPrompt}${PERSONA_WRITE_NUDGE}`);
      if (nudgeFailure) return done("failed", nudgeFailure);
      if (this.toolset?.violated()) {
        this.logger?.error(`${TAG} LLM tried to write outside ${targetFile}; generation failed`);
        return done("failed", "write-error");
      }
      personaWriteMarkerAfter = await readPersonaWriteMarker(this.dataDir, targetFile, this.storage);
      if (personaWriteMarkerAfter === personaWriteMarkerBefore) {
        this.logger?.error(
          `${TAG} LLM completed without writing ${targetFile} after the nudge; preserving checkpoint for retry`,
        );
        return done("failed", "no-write");
      }
      nudgeWrote = true;
      personaText = await readPersona();
      if (personaText === undefined) {
        this.logger?.error(`${TAG} LLM did not write ${targetFile} — file not found after the nudge`);
        return done("failed", "no-write");
      }
    }

    // 10. Strip any navigation the LLM might have added + sanitize for safe injection
    personaText = escapeXmlTags(stripSceneNavigation(personaText).trim());

    if (!personaText) {
      this.logger?.error(`${TAG} LLM wrote empty ${targetFile} — skipping`);
      return done("failed", "write-error");
    }

    const personaUnchanged =
      existingPersona !== undefined &&
      personaText === escapeXmlTags(existingPersona);

    // Length check on the body (navigation excluded): over the limit is kept
    // and reported, never rejected.
    const finalChars = personaText.length;
    const overLimit = finalChars > this.maxChars;
    if (overLimit) {
      this.logger?.warn(
        `${TAG} ${targetFile} body is ${finalChars} chars, over the ${this.maxChars}-char limit (promptMode=${this.promptMode}); kept`,
      );
    }

    // 11. Append fresh scene navigation and write final content — unless the
    // caller aborted or the generation deadline has passed.
    if (this.abortSignal?.aborted) {
      this.logger?.error(`${TAG} Aborted before the final ${targetFile} write; nothing written`);
      return done("failed", "aborted");
    }
    if (Date.now() >= deadlineAt) {
      this.logger?.error(`${TAG} Generation deadline passed before the final ${targetFile} write; nothing written`);
      return done("failed", "deadline");
    }
    const nav = generateSceneNavigation(index, undefined, false);
    const finalContent = nav ? `${personaText}\n\n${nav}\n` : personaText;
    try {
      if (this.storage) {
        await this.storage.writeFile(targetFile, finalContent);
      } else {
        const fs = await import("node:fs/promises");
        await fs.default.writeFile(personaFilePath, finalContent, "utf-8");
      }
    } catch (err) {
      this.logger?.error(`${TAG} Final ${targetFile} write failed: ${err instanceof Error ? err.message : String(err)}`);
      return done("failed", "write-error");
    }

    const elapsedMs = Date.now() - startMs;
    this.logger?.info(`${TAG} ${targetFile} written (${finalContent.length} chars) in ${elapsedMs}ms`);

    // ── l3_persona_generation metric ──
    if (this.instanceId && this.logger) {
      report("l3_persona_generation", {
        triggerReason: triggerReason ?? "unknown",
        mode: existingPersona ? "incremental" : "initial",
        newPersonaContent: personaText,
        newPersonaLength: personaText.length,
        totalDurationMs: elapsedMs,
        success: true,
        error: null,
      });
    }

    // ── 评测指标：L3 延迟 + 画像变化 ──
    try {
      reportL3LatencyMetrics({
        instanceId: this.instanceId ?? "",
        generationLatencyMs: elapsedMs,
        personaLengthBefore: existingPersona ? existingPersona.length : 0,
        personaLengthAfter: finalContent.length,
        personaTextBefore: existingPersona ?? "",
        personaTextAfter: personaText,
        hasError: false,
      });
    } catch {
      // 静默忽略
    }

    return done(personaUnchanged ? "unchanged" : "generated", "none", { finalChars, overLimit });
  }

  /**
   * Backward-compatible wrapper: local generation + checkpoint advance.
   */
  async generate(triggerReason?: string): Promise<boolean> {
    const result = await this.generateLocalPersona(triggerReason);
    if (result !== "generated") return false;

    const cpManager = new CheckpointManager(this.dataDir, this.logger, this.storage);
    const cp = await cpManager.read();
    await cpManager.markPersonaGenerated(cp.total_processed);
    return true;
  }
}

/** Map a runner error to the closed reason enum. */
function classifyRunError(err: unknown, signal: AbortSignal | undefined): PersonaGenerationReason {
  if (signal?.aborted) return "aborted";
  const name = err instanceof Error ? err.name : "";
  const message = err instanceof Error ? err.message : String(err);
  // The runner's own timeout aborts the call (AbortSignal.timeout): with no
  // caller abort, an abort/timeout error is the generation timing out.
  if (name === "TimeoutError" || name === "AbortError" || /time[d]?[\s-]?out/i.test(message)) return "timeout";
  return "provider-error";
}

async function readPersonaWriteMarker(
  dataDir: string,
  targetFile: string,
  storage?: StorageAdapter,
): Promise<string | undefined> {
  try {
    if (storage) {
      const stat = await storage.stat(targetFile);
      return stat ? `${stat.size}:${stat.lastModified}` : undefined;
    }
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const stat = await fs.default.stat(path.default.join(dataDir, targetFile));
    return `${stat.size}:${stat.mtimeMs}`;
  } catch {
    return undefined;
  }
}
