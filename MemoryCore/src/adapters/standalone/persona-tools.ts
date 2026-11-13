/**
 * L3 persona tool set: the default read/write/edit tools (local FS or
 * StorageAdapter), with write and edit allowed on exactly one target,
 * `persona.md` at the sandbox root.
 *
 * The allowlist is checked on the path/key the sandbox resolver produces
 * (resolveSandboxedPath for FS, resolveStorageKey for storage), so the check
 * and the real write agree on the target. A path with a ".." segment is
 * refused even when it resolves back to persona.md, on both backends.
 *
 * A refused call returns an error to the model, is logged (path only, never
 * content) and sets a per-generation violation flag. The persona generator
 * reads the flag after its runner calls and fails the generation when set,
 * even if the model later wrote persona.md correctly.
 *
 * Write guard: with `guard`, no write/edit starts after the caller's abort or
 * the generation deadline. It is checked before the inner tool runs and again
 * by the inner tool right before its write (after edit's read). A refusal is a
 * tool error, not a violation.
 *
 * Write evidence: `writesObserved()` counts write/edit calls to persona.md
 * whose inner tool answered `{"success": true}`. The generator uses it as
 * positive proof of a write, also on a backend whose `stat` cannot give a
 * modification marker.
 */

import path from "node:path";
import type { StorageAdapter } from "../../core/storage/adapter.js";
import { assertWriteAllowed, CheckpointWriteRejectedError, type CheckpointWriteGuard } from "../../utils/checkpoint.js";
import { createSandboxedTools, resolveSandboxedPath } from "./llm-runner.js";
import { createStorageTools, resolveStorageKey } from "./storage-tools.js";

const TAG = "[memory-tdai] [persona-tools]";

/** The only file an L3 persona generation may write. */
export const PERSONA_TARGET_FILE = "persona.md";

interface Logger {
  debug?: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

export interface PersonaWriteToolset {
  /** Tool dict for LLMRunParams.tools. */
  tools: Record<string, unknown>;
  /** True once a write/edit targeted anything but persona.md in this generation. */
  violated(): boolean;
  /** Successful write/edit calls to persona.md in this generation (starts at 0). */
  writesObserved(): number;
}

interface ExecutableTool {
  execute?: (args: { path: string }, options?: unknown) => Promise<unknown>;
}

function hasParentSegment(rawPath: string): boolean {
  return rawPath.replace(/\\/g, "/").split("/").includes("..");
}

/** The inner tools answer a JSON string; `{"success": true}` means the write happened. */
function reportsSuccess(result: unknown): boolean {
  if (typeof result !== "string") return false;
  try {
    return (JSON.parse(result) as { success?: unknown }).success === true;
  } catch {
    return false;
  }
}

/**
 * Build the persona tool set for one generation (one scope run).
 *
 * @param workspaceDir - FS sandbox root (scoped data dir); used when `storage` is absent
 * @param storage      - scoped StorageAdapter (service mode); keys are resolved with prefix ""
 * @param guard        - caller abort + generation deadline; no write starts after either
 */
export function createPersonaWriteTools(opts: {
  workspaceDir: string;
  storage?: StorageAdapter;
  logger?: Logger;
  guard?: CheckpointWriteGuard;
}): PersonaWriteToolset {
  const { workspaceDir, storage, logger, guard } = opts;
  let violation = false;
  let writes = 0;

  let base: Record<string, unknown>;
  let isAllowed: (rawPath: string) => boolean;
  if (storage) {
    base = createStorageTools(storage, "", logger, guard);
    const target = resolveStorageKey("", PERSONA_TARGET_FILE);
    isAllowed = (rawPath) => !hasParentSegment(rawPath) && resolveStorageKey("", rawPath) === target;
  } else {
    base = createSandboxedTools(workspaceDir, logger as never, guard);
    const target = path.resolve(workspaceDir, PERSONA_TARGET_FILE);
    isAllowed = (rawPath) => {
      if (hasParentSegment(rawPath)) return false;
      const resolved = resolveSandboxedPath(workspaceDir, rawPath);
      return resolved !== null && path.resolve(resolved) === target;
    };
  }

  const wrap = (name: "write" | "edit") => {
    const inner = base[name] as ExecutableTool;
    return {
      ...inner,
      execute: async (args: { path: string }, options?: unknown) => {
        const rawPath = typeof args?.path === "string" ? args.path : "";
        if (!isAllowed(rawPath)) {
          violation = true;
          logger?.error(`${TAG} refused ${name} to "${rawPath}": only ${PERSONA_TARGET_FILE} may be written`);
          return JSON.stringify({ error: `Only ${PERSONA_TARGET_FILE} may be written; "${rawPath}" was refused.` });
        }
        try {
          assertWriteAllowed(guard);
        } catch (err) {
          if (!(err instanceof CheckpointWriteRejectedError)) throw err;
          logger?.warn(`${TAG} ${name} to ${PERSONA_TARGET_FILE} not started: generation ${err.rejection}`);
          return JSON.stringify({ error: `${name} not started: generation ${err.rejection}` });
        }
        const result = await inner.execute!(args, options);
        if (reportsSuccess(result)) writes++;
        return result;
      },
    };
  };

  return {
    tools: { ...base, write: wrap("write"), edit: wrap("edit") },
    violated: () => violation,
    writesObserved: () => writes,
  };
}
