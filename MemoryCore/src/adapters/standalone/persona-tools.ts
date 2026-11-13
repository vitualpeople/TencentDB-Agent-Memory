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
 */

import path from "node:path";
import type { StorageAdapter } from "../../core/storage/adapter.js";
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
}

interface ExecutableTool {
  execute?: (args: { path: string }, options?: unknown) => Promise<unknown>;
}

function hasParentSegment(rawPath: string): boolean {
  return rawPath.replace(/\\/g, "/").split("/").includes("..");
}

/**
 * Build the persona tool set for one generation.
 *
 * @param workspaceDir - FS sandbox root (scoped data dir); used when `storage` is absent
 * @param storage      - scoped StorageAdapter (service mode); keys are resolved with prefix ""
 */
export function createPersonaWriteTools(opts: {
  workspaceDir: string;
  storage?: StorageAdapter;
  logger?: Logger;
}): PersonaWriteToolset {
  const { workspaceDir, storage, logger } = opts;
  let violation = false;

  let base: Record<string, unknown>;
  let isAllowed: (rawPath: string) => boolean;
  if (storage) {
    base = createStorageTools(storage, "", logger);
    const target = resolveStorageKey("", PERSONA_TARGET_FILE);
    isAllowed = (rawPath) => !hasParentSegment(rawPath) && resolveStorageKey("", rawPath) === target;
  } else {
    base = createSandboxedTools(workspaceDir, logger as never);
    const target = path.resolve(workspaceDir, PERSONA_TARGET_FILE);
    isAllowed = (rawPath) => {
      if (hasParentSegment(rawPath)) return false;
      const resolved = resolveSandboxedPath(workspaceDir, rawPath);
      return resolved !== null && path.resolve(resolved) === target;
    };
  }

  const guard = (name: "write" | "edit") => {
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
        return inner.execute!(args, options);
      },
    };
  };

  return {
    tools: { ...base, write: guard("write"), edit: guard("edit") },
    violated: () => violation,
  };
}
