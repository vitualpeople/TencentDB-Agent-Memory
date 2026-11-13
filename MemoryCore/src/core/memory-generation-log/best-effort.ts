import type { Logger } from "../types.js";
import { assertWriteAllowed, type CheckpointWriteGuard } from "../../utils/checkpoint.js";

/**
 * Write a generation log, then its references, never failing the generation.
 *
 * `guard` (L3 only; the L1 and L2 callers pass none): checked before each of
 * the two writes, so neither starts after the caller's abort or deadline. A
 * write already in flight finishes; when the guard trips while writeLog()
 * runs, writeRefs() is not started.
 */
export async function writeGenerationProvenanceBestEffort(params: {
  layer: "l1" | "l2" | "l3";
  logger?: Logger;
  writeLog: () => Promise<void>;
  writeRefs?: () => Promise<void>;
  guard?: CheckpointWriteGuard;
}): Promise<void> {
  const { layer, logger, writeLog, writeRefs, guard } = params;
  try {
    assertWriteAllowed(guard);
    await writeLog();
  } catch (error) {
    logger?.warn?.(
      `[memory-generation-log] ${layer} log write failed; memory generation remains successful: ${error instanceof Error ? error.message : String(error)}`,
    );
    return;
  }

  if (!writeRefs) return;
  try {
    assertWriteAllowed(guard);
    await writeRefs();
  } catch (error) {
    logger?.warn?.(
      `[memory-generation-log] ${layer} reference write failed; memory generation remains successful: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}
