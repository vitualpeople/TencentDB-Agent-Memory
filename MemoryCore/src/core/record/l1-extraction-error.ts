/**
 * Typed L1 extraction failure.
 *
 * Thrown by extractL1Memories when the LLM call fails or its reply is not a
 * valid scene array, so the caller can tell "the extraction failed" apart
 * from "the LLM found no memory" (a valid empty array `[]`). The L1 runner
 * re-throws one aggregated error per slice; PipelineWorker reads it to retry,
 * dead-letter and cap slices that keep failing to parse.
 *
 * Kept in its own module so PipelineWorker can recognise it with
 * `instanceof` without importing the extractor and its dependencies.
 *
 * - kind "provider": the LLM call itself failed (HTTP 401/429/5xx, timeout,
 *   abort, network). Retrying later can succeed.
 * - kind "parse": the call returned, but the reply is not a JSON array of
 *   scene objects with `message_ids` and `memories` arrays.
 */
export type L1ExtractionErrorKind = "provider" | "parse";

export class L1ExtractionError extends Error {
  readonly kind: L1ExtractionErrorKind;
  /** L0 message ids of the group(s) whose extraction failed. */
  frozenMessageIds: string[];
  /** Highest recorded_at_ms of the slice the failure belongs to (set by the runner). */
  upperBoundMs?: number;
  /** L2 profile keys of groups in the same slice that did finish (set by the runner). */
  completedProfileScopes: string[];

  constructor(
    kind: L1ExtractionErrorKind,
    message: string,
    opts?: { frozenMessageIds?: string[]; upperBoundMs?: number; completedProfileScopes?: string[]; cause?: unknown },
  ) {
    super(message, opts?.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = "L1ExtractionError";
    this.kind = kind;
    this.frozenMessageIds = opts?.frozenMessageIds ?? [];
    this.upperBoundMs = opts?.upperBoundMs;
    this.completedProfileScopes = opts?.completedProfileScopes ?? [];
  }
}

export function isL1ExtractionError(err: unknown): err is L1ExtractionError {
  return err instanceof L1ExtractionError;
}
