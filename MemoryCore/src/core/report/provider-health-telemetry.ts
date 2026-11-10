/**
 * Process-wide health telemetry for the two upstream providers the pipeline
 * depends on: the embedding API and the LLM API.
 *
 * Why a shared sink rather than a counter on one instance: runners are built
 * in many places (gateway skill extractor, TdaiCore L1/dedup/L2/L3 factories,
 * seed runtime, host adapter) and embedding services exist per instance in
 * service mode. A counter attached to one of them would miss the failures of
 * the others. Every StandaloneLLMRunner and OpenAIEmbeddingService reports
 * into the default sink below unless a test injects its own, and /health
 * reads the same sink.
 *
 * Only the HTTP status is kept, never a response body: provider error bodies
 * can echo part of the API key.
 */

/** One provider's health block as /health reports it. */
export interface ProviderHealthSnapshot {
  /** ISO time of the last successful call, null if none yet. */
  lastOkAt: string | null;
  /** ISO time of the last failed call, null if none yet. */
  lastErrorAt: string | null;
  /** HTTP status of the last failure; null when it had none (timeout, network, bad body). */
  lastErrorStatus: number | null;
  /** Failed calls since the last success. */
  consecutiveFailures: number;
}

/** Consecutive failures at which /health reports `degraded`. */
export const PROVIDER_DEGRADED_THRESHOLD = 3;

export class ProviderHealthSink {
  private lastOkAt: string | null = null;
  private lastErrorAt: string | null = null;
  private lastErrorStatus: number | null = null;
  private consecutiveFailures = 0;

  recordSuccess(now: Date = new Date()): void {
    this.lastOkAt = now.toISOString();
    this.consecutiveFailures = 0;
  }

  recordFailure(httpStatus: number | null | undefined, now: Date = new Date()): void {
    this.lastErrorAt = now.toISOString();
    this.lastErrorStatus = typeof httpStatus === "number" && Number.isFinite(httpStatus) ? httpStatus : null;
    this.consecutiveFailures++;
  }

  snapshot(): ProviderHealthSnapshot {
    return {
      lastOkAt: this.lastOkAt,
      lastErrorAt: this.lastErrorAt,
      lastErrorStatus: this.lastErrorStatus,
      consecutiveFailures: this.consecutiveFailures,
    };
  }

  isDegraded(): boolean {
    return this.consecutiveFailures >= PROVIDER_DEGRADED_THRESHOLD;
  }
}

/** Default sink for every remote embedding service in this process. */
export const embeddingProviderHealth = new ProviderHealthSink();

/** Default sink for every StandaloneLLMRunner in this process. */
export const llmProviderHealth = new ProviderHealthSink();

/**
 * L0 records written without a vector because the embedding call failed
 * (`/v2/conversation/add`). The record is kept so no message is lost; this
 * count says how many need a vector backfilled.
 */
let l0EmbeddingFailed = 0;

export function incrementL0EmbeddingFailed(): void {
  l0EmbeddingFailed++;
}

export function getL0EmbeddingFailed(): number {
  return l0EmbeddingFailed;
}
