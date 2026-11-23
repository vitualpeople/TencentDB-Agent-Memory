/**
 * Classification of LLM failures caused by an exhausted quota or a rate limit.
 *
 * PipelineWorker defers an L1/L2/L3 task that failed this way instead of
 * retrying it into the dead-letter queue: a quota window lasts minutes to
 * hours, far longer than the worker's 5 s / 15 s / 45 s retries.
 *
 * An error is a quota error when, anywhere in its chain (`cause`, the AI SDK
 * RetryError's `lastError` / `errors`), it carries:
 *   - HTTP status 429 (`statusCode` on the AI SDK APICallError, `status` on
 *     other clients), or
 *   - a MiniMax `base_resp.status_code` from LLM_QUOTA_PROVIDER_CODES, read
 *     from the error's parsed body (`data`, `value`) or its raw
 *     `responseBody`. MiniMax puts `base_resp` at the top level of both its
 *     OpenAI- and Anthropic-compatible bodies (probe 2026-11-23); the error
 *     codes come from the MiniMax docs, no failing call was observed yet.
 *
 * Nothing here returns or logs the provider body: describeLlmQuotaError
 * reports only the status and the provider code.
 */
import type { LLMRunner, LLMRunParams } from "../core/types.js";

/**
 * MiniMax `base_resp.status_code` values that mean "quota or rate limit":
 * 1002 rate limit, 1008 insufficient balance, 1039 token-per-minute limit,
 * 2045 request growth-rate limit, 2056 plan usage limit exceeded.
 */
export const LLM_QUOTA_PROVIDER_CODES: ReadonlySet<number> = new Set([1002, 1008, 1039, 2045, 2056]);

/** How deep the error chain is followed (guards against cyclic causes). */
const MAX_CHAIN_DEPTH = 8;

export interface LlmQuotaSignal {
  /** HTTP status of the failing call, when one was found. */
  status?: number;
  /** MiniMax base_resp.status_code, when one was found. */
  providerCode?: number;
}

/**
 * Typed quota failure thrown by the L2 and L3 runners, whose generators
 * otherwise turn every LLM error into a plain "failed" result. The message
 * names only the status / provider code, never the provider body.
 */
export class LlmQuotaError extends Error {
  readonly signal: LlmQuotaSignal;

  constructor(cause: unknown, context?: string) {
    const signal = findLlmQuotaSignal(cause) ?? {};
    super(`LLM quota or rate limit${context ? ` (${context})` : ""}: ${formatSignal(signal)}`, { cause });
    this.name = "LlmQuotaError";
    this.signal = signal;
  }
}

/** True when `err` (or an error in its chain) is a quota / rate-limit failure. */
export function isLlmQuotaError(err: unknown): boolean {
  return findLlmQuotaSignal(err) !== undefined;
}

/** Short, body-free description of a quota error for logs ("status=429", "provider_code=1008"). */
export function describeLlmQuotaError(err: unknown): string {
  return formatSignal(findLlmQuotaSignal(err) ?? {});
}

/** The first quota signal found in the error chain, or undefined. */
export function findLlmQuotaSignal(err: unknown): LlmQuotaSignal | undefined {
  const seen = new Set<unknown>();
  const queue: Array<{ value: unknown; depth: number }> = [{ value: err, depth: 0 }];
  while (queue.length > 0) {
    const { value, depth } = queue.shift()!;
    if (!value || typeof value !== "object" || seen.has(value) || depth > MAX_CHAIN_DEPTH) continue;
    seen.add(value);
    if (value instanceof LlmQuotaError) return value.signal;

    const e = value as Record<string, unknown>;
    const status = numeric(e.statusCode) ?? numeric(e.status);
    if (status === 429) return { status };
    const providerCode = providerCodeOf(e);
    if (providerCode !== undefined) return { ...(status !== undefined ? { status } : {}), providerCode };

    for (const next of [e.cause, e.lastError]) queue.push({ value: next, depth: depth + 1 });
    if (Array.isArray(e.errors) && e.errors.length > 0) queue.push({ value: e.errors[e.errors.length - 1], depth: depth + 1 });
  }
  return undefined;
}

/** MiniMax quota code from a parsed body (`data`, `value`) or the raw `responseBody`. */
function providerCodeOf(e: Record<string, unknown>): number | undefined {
  for (const body of [e.data, e.value, parseJson(e.responseBody)]) {
    const code = baseRespCode(body);
    if (code !== undefined && LLM_QUOTA_PROVIDER_CODES.has(code)) return code;
  }
  return undefined;
}

function baseRespCode(body: unknown): number | undefined {
  if (!body || typeof body !== "object") return undefined;
  const baseResp = (body as Record<string, unknown>).base_resp;
  if (!baseResp || typeof baseResp !== "object") return undefined;
  return numeric((baseResp as Record<string, unknown>).status_code);
}

function parseJson(raw: unknown): unknown {
  if (typeof raw !== "string" || !raw.includes("base_resp")) return undefined;
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function numeric(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function formatSignal(signal: LlmQuotaSignal): string {
  const parts: string[] = [];
  if (signal.status !== undefined) parts.push(`status=${signal.status}`);
  if (signal.providerCode !== undefined) parts.push(`provider_code=${signal.providerCode}`);
  return parts.length > 0 ? parts.join(", ") : "status=?";
}

/**
 * Wraps an LLMRunner and remembers the last quota error one of its calls
 * threw. The L2 and L3 generators swallow LLM errors into a result value;
 * their runners read `quotaError` afterwards to throw a typed LlmQuotaError.
 * Pure delegation otherwise (create one per run, never shared).
 */
export function watchLlmQuota(runner: LLMRunner): { runner: LLMRunner; quotaError: () => unknown } {
  let quotaError: unknown;
  const watched = Object.create(runner) as LLMRunner;
  watched.run = async (params: LLMRunParams) => {
    try {
      return await runner.run(params);
    } catch (err) {
      if (isLlmQuotaError(err)) quotaError = err;
      throw err;
    }
  };
  return { runner: watched, quotaError: () => quotaError };
}
