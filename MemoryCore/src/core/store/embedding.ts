/**
 * Embedding Service: converts text to vector embeddings.
 *
 * Supports two providers:
 * - "openai": OpenAI-compatible embedding APIs (OpenAI, Azure OpenAI, self-hosted)
 * - "local": node-llama-cpp with embeddinggemma-300m GGUF model (fully offline)
 *
 * When no remote embedding is configured, automatically falls back to local provider.
 *
 * Design:
 * - Single `embed()` for one text, `embedBatch()` for multiple.
 * - `getDimensions()` returns configured vector dimensions.
 * - Throws on failure; callers decide fallback strategy.
 */

import type { Logger } from "../types.js";
import { embeddingProviderHealth, type ProviderHealthSink } from "../report/provider-health-telemetry.js";

// ============================
// Types
// ============================

export interface OpenAIEmbeddingConfig {
  /** Provider identifier — any value other than "local" (e.g. "openai", "deepseek", "azure", "qclaw") */
  provider: string;
  /** API base URL (required — must be specified by user, e.g. "https://api.openai.com/v1") */
  baseUrl: string;
  /** API Key (required) */
  apiKey: string;
  /** Model name (required — must be specified by user) */
  model: string;
  /** Output dimensions (required — must match the chosen model) */
  dimensions: number;
  /**
   * Whether to include the `dimensions` field in the embeddings request body.
   * Defaults to `true` for backward compatibility with OpenAI's `text-embedding-3-*`
   * (Matryoshka representation). Some self-hosted / OSS models (e.g. BGE-M3) reject
   * unknown `dimensions` parameters with HTTP 400; set this to `false` for those.
   */
  sendDimensions?: boolean;
  /** Local proxy URL (only for provider="qclaw") — requests are forwarded through this proxy with Remote-URL header */
  proxyUrl?: string;
  /** Max input text length in characters before truncation (default: 5000). */
  maxInputChars?: number;
  /** Timeout per API call in milliseconds (default: 10000). */
  timeoutMs?: number;
}

export interface LocalEmbeddingConfig {
  provider: "local";
  /** Custom GGUF model path (default: embeddinggemma-300m from HuggingFace) */
  modelPath?: string;
  /** Model cache directory (default: node-llama-cpp default cache) */
  modelCacheDir?: string;
}

export type EmbeddingConfig = OpenAIEmbeddingConfig | LocalEmbeddingConfig;

/** Identifies the embedding provider + model for change detection. */
export interface EmbeddingProviderInfo {
  /** Provider identifier (e.g. "local", "openai", "deepseek") */
  provider: string;
  /** Model identifier (e.g. "embeddinggemma-300m", "text-embedding-3-large") */
  model: string;
}

export interface EmbeddingCallOptions {
  /** Override the default timeout for this call (milliseconds). */
  timeoutMs?: number;
}

export interface EmbeddingService {
  /** Get embedding for a single text */
  embed(text: string, options?: EmbeddingCallOptions): Promise<Float32Array>;
  /** Get embeddings for multiple texts (batched API call) */
  embedBatch(texts: string[], options?: EmbeddingCallOptions): Promise<Float32Array[]>;
  /** Return the configured vector dimensions */
  getDimensions(): number;
  /** Return provider + model identifiers for change detection */
  getProviderInfo(): EmbeddingProviderInfo;
  /**
   * Whether the service is ready to serve embed requests.
   * For remote providers (OpenAI), always true (stateless HTTP).
   * For local providers, true only after model download + load completes.
   */
  isReady(): boolean;
  /**
   * Start background warmup (model download + load).
   * For remote providers, this is a no-op.
   * For local providers, triggers async initialization without blocking.
   * Safe to call multiple times (idempotent).
   */
  startWarmup(): void;
  /** Optional: release resources (model memory, GPU, etc.) on shutdown */
  close?(): void | Promise<void>;
}

/**
 * Error thrown when embed() / embedBatch() is called before the local
 * embedding model has finished downloading and loading.
 * Callers should catch this and fall back to keyword-only mode.
 */
export class EmbeddingNotReadyError extends Error {
  constructor(message?: string) {
    super(message ?? "Local embedding model is not ready yet (still downloading or loading)");
    this.name = "EmbeddingNotReadyError";
  }
}

const TAG = "[memory-tdai][embedding]";

// ============================
// Local (node-llama-cpp) implementation
// ============================

/** Default model: Google's embeddinggemma-300m, quantized Q8_0 (~300MB) */
const DEFAULT_LOCAL_MODEL =
  "hf:ggml-org/embeddinggemma-300m-qat-q8_0-GGUF/embeddinggemma-300m-qat-Q8_0.gguf";

/** embeddinggemma-300m outputs 768-dimensional vectors */
const LOCAL_DIMENSIONS = 768;

/**
 * embeddinggemma-300m has a 256-token context window.
 * As a safe heuristic, we limit input to ~600 chars for CJK text
 * (CJK characters typically tokenize to 1-2 tokens each,
 *  so 600 chars ≈ 200-400 tokens, keeping well within 256-token limit
 *  after accounting for special tokens).
 * For Latin text, ~800 chars is a safe limit (~200 tokens).
 * We use 512 chars as a conservative universal limit.
 */
const LOCAL_MAX_INPUT_CHARS = 512;

/**
 * Sanitize NaN/Inf values and L2-normalize the vector.
 * Matches OpenClaw's own sanitizeAndNormalizeEmbedding().
 */
function sanitizeAndNormalize(vec: number[] | Float32Array): Float32Array {
  const arr = Array.from(vec).map((v) => (Number.isFinite(v) ? v : 0));
  const magnitude = Math.sqrt(arr.reduce((sum, v) => sum + v * v, 0));
  if (magnitude < 1e-10) {
    return new Float32Array(arr);
  }
  return new Float32Array(arr.map((v) => v / magnitude));
}

/**
 * Initialization state for LocalEmbeddingService.
 * - "idle":         not started yet
 * - "initializing": model download / load is in progress (background)
 * - "ready":        model is loaded and ready to serve
 * - "failed":       initialization failed (will retry on next startWarmup)
 */
type LocalInitState = "idle" | "initializing" | "ready" | "failed";

/** Function that dynamically imports node-llama-cpp. Overridable for testing. */
export type ImportLlamaFn = () => Promise<{
  getLlama: (opts: { logLevel: number }) => Promise<unknown>;
  resolveModelFile: (model: string, cacheDir?: string) => Promise<string>;
  LlamaLogLevel: { error: number };
}>;

const defaultImportLlama: ImportLlamaFn = () => import("node-llama-cpp") as unknown as ReturnType<ImportLlamaFn>;

export class LocalEmbeddingService implements EmbeddingService {
  private readonly modelPath: string;
  private readonly modelCacheDir?: string;
  private readonly logger?: Logger;
  private readonly importLlama: ImportLlamaFn;

  // Initialization state machine
  private initState: LocalInitState = "idle";
  private initPromise: Promise<void> | null = null;
  private initError: Error | null = null;
  private embeddingContext: {
    getEmbeddingFor: (text: string) => Promise<{ vector: Float32Array | number[] }>;
  } | null = null;

  constructor(config?: LocalEmbeddingConfig, logger?: Logger, importLlama?: ImportLlamaFn) {
    this.modelPath = config?.modelPath?.trim() || DEFAULT_LOCAL_MODEL;
    this.modelCacheDir = config?.modelCacheDir?.trim();
    this.logger = logger;
    this.importLlama = importLlama ?? defaultImportLlama;
  }

  getDimensions(): number {
    return LOCAL_DIMENSIONS;
  }

  getProviderInfo(): EmbeddingProviderInfo {
    return { provider: "local", model: this.modelPath };
  }

  /**
   * Whether the local model is fully loaded and ready to serve requests.
   */
  isReady(): boolean {
    return this.initState === "ready" && this.embeddingContext !== null;
  }

  /**
   * Start background warmup: download model (if needed) and load into memory.
   * Does NOT block the caller — returns immediately.
   * Safe to call multiple times (idempotent); re-triggers on "failed" state.
   */
  startWarmup(): void {
    if (this.initState === "initializing" || this.initState === "ready") {
      return; // already in progress or done
    }
    this.logger?.info(`${TAG} Starting background warmup for local embedding model...`);
    this.initState = "initializing";
    this.initError = null;

    this.initPromise = this._doInitialize()
      .then(() => {
        this.initState = "ready";
        this.logger?.info(`${TAG} Background warmup complete — local embedding ready`);
      })
      .catch((err) => {
        this.initState = "failed";
        this.initError = err instanceof Error ? err : new Error(String(err));
        this.logger?.error(
          `${TAG} Background warmup failed: ${this.initError.message}. ` +
          `embed() calls will throw EmbeddingNotReadyError until retried.`,
        );
      });
  }

  /**
   * Get embedding for a single text.
   * @throws {EmbeddingNotReadyError} if model is not yet ready.
   */
  async embed(text: string, _options?: EmbeddingCallOptions): Promise<Float32Array> {
    this.assertReady();
    const truncated = this.truncateInput(text);
    const embedding = await this.embeddingContext!.getEmbeddingFor(truncated);
    return sanitizeAndNormalize(embedding.vector);
  }

  /**
   * Get embeddings for multiple texts.
   * @throws {EmbeddingNotReadyError} if model is not yet ready.
   */
  async embedBatch(texts: string[], _options?: EmbeddingCallOptions): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    this.assertReady();

    const results: Float32Array[] = [];
    for (const text of texts) {
      const truncated = this.truncateInput(text);
      const embedding = await this.embeddingContext!.getEmbeddingFor(truncated);
      results.push(sanitizeAndNormalize(embedding.vector));
    }
    return results;
  }

  /**
   * Release the node-llama-cpp embedding context and model resources.
   * Safe to call multiple times (idempotent).
   */
  close(): void {
    if (this.embeddingContext) {
      try {
        const ctx = this.embeddingContext as unknown as { dispose?: () => void };
        ctx.dispose?.();
      } catch {
        // best-effort cleanup
      }
      this.embeddingContext = null;
      this.initPromise = null;
      this.initState = "idle";
      this.initError = null;
      this.logger?.info(`${TAG} Local embedding resources released`);
    }
  }

  /**
   * Assert the model is ready. Throws EmbeddingNotReadyError if not.
   */
  private assertReady(): void {
    if (this.initState === "ready" && this.embeddingContext) {
      return;
    }
    if (this.initState === "failed") {
      throw new EmbeddingNotReadyError(
        `Local embedding model initialization failed: ${this.initError?.message ?? "unknown error"}. ` +
        `Call startWarmup() to retry.`,
      );
    }
    if (this.initState === "initializing") {
      throw new EmbeddingNotReadyError(
        "Local embedding model is still loading (download/initialization in progress). Please try again later.",
      );
    }
    // "idle" — startWarmup() was never called
    throw new EmbeddingNotReadyError(
      "Local embedding model warmup has not been started. Call startWarmup() first.",
    );
  }

  /**
   * Truncate input text to stay within the model's context window.
   * embeddinggemma-300m has a 256-token limit; we use a character-based
   * heuristic (LOCAL_MAX_INPUT_CHARS) as a safe proxy.
   */
  private truncateInput(text: string): string {
    if (text.length <= LOCAL_MAX_INPUT_CHARS) return text;
    this.logger?.debug?.(
      `${TAG} Input truncated from ${text.length} to ${LOCAL_MAX_INPUT_CHARS} chars (model context limit)`,
    );
    return text.slice(0, LOCAL_MAX_INPUT_CHARS);
  }

  /**
   * Internal: perform the actual model download + load.
   * Called by startWarmup(), runs in background.
   */
  private async _doInitialize(): Promise<void> {
    // Track partially-initialized resources for cleanup on failure
    let model: { createEmbeddingContext: () => Promise<unknown>; dispose?: () => void } | undefined;
    try {
      this.logger?.debug?.(`${TAG} Loading node-llama-cpp for local embedding...`);

      // Dynamic import — node-llama-cpp is a peer dependency of OpenClaw
      const { getLlama, resolveModelFile, LlamaLogLevel } = await this.importLlama();

      const llama = await getLlama({ logLevel: LlamaLogLevel.error });
      this.logger?.debug?.(`${TAG} Llama instance created`);

      const resolvedPath = await resolveModelFile(
        this.modelPath,
        this.modelCacheDir || undefined,
      );
      this.logger?.debug?.(`${TAG} Model resolved: ${resolvedPath}`);

      model = await (llama as unknown as { loadModel: (opts: { modelPath: string }) => Promise<typeof model> }).loadModel({ modelPath: resolvedPath });
      this.logger?.debug?.(`${TAG} Model loaded, creating embedding context...`);

      this.embeddingContext = await model!.createEmbeddingContext() as typeof this.embeddingContext;
      this.logger?.info(`${TAG} Local embedding ready (model=${this.modelPath}, dims=${LOCAL_DIMENSIONS})`);
    } catch (err) {
      // Clean up partially-initialized resources to prevent leaks
      if (model?.dispose) {
        try { model.dispose(); } catch { /* best-effort */ }
      }
      this.embeddingContext = null;
      throw err;
    }
  }

  /**
   * Wait for ongoing warmup to complete (used internally by tests).
   * Returns immediately if already ready or idle.
   */
  async waitForReady(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }
  }
}

// ============================
// OpenAI-compatible implementation
// ============================

/** Max texts per batch (OpenAI limit is 2048, we use a safe value) */
const MAX_BATCH_SIZE = 256;

/** Max retries for API calls (HTTP 429/5xx, network error, timeout). */
const MAX_RETRIES = 2;
/** Default timeout per API call in milliseconds */
const DEFAULT_API_TIMEOUT_MS = 10_000;
/** Wait before retry n (0-based) is RETRY_BASE_DELAY_MS * (n + 1): 500 ms, then 1000 ms. */
const RETRY_BASE_DELAY_MS = 500;
/**
 * Upper bound on a server-sent `Retry-After`. Deliberately not the call
 * timeout: the caller's time budget for one write assumes at most 1 s per wait.
 */
const RETRY_AFTER_CAP_MS = 1000;

/**
 * Wait in ms asked for by a `Retry-After` header (delta-seconds or HTTP-date),
 * capped at RETRY_AFTER_CAP_MS. Undefined when absent or unparsable.
 */
function retryAfterMs(header: string | null): number | undefined {
  const raw = header?.trim();
  if (!raw) return undefined;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) {
    return seconds >= 0 ? Math.min(seconds * 1000, RETRY_AFTER_CAP_MS) : undefined;
  }
  const at = Date.parse(raw);
  if (Number.isNaN(at)) return undefined;
  return Math.min(Math.max(at - Date.now(), 0), RETRY_AFTER_CAP_MS);
}

/**
 * Custom error class for embedding API errors that carries HTTP status code.
 * Used to distinguish non-retryable client errors (4xx except 429) from
 * retryable server errors (5xx) and rate limits (429).
 */
class EmbeddingApiError extends Error {
  readonly httpStatus: number;
  constructor(message: string, httpStatus: number) {
    super(message);
    this.name = "EmbeddingApiError";
    this.httpStatus = httpStatus;
  }
  /** Returns true for 4xx errors that should NOT be retried (excluding 429). */
  isClientError(): boolean {
    return this.httpStatus >= 400 && this.httpStatus < 500 && this.httpStatus !== 429;
  }
}

/**
 * The API answered 2xx with a body that cannot be used: no `data` array, a
 * count that differs from the number of inputs, or a bad `index`. Not retried:
 * the same request gets the same answer.
 */
class EmbeddingResponseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbeddingResponseError";
  }
}

/**
 * A returned vector's length differs from the configured `dimensions`. Storing
 * it would fail the whole vec0 write, so it is rejected here and the caller
 * keeps the record without a vector instead.
 */
export class EmbeddingDimensionError extends EmbeddingResponseError {
  readonly expected: number;
  readonly actual: number;
  constructor(expected: number, actual: number) {
    super(`Embedding API returned a ${actual}-dimension vector, configured dimensions=${expected}`);
    this.name = "EmbeddingDimensionError";
    this.expected = expected;
    this.actual = actual;
  }
}

interface OpenAIEmbeddingResponse {
  data: Array<{
    index: number;
    embedding: number[];
  }>;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export class OpenAIEmbeddingService implements EmbeddingService {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly dims: number;
  private readonly sendDimensions: boolean;
  private readonly providerName: string;
  private readonly proxyUrl?: string;
  private readonly maxInputChars?: number;
  private readonly timeoutMs: number;
  private readonly logger?: Logger;
  private readonly health: ProviderHealthSink;

  /**
   * @param health  Telemetry sink read by /health. Defaults to the process-wide
   *   embedding sink; tests may inject their own.
   */
  constructor(config: OpenAIEmbeddingConfig, logger?: Logger, health: ProviderHealthSink = embeddingProviderHealth) {
    if (!config.apiKey) {
      throw new Error("EmbeddingService: apiKey is required for remote provider");
    }
    if (!config.baseUrl) {
      throw new Error("EmbeddingService: baseUrl is required for remote provider");
    }
    if (!config.model) {
      throw new Error("EmbeddingService: model is required for remote provider");
    }
    if (!config.dimensions || config.dimensions <= 0) {
      throw new Error("EmbeddingService: dimensions is required for remote provider (must be a positive integer)");
    }
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.model = config.model;
    this.dims = config.dimensions;
    this.sendDimensions = config.sendDimensions ?? true;
    this.providerName = config.provider || "openai";
    this.proxyUrl = config.proxyUrl?.trim() || undefined;
    this.maxInputChars = config.maxInputChars && config.maxInputChars > 0 ? config.maxInputChars : undefined;
    this.timeoutMs = config.timeoutMs && config.timeoutMs > 0 ? config.timeoutMs : DEFAULT_API_TIMEOUT_MS;
    this.logger = logger;
    this.health = health;
  }

  getDimensions(): number {
    return this.dims;
  }

  getProviderInfo(): EmbeddingProviderInfo {
    return { provider: this.providerName, model: this.model };
  }

  /** Remote embedding is always ready (stateless HTTP). */
  isReady(): boolean {
    return true;
  }

  /** No-op for remote embedding (no local model to warm up). */
  startWarmup(): void {
    // nothing to do — remote API is stateless
  }

  async embed(text: string, options?: EmbeddingCallOptions): Promise<Float32Array> {
    const [result] = await this.embedBatch([text], options);
    return result;
  }

  async embedBatch(texts: string[], options?: EmbeddingCallOptions): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    // Truncate texts exceeding maxInputChars limit
    const processedTexts = this.maxInputChars
      ? texts.map((t) => this.truncateInput(t))
      : texts;

    // Split into sub-batches if needed
    if (processedTexts.length > MAX_BATCH_SIZE) {
      const results: Float32Array[] = [];
      for (let i = 0; i < processedTexts.length; i += MAX_BATCH_SIZE) {
        const chunk = processedTexts.slice(i, i + MAX_BATCH_SIZE);
        const chunkResults = await this._callApi(chunk, options?.timeoutMs);
        results.push(...chunkResults);
      }
      return results;
    }

    return this._callApi(processedTexts, options?.timeoutMs);
  }

  /**
   * Truncate input text to stay within the configured maxInputChars limit.
   * Logs a warning when truncation occurs.
   */
  private truncateInput(text: string): string {
    if (!this.maxInputChars || text.length <= this.maxInputChars) return text;
    this.logger?.warn?.(
      `${TAG} Input truncated from ${text.length} to ${this.maxInputChars} chars (maxInputChars limit)`,
    );
    return text.slice(0, this.maxInputChars);
  }

  private async _callApi(texts: string[], timeoutOverride?: number): Promise<Float32Array[]> {
    const body: Record<string, unknown> = {
      input: texts,
      model: this.model,
    };
    if (this.sendDimensions) {
      body.dimensions = this.dims;
    }

    // Determine fetch URL and headers based on proxy mode
    const useProxy = this.providerName === "qclaw" && !!this.proxyUrl;
    const fetchUrl = useProxy ? this.proxyUrl! : `${this.baseUrl}/embeddings`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
    };
    if (useProxy) {
      headers["Remote-URL"] = `${this.baseUrl}/embeddings`;
      this.logger?.debug?.(
        `${TAG} [qclaw-proxy] Forwarding embedding request via proxy: ${fetchUrl}, Remote-URL: ${headers["Remote-URL"]}`,
      );
    }

    // Retry loop with timeout. HTTP 429/5xx, network errors and timeouts are
    // retried after a backoff; 4xx (except 429) and unusable 2xx bodies are not.
    let lastError: Error | undefined;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      let waitMs = RETRY_BASE_DELAY_MS * (attempt + 1);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutOverride ?? this.timeoutMs);
      try {
        const resp = await fetch(fetchUrl, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!resp.ok) {
          // The body is read only for its length: provider error bodies can
          // echo part of the API key, so it never enters a message or log.
          const errBody = await resp.text().catch(() => "");
          const err = new EmbeddingApiError(
            `Embedding API error: HTTP ${resp.status} ${resp.statusText} (body ${errBody.length} bytes)`,
            resp.status,
          );
          if (err.isClientError()) throw err;
          lastError = err;
          waitMs = retryAfterMs(resp.headers.get("retry-after")) ?? waitMs;
        } else {
          const json = (await resp.json()) as OpenAIEmbeddingResponse;
          const vectors = this.toVectors(json, texts.length);
          this.health.recordSuccess();
          return vectors;
        }
      } catch (err) {
        if ((err instanceof EmbeddingApiError && err.isClientError()) || err instanceof EmbeddingResponseError) {
          this.health.recordFailure(err instanceof EmbeddingApiError ? err.httpStatus : null);
          throw err;
        }
        // AbortError = timeout, or a network error: retry
        lastError = err instanceof Error ? err : new Error(String(err));
      } finally {
        clearTimeout(timeoutId);
      }
      if (attempt < MAX_RETRIES) {
        await new Promise((r) => setTimeout(r, waitMs));
      }
    }

    const finalError = lastError ?? new Error("Embedding API call failed after retries");
    this.health.recordFailure(finalError instanceof EmbeddingApiError ? finalError.httpStatus : null);
    throw finalError;
  }

  /**
   * Validate a 2xx body and return one vector per input, in input order.
   * Checks, in order: a `data` array, one entry per input, every `index` an
   * integer in range and unique, then every vector's length against the
   * configured dimensions.
   */
  private toVectors(json: OpenAIEmbeddingResponse, expected: number): Float32Array[] {
    if (!json || !Array.isArray(json.data)) {
      throw new EmbeddingResponseError("Embedding API returned unexpected format: missing 'data' array");
    }
    if (json.data.length !== expected) {
      throw new EmbeddingResponseError(
        `Embedding API returned ${json.data.length} embeddings, expected ${expected} embedding(s)`,
      );
    }
    const slots: Array<number[] | undefined> = new Array(expected);
    for (const item of json.data) {
      const idx = item?.index;
      if (!Number.isInteger(idx) || idx < 0 || idx >= expected || slots[idx] !== undefined) {
        throw new EmbeddingResponseError(`Embedding API returned an out-of-range or duplicate index: ${String(idx)}`);
      }
      const emb = item.embedding;
      if (!Array.isArray(emb)) {
        throw new EmbeddingResponseError(`Embedding API returned no embedding array at index ${idx}`);
      }
      if (emb.length !== this.dims) {
        throw new EmbeddingDimensionError(this.dims, emb.length);
      }
      slots[idx] = emb;
    }
    // Sanitize + normalize for consistency with the local provider
    return slots.map((v) => sanitizeAndNormalize(v!));
  }
}

// ============================
// Factory
// ============================

/**
 * Create an EmbeddingService from config.
 *
 * Strategy:
 * - If config has provider != "local" with valid apiKey, model, and dimensions → use remote OpenAI-compatible embedding
 * - If config has provider="local" → use node-llama-cpp local embedding
 * - If config is undefined or missing required fields → fall back to local embedding
 *
 * NOTE: For local providers, `startWarmup()` is NOT called here.
 * The caller is responsible for calling `startWarmup()` at the right time
 * (e.g. on first conversation) to avoid triggering model download during
 * short-lived CLI commands like `gateway stop` or `agents list`.
 */
export function createEmbeddingService(
  config: EmbeddingConfig | undefined,
  logger?: Logger,
): EmbeddingService {
  // Remote OpenAI-compatible provider: any provider value other than "local"
  if (config && config.provider !== "local" && "apiKey" in config && config.apiKey) {
    logger?.debug?.(`${TAG} Using remote embedding (provider=${config.provider}, model=${config.model})`);
    return new OpenAIEmbeddingService(config as OpenAIEmbeddingConfig, logger);
  }

  // Explicit local config
  if (config && config.provider === "local") {
    const localConfig = config as LocalEmbeddingConfig;
    logger?.debug?.(`${TAG} Using local embedding (node-llama-cpp, model=${localConfig.modelPath ?? DEFAULT_LOCAL_MODEL})`);
    return new LocalEmbeddingService(localConfig, logger);
  }

  // Fallback: no config or empty apiKey → use local
  logger?.debug?.(`${TAG} No remote embedding configured, falling back to local embedding (node-llama-cpp)`);
  return new LocalEmbeddingService(undefined, logger);
}

// ============================
// NoopEmbeddingService (for server-side embedding backends)
// ============================

/**
 * No-op embedding service for backends with built-in server-side embedding
 * (e.g., TCVDB with Collection-level embedding config).
 *
 * All embed() calls return an empty Float32Array because the server generates
 * vectors automatically from the text field during upsert/search.
 */
export class NoopEmbeddingService implements EmbeddingService {
  embed(_text: string): Promise<Float32Array> {
    return Promise.resolve(new Float32Array(0));
  }

  embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(texts.map(() => new Float32Array(0)));
  }

  getDimensions(): number {
    return 0;
  }

  getProviderInfo(): EmbeddingProviderInfo {
    return { provider: "noop", model: "server-side" };
  }

  isReady(): boolean {
    return true;
  }

  startWarmup(): void {
    // no-op
  }
}
