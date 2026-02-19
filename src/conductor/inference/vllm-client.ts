/**
 * vLLM Inference Client.
 *
 * Concrete HTTP client for vLLM's OpenAI-compatible API.
 * Handles:
 *   - Retries with exponential backoff on transient failures
 *   - Streaming token delivery for long generations
 *   - LoRA adapter routing (per-request model selection)
 *   - Request/response metrics collection
 *   - Timeout enforcement
 *
 * Designed to be the sole dependency-injected `infer` function
 * throughout the Conductor pipeline.
 */

import { randomBytes } from "node:crypto";
import type {
  InferenceRequest,
  InferenceResponse,
  VllmEndpoint,
} from "../types.js";

export type VllmClientConfig = {
  /** Primary vLLM endpoint. */
  endpoint: VllmEndpoint;
  /** Fallback endpoints (tried in order on failure). */
  fallbackEndpoints?: VllmEndpoint[];
  /** Maximum retries per endpoint. */
  maxRetries: number;
  /** Base delay for exponential backoff (ms). */
  baseRetryDelayMs: number;
  /** Request timeout (ms). */
  timeoutMs: number;
  /** Whether to collect per-request metrics. */
  collectMetrics: boolean;
  /** API key for vLLM (if auth is enabled). */
  apiKey?: string;
};

const DEFAULT_CONFIG: VllmClientConfig = {
  endpoint: { baseUrl: "http://localhost:8000", modelId: "default" },
  maxRetries: 3,
  baseRetryDelayMs: 1_000,
  timeoutMs: 120_000,
  collectMetrics: true,
};

export type InferenceMetric = {
  requestId: string;
  modelId: string;
  loraAdapter: string | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  latencyMs: number;
  timeToFirstTokenMs: number | null;
  retries: number;
  endpoint: string;
  temperature: number;
  thinkingMode: boolean;
  finishReason: string;
  ts: number;
};

export type VllmClient = {
  /** Execute a single inference request. */
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
  /** Execute with streaming — calls onToken for each chunk. */
  inferStream: (
    req: InferenceRequest,
    onToken: (chunk: string) => void,
  ) => Promise<InferenceResponse>;
  /** Get collected metrics. */
  getMetrics: () => InferenceMetric[];
  /** Clear collected metrics. */
  clearMetrics: () => void;
  /** Health check the endpoint. */
  healthCheck: () => Promise<boolean>;
};

/**
 * Create a vLLM inference client.
 */
export function createVllmClient(
  config: Partial<VllmClientConfig> & { endpoint: VllmEndpoint },
): VllmClient {
  const cfg: VllmClientConfig = { ...DEFAULT_CONFIG, ...config };
  const metrics: InferenceMetric[] = [];

  return {
    infer: (req) => executeInference(cfg, req, metrics),
    inferStream: (req, onToken) => executeStreamingInference(cfg, req, onToken, metrics),
    getMetrics: () => [...metrics],
    clearMetrics: () => { metrics.length = 0; },
    healthCheck: () => checkHealth(cfg),
  };
}

// ---------------------------------------------------------------------------
// Core inference
// ---------------------------------------------------------------------------

async function executeInference(
  cfg: VllmClientConfig,
  req: InferenceRequest,
  metrics: InferenceMetric[],
): Promise<InferenceResponse> {
  const endpoints = [cfg.endpoint, ...(cfg.fallbackEndpoints ?? [])];
  let lastError: Error | null = null;

  for (const endpoint of endpoints) {
    for (let attempt = 0; attempt <= cfg.maxRetries; attempt++) {
      try {
        const result = await callVllmApi(cfg, endpoint, req);

        if (cfg.collectMetrics) {
          metrics.push(buildMetric(req, result, endpoint, attempt));
        }

        return result;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        // Don't retry on 4xx (client errors)
        if (isClientError(lastError)) throw lastError;

        // Exponential backoff before retry
        if (attempt < cfg.maxRetries) {
          const delay = cfg.baseRetryDelayMs * Math.pow(2, attempt);
          await sleep(delay);
        }
      }
    }
  }

  throw new Error(
    `All inference endpoints exhausted after retries. Last error: ${lastError?.message ?? "unknown"}`,
  );
}

async function executeStreamingInference(
  cfg: VllmClientConfig,
  req: InferenceRequest,
  onToken: (chunk: string) => void,
  metrics: InferenceMetric[],
): Promise<InferenceResponse> {
  const endpoints = [cfg.endpoint, ...(cfg.fallbackEndpoints ?? [])];
  let lastError: Error | null = null;

  for (const endpoint of endpoints) {
    for (let attempt = 0; attempt <= cfg.maxRetries; attempt++) {
      try {
        const result = await callVllmStreamingApi(cfg, endpoint, req, onToken);

        if (cfg.collectMetrics) {
          metrics.push(buildMetric(req, result, endpoint, attempt));
        }

        return result;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        if (isClientError(lastError)) throw lastError;

        if (attempt < cfg.maxRetries) {
          const delay = cfg.baseRetryDelayMs * Math.pow(2, attempt);
          await sleep(delay);
        }
      }
    }
  }

  throw new Error(
    `All streaming endpoints exhausted. Last error: ${lastError?.message ?? "unknown"}`,
  );
}

// ---------------------------------------------------------------------------
// HTTP layer
// ---------------------------------------------------------------------------

async function callVllmApi(
  cfg: VllmClientConfig,
  endpoint: VllmEndpoint,
  req: InferenceRequest,
): Promise<InferenceResponse> {
  const startTime = Date.now();
  const body = buildRequestBody(endpoint, req);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);

  try {
    const response = await fetch(`${endpoint.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: buildHeaders(cfg),
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new HttpError(response.status, `vLLM API error ${response.status}: ${errorText}`);
    }

    const data = (await response.json()) as VllmChatResponse;
    const choice = data.choices?.[0];

    return {
      text: choice?.message?.content ?? "",
      thinkingTrace: extractThinkingTrace(choice?.message?.content ?? ""),
      tokenCount: data.usage?.completion_tokens ?? 0,
      finishReason: mapFinishReason(choice?.finish_reason),
      latencyMs: Date.now() - startTime,
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function callVllmStreamingApi(
  cfg: VllmClientConfig,
  endpoint: VllmEndpoint,
  req: InferenceRequest,
  onToken: (chunk: string) => void,
): Promise<InferenceResponse> {
  const startTime = Date.now();
  let timeToFirstToken: number | null = null;
  const body = { ...buildRequestBody(endpoint, req), stream: true };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);

  try {
    const response = await fetch(`${endpoint.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: buildHeaders(cfg),
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new HttpError(response.status, `vLLM streaming error ${response.status}: ${errorText}`);
    }

    if (!response.body) {
      throw new Error("Response body is null — streaming not supported");
    }

    let fullText = "";
    let finishReason: string = "stop";
    let totalTokens = 0;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;

        try {
          const chunk = JSON.parse(payload) as VllmStreamChunk;
          const delta = chunk.choices?.[0]?.delta?.content ?? "";

          if (delta) {
            if (timeToFirstToken === null) {
              timeToFirstToken = Date.now() - startTime;
            }
            fullText += delta;
            totalTokens++;
            onToken(delta);
          }

          if (chunk.choices?.[0]?.finish_reason) {
            finishReason = chunk.choices[0].finish_reason;
          }
        } catch {
          // Skip malformed chunks
        }
      }
    }

    return {
      text: fullText,
      thinkingTrace: extractThinkingTrace(fullText),
      tokenCount: totalTokens,
      finishReason: mapFinishReason(finishReason),
      latencyMs: Date.now() - startTime,
    };
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

async function checkHealth(cfg: VllmClientConfig): Promise<boolean> {
  try {
    const response = await fetch(`${cfg.endpoint.baseUrl}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(5_000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

function buildRequestBody(
  endpoint: VllmEndpoint,
  req: InferenceRequest,
): Record<string, unknown> {
  const messages = [
    { role: "system", content: req.systemPrompt },
    { role: "user", content: req.prompt },
  ];

  const body: Record<string, unknown> = {
    model: endpoint.loraAdapter ?? endpoint.modelId,
    messages,
    temperature: req.temperature,
    top_p: req.topP,
    max_tokens: req.maxTokens,
  };

  if (req.stopSequences && req.stopSequences.length > 0) {
    body.stop = req.stopSequences;
  }

  // vLLM extended thinking support (model-dependent)
  if (req.thinkingMode) {
    body.extra_body = {
      enable_thinking: true,
    };
  }

  return body;
}

function buildHeaders(cfg: VllmClientConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (cfg.apiKey) {
    headers["Authorization"] = `Bearer ${cfg.apiKey}`;
  }
  return headers;
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

type VllmChatResponse = {
  choices?: Array<{
    message?: { content?: string; role?: string };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

type VllmStreamChunk = {
  choices?: Array<{
    delta?: { content?: string };
    finish_reason?: string | null;
  }>;
};

function extractThinkingTrace(text: string): string | undefined {
  // vLLM/Qwen thinking blocks: <think>...</think>
  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
  return thinkMatch ? thinkMatch[1].trim() : undefined;
}

function mapFinishReason(reason?: string | null): InferenceResponse["finishReason"] {
  if (!reason) return "stop";
  if (reason === "length") return "length";
  if (reason === "stop") return "stop";
  return "error";
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function buildMetric(
  req: InferenceRequest,
  result: InferenceResponse,
  endpoint: VllmEndpoint,
  retries: number,
): InferenceMetric {
  return {
    requestId: `req-${Date.now()}-${randomBytes(4).toString("hex")}`,
    modelId: endpoint.modelId,
    loraAdapter: endpoint.loraAdapter ?? null,
    promptTokens: 0, // Not available from non-usage response
    completionTokens: result.tokenCount,
    totalTokens: result.tokenCount,
    latencyMs: result.latencyMs,
    timeToFirstTokenMs: null,
    retries,
    endpoint: endpoint.baseUrl,
    temperature: req.temperature,
    thinkingMode: req.thinkingMode,
    finishReason: result.finishReason,
    ts: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

class HttpError extends Error {
  constructor(
    public readonly statusCode: number,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

function isClientError(err: Error): boolean {
  return err instanceof HttpError && err.statusCode >= 400 && err.statusCode < 500;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
