/**
 * Layer 2: Compressed History.
 *
 * Rolling compaction of older conversation turns. An asynchronous background
 * process identifies the oldest material in Layer 1, extracts decisions,
 * outcomes, and reasoning, discards redundant discussion, and appends
 * compressed summaries.
 *
 * The human never experiences a compaction cliff. Material flows from
 * Layer 1 to Layer 2 continuously, not in a single catastrophic event.
 *
 * Typical size: 4,000–16,000 tokens.
 */

import type { CompressedEntry, WorkingMemoryTurn, InferenceRequest } from "../types.js";
import { extractDecisionsAndOutcomes } from "./layer1-working.js";

export type Layer2Config = {
  /** Maximum number of compressed entries to retain. */
  maxEntries: number;
  /** Maximum token estimate for the compressed layer. */
  maxTokenEstimate: number;
  /** Minimum turns to accumulate before compressing a batch. */
  minBatchSize: number;
};

const DEFAULT_CONFIG: Layer2Config = {
  maxEntries: 200,
  maxTokenEstimate: 16_000,
  minBatchSize: 5,
};

const CHARS_PER_TOKEN = 4;

export type Layer2State = {
  config: Layer2Config;
  entries: CompressedEntry[];
  /** Turns waiting to be compressed (accumulated from L1 overflow). */
  pendingTurns: WorkingMemoryTurn[];
  /** Counter for turn numbering across the entire session. */
  globalTurnCounter: number;
};

export function createCompressedHistory(config?: Partial<Layer2Config>): Layer2State {
  return {
    config: { ...DEFAULT_CONFIG, ...config },
    entries: [],
    pendingTurns: [],
    globalTurnCounter: 0,
  };
}

/**
 * Receive overflow turns from Layer 1 and queue them for compression.
 */
export function receiveTurns(
  state: Layer2State,
  turns: WorkingMemoryTurn[],
): Layer2State {
  return {
    ...state,
    pendingTurns: [...state.pendingTurns, ...turns],
    globalTurnCounter: state.globalTurnCounter + turns.length,
  };
}

/**
 * Check if there are enough pending turns to warrant a compression pass.
 */
export function needsCompaction(state: Layer2State): boolean {
  return state.pendingTurns.length >= state.config.minBatchSize;
}

/**
 * Build the inference request for the compression model.
 * The actual inference call is handled by the caller.
 */
export function buildCompressionPrompt(
  turns: WorkingMemoryTurn[],
): InferenceRequest {
  const turnText = turns
    .map((t) => {
      const role = t.role === "human" ? "Human" : t.role === "conductor" ? "Conductor" : "Agent";
      return `[${role}]: ${t.content}`;
    })
    .join("\n\n");

  return {
    prompt: `Compress the following conversation segment into a concise summary.

RULES:
1. Preserve ALL decisions made and their reasoning
2. Preserve ALL outcomes (success/failure) and their causes
3. Preserve ALL constraints or requirements discovered
4. Discard greetings, acknowledgments, and redundant discussion
5. Use bullet points for decisions and outcomes
6. Keep the summary under 200 words

CONVERSATION:
${turnText}

COMPRESSED SUMMARY:`,
    systemPrompt: "You are a precise summarizer. Extract decisions, outcomes, and reasoning. Discard noise.",
    temperature: 0.1,
    topP: 0.9,
    maxTokens: 500,
    thinkingMode: false,
  };
}

/**
 * Perform local compression without LLM (fallback when inference is unavailable).
 * Extracts decisions and outcomes using heuristics.
 */
export function compressLocally(
  state: Layer2State,
): Layer2State {
  if (state.pendingTurns.length < state.config.minBatchSize) {
    return state;
  }

  const batch = state.pendingTurns.slice(0, state.config.minBatchSize);
  const remaining = state.pendingTurns.slice(state.config.minBatchSize);

  const { decisions, outcomes } = extractDecisionsAndOutcomes(batch);
  const startTurn = state.globalTurnCounter - state.pendingTurns.length;

  const entry: CompressedEntry = {
    originalTurnRange: [startTurn, startTurn + batch.length - 1],
    summary: buildFallbackSummary(batch),
    decisions,
    outcomes,
    compressedAt: Date.now(),
  };

  let entries = [...state.entries, entry];

  // Evict oldest entries if we exceed limits
  while (entries.length > state.config.maxEntries) {
    entries = entries.slice(1);
  }

  return {
    ...state,
    entries,
    pendingTurns: remaining,
  };
}

/**
 * Apply LLM-generated compression result.
 */
export function applyCompression(
  state: Layer2State,
  summary: string,
  batchSize: number,
): Layer2State {
  const batch = state.pendingTurns.slice(0, batchSize);
  const remaining = state.pendingTurns.slice(batchSize);

  const { decisions, outcomes } = extractDecisionsAndOutcomes(batch);
  const startTurn = state.globalTurnCounter - state.pendingTurns.length;

  const entry: CompressedEntry = {
    originalTurnRange: [startTurn, startTurn + batch.length - 1],
    summary,
    decisions,
    outcomes,
    compressedAt: Date.now(),
  };

  let entries = [...state.entries, entry];

  while (entries.length > state.config.maxEntries) {
    entries = entries.slice(1);
  }

  return {
    ...state,
    entries,
    pendingTurns: remaining,
  };
}

function buildFallbackSummary(turns: WorkingMemoryTurn[]): string {
  const parts: string[] = [];
  for (const turn of turns) {
    // Take first 100 chars of each turn for a rough summary
    const snippet = turn.content.slice(0, 100).replace(/\n/g, " ");
    parts.push(`[${turn.role}] ${snippet}${turn.content.length > 100 ? "..." : ""}`);
  }
  return parts.join(" | ");
}

/**
 * Build the prompt fragment for Layer 2 inclusion.
 */
export function buildLayer2Prompt(state: Layer2State): string {
  if (state.entries.length === 0) {
    return "";
  }

  const lines = ["=== COMPRESSED HISTORY (Layer 2) ===", ""];

  for (const entry of state.entries) {
    lines.push(`[Turns ${entry.originalTurnRange[0]}-${entry.originalTurnRange[1]}]`);
    lines.push(entry.summary);
    if (entry.decisions.length > 0) {
      lines.push("Decisions:");
      for (const d of entry.decisions) {
        lines.push(`  - ${d.slice(0, 200)}`);
      }
    }
    if (entry.outcomes.length > 0) {
      lines.push("Outcomes:");
      for (const o of entry.outcomes) {
        lines.push(`  - ${o.slice(0, 200)}`);
      }
    }
    lines.push("");
  }

  lines.push("=== END COMPRESSED HISTORY ===");
  return lines.join("\n");
}

export function estimateLayer2Tokens(state: Layer2State): number {
  const prompt = buildLayer2Prompt(state);
  return Math.ceil(prompt.length / CHARS_PER_TOKEN);
}
