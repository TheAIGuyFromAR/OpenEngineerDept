/**
 * Layer 1: Working Memory.
 *
 * Full-fidelity recent conversation. The last N turns of interaction
 * between human and Conductor, plus the last M completed task specs
 * and results. No compression.
 *
 * Typical size: 8,000–32,000 tokens.
 */

import type { AgentId, WorkingMemoryTurn } from "../types.js";

export type Layer1Config = {
  /** Maximum number of turns to keep in working memory. */
  maxTurns: number;
  /** Maximum total token estimate before triggering L1→L2 flow. */
  maxTokenEstimate: number;
};

const DEFAULT_CONFIG: Layer1Config = {
  maxTurns: 100,
  maxTokenEstimate: 32_000,
};

// Rough token estimate: ~4 chars per token for English text.
const CHARS_PER_TOKEN = 4;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function createWorkingMemory(config?: Partial<Layer1Config>): Layer1State {
  return {
    config: { ...DEFAULT_CONFIG, ...config },
    turns: [],
  };
}

export type Layer1State = {
  config: Layer1Config;
  turns: WorkingMemoryTurn[];
};

export function addTurn(
  state: Layer1State,
  role: WorkingMemoryTurn["role"],
  content: string,
  agentId?: AgentId,
): { state: Layer1State; overflow: WorkingMemoryTurn[] } {
  const turn: WorkingMemoryTurn = {
    role,
    content,
    agentId,
    ts: Date.now(),
  };

  const nextTurns = [...state.turns, turn];
  const overflow: WorkingMemoryTurn[] = [];

  // Evict oldest turns if we exceed limits
  while (
    nextTurns.length > state.config.maxTurns ||
    estimateTotalTokens(nextTurns) > state.config.maxTokenEstimate
  ) {
    const evicted = nextTurns.shift();
    if (evicted) {
      overflow.push(evicted);
    }
  }

  return {
    state: { ...state, turns: nextTurns },
    overflow,
  };
}

export function estimateTotalTokens(turns: WorkingMemoryTurn[]): number {
  let total = 0;
  for (const turn of turns) {
    total += estimateTokens(turn.content);
    // Add overhead for role/metadata markers
    total += 10;
  }
  return total;
}

export function getRecentTurns(
  state: Layer1State,
  count: number,
): WorkingMemoryTurn[] {
  return state.turns.slice(-count);
}

/**
 * Build the prompt fragment for Layer 1 inclusion.
 */
export function buildLayer1Prompt(state: Layer1State): string {
  if (state.turns.length === 0) {
    return "";
  }

  const lines = ["=== WORKING MEMORY (Layer 1) ===", ""];

  for (const turn of state.turns) {
    const roleLabel =
      turn.role === "human"
        ? "HUMAN"
        : turn.role === "conductor"
          ? "CONDUCTOR"
          : `AGENT[${turn.agentId ?? "unknown"}]`;

    lines.push(`[${roleLabel}] ${turn.content}`);
    lines.push("");
  }

  lines.push("=== END WORKING MEMORY ===");
  return lines.join("\n");
}

/**
 * Extract key decisions and outcomes from turns (used by L2 compaction).
 */
export function extractDecisionsAndOutcomes(
  turns: WorkingMemoryTurn[],
): { decisions: string[]; outcomes: string[] } {
  const decisions: string[] = [];
  const outcomes: string[] = [];

  for (const turn of turns) {
    const content = turn.content.toLowerCase();

    // Heuristic: conductor messages with decision-like language
    if (turn.role === "conductor") {
      if (
        content.includes("decided to") ||
        content.includes("will use") ||
        content.includes("approach:") ||
        content.includes("chosen") ||
        content.includes("strategy:")
      ) {
        decisions.push(turn.content);
      }
    }

    // Heuristic: agent messages with outcome-like language
    if (turn.role === "agent") {
      if (
        content.includes("completed") ||
        content.includes("tests pass") ||
        content.includes("implemented") ||
        content.includes("failed:") ||
        content.includes("error:")
      ) {
        outcomes.push(turn.content);
      }
    }
  }

  return { decisions, outcomes };
}
