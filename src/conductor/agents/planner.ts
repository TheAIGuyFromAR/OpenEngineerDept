/**
 * Planner Agent.
 *
 * Estimates task difficulty and assigns compute tiers using
 * Pareto-optimal allocation. Uses historical success rates from
 * Layer 3 (changelog) to calibrate estimates.
 *
 * The core formula: P(success | N) = 1 - (1-p)^N
 * where p = single-generation success probability, N = generations.
 *
 * Four tiers:
 *   T1: Trivial — 1 generation, no thinking mode
 *   T2: Standard — 3 generations, majority voting
 *   T3: Complex — 5 generations with thinking mode, iterative refinement
 *   T4: Research — 10 generations, full ensemble, escalation path
 *
 * Permissions: read-only. Cannot write or execute.
 */

import crypto from "node:crypto";
import type {
  AgentId,
  ComputeTier,
  DifficultyEstimate,
  InferenceRequest,
  InferenceResponse,
  TaskSpec,
  TIER_CONFIGS,
  UltraThinkConfig,
} from "../types.js";
import { TIER_CONFIGS as tierConfigs } from "../types.js";
import { type MemoryStackState } from "../memory/stack.js";
import { getHistoricalSuccessRate } from "../memory/layer3-changelog.js";

export type PlannerConfig = {
  modelEndpoint: string;
  modelId: string;
  /** Override the default success probability thresholds. */
  tierThresholds?: {
    t1Min: number;  // Default: 0.85 — trivial tasks
    t2Min: number;  // Default: 0.60 — standard tasks
    t3Min: number;  // Default: 0.35 — complex tasks
    // Below t3Min → T4 (research grade)
  };
  /** Target success probability for the ensemble. */
  targetSuccessP?: number;  // Default: 0.95
};

const DEFAULT_THRESHOLDS = {
  t1Min: 0.85,
  t2Min: 0.60,
  t3Min: 0.35,
};

const PLANNER_SYSTEM_PROMPT = `You are a task difficulty estimator. Given a software engineering task, estimate:

1. The difficulty tier (1-4):
   - Tier 1: Trivial — simple rename, typo fix, config change. High confidence a single generation succeeds.
   - Tier 2: Standard — adding a function, fixing a bug with clear reproduction. Moderate complexity.
   - Tier 3: Complex — cross-module refactor, new feature with edge cases, performance optimization.
   - Tier 4: Research — novel algorithm, architectural redesign, concurrency bugs, security-critical changes.

2. The estimated single-generation success probability (0.0-1.0):
   - 0.9+ : Very likely to succeed on first try
   - 0.6-0.9 : Likely but may need iteration
   - 0.3-0.6 : Uncertain, benefits from multiple attempts
   - 0.1-0.3 : Difficult, needs ensemble approach
   - <0.1 : Research-grade, may require human intervention

3. Which modules/files are affected.

Respond in JSON:
{
  "tier": 1-4,
  "estimatedP": 0.0-1.0,
  "reasoning": "2-3 sentences explaining the estimate",
  "modulesAffected": ["path/to/module1", "path/to/module2"],
  "riskFactors": ["description of risk 1", "description of risk 2"]
}

Be calibrated. Most bug fixes are T2. Most new features are T2-T3. Security changes and architectural rewrites are T3-T4.`;

/**
 * Estimate task difficulty using LLM + historical data.
 */
export async function estimateDifficulty(params: {
  taskDescription: string;
  contextFiles: string[];
  memory: MemoryStackState;
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
  config?: PlannerConfig;
}): Promise<DifficultyEstimate> {
  const { taskDescription, contextFiles, memory, infer } = params;
  const agentId: AgentId = `planner-${crypto.randomUUID().slice(0, 8)}`;

  // Check historical success rate for similar tasks
  const historicalRate = getHistoricalSuccessRate(memory.layer3, taskDescription);

  const prompt = [
    `Task: ${taskDescription}`,
    "",
    `Files involved: ${contextFiles.length > 0 ? contextFiles.join(", ") : "Unknown"}`,
    historicalRate !== null
      ? `Historical success rate for similar tasks: ${(historicalRate * 100).toFixed(0)}%`
      : "No historical data for similar tasks.",
    "",
    "Estimate the difficulty tier and single-generation success probability.",
  ].join("\n");

  const response = await infer({
    systemPrompt: PLANNER_SYSTEM_PROMPT,
    prompt,
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 1_000,
    thinkingMode: false,
  });

  const estimate = parseDifficultyResponse(response.text, historicalRate);

  // Calibrate against historical data if available
  return calibrateEstimate(estimate, historicalRate, params.config);
}

/**
 * Compute the optimal number of generations for a given tier.
 *
 * Uses P(success | N) = 1 - (1-p)^N to find N such that
 * P(success | N) >= target (default 0.95).
 */
export function computeOptimalN(
  estimatedP: number,
  target: number = 0.95,
  maxN: number = 20,
): number {
  if (estimatedP <= 0) return maxN;
  if (estimatedP >= 1) return 1;

  // Solve: 1 - (1-p)^N >= target
  // N >= log(1-target) / log(1-p)
  const n = Math.ceil(Math.log(1 - target) / Math.log(1 - estimatedP));
  return Math.max(1, Math.min(n, maxN));
}

/**
 * Get the UltraThinkConfig for a given difficulty estimate.
 * Adjusts the default tier config based on the specific probability estimate.
 */
export function getComputeConfig(
  estimate: DifficultyEstimate,
  config?: PlannerConfig,
): UltraThinkConfig {
  const base = tierConfigs[estimate.tier];
  const target = config?.targetSuccessP ?? 0.95;

  // Compute optimal N, but cap at tier's default or 2x default
  const optimalN = computeOptimalN(estimate.estimatedP, target);
  const cappedN = Math.min(optimalN, base.n * 2);

  return {
    ...base,
    n: Math.max(base.n, cappedN),
  };
}

/**
 * Assign a tier based on the estimated single-generation success probability.
 */
export function assignTier(estimatedP: number, config?: PlannerConfig): ComputeTier {
  const thresholds = config?.tierThresholds ?? DEFAULT_THRESHOLDS;

  if (estimatedP >= thresholds.t1Min) return 1;
  if (estimatedP >= thresholds.t2Min) return 2;
  if (estimatedP >= thresholds.t3Min) return 3;
  return 4;
}

/**
 * Compute expected cost (in generations) for achieving the target success rate.
 * Used for Pareto-optimal budget allocation across multiple tasks.
 */
export function expectedCost(
  estimatedP: number,
  target: number = 0.95,
): number {
  const n = computeOptimalN(estimatedP, target);
  // Expected generations consumed = N (worst case) or
  // geometric expectation = 1/p (expected tries until first success)
  // Use the worse of the two for budgeting
  return Math.max(n, Math.ceil(1 / Math.max(estimatedP, 0.01)));
}

/**
 * Allocate compute budget across multiple tasks using Pareto optimality.
 * Given a total generation budget, distribute generations to maximize
 * the expected number of tasks completed.
 */
export function allocateBudget(
  tasks: Array<{ id: string; estimate: DifficultyEstimate }>,
  totalBudget: number,
  config?: PlannerConfig,
): Map<string, UltraThinkConfig> {
  const target = config?.targetSuccessP ?? 0.95;
  const allocations = new Map<string, UltraThinkConfig>();

  // Sort tasks by expected marginal value (bang per buck)
  // Marginal value = P(success | N) / N
  const scored = tasks.map((t) => {
    const minN = computeOptimalN(t.estimate.estimatedP, target);
    const tierConfig = tierConfigs[t.estimate.tier];
    return {
      ...t,
      minN: Math.max(1, Math.min(minN, tierConfig.n * 2)),
      tierConfig,
      marginalValue: t.estimate.estimatedP / Math.max(1, minN),
    };
  });

  // Greedy allocation: give each task its minimum, then redistribute surplus
  let remaining = totalBudget;

  // Phase 1: Ensure every task gets at least 1 generation
  for (const task of scored) {
    const allocated = Math.min(1, remaining);
    allocations.set(task.id, {
      ...task.tierConfig,
      n: allocated,
    });
    remaining -= allocated;
  }

  if (remaining <= 0) return allocations;

  // Phase 2: Allocate up to each task's optimal N, sorted by marginal value
  scored.sort((a, b) => b.marginalValue - a.marginalValue);

  for (const task of scored) {
    const current = allocations.get(task.id)!;
    const needed = task.minN - current.n;
    if (needed > 0 && remaining > 0) {
      const extra = Math.min(needed, remaining);
      allocations.set(task.id, { ...current, n: current.n + extra });
      remaining -= extra;
    }
  }

  return allocations;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseDifficultyResponse(
  text: string,
  historicalRate: number | null,
): DifficultyEstimate {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return defaultEstimate("Failed to parse planner response", historicalRate);
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as {
      tier?: number;
      estimatedP?: number;
      reasoning?: string;
      modulesAffected?: string[];
    };

    const tier = clampTier(parsed.tier ?? 2);
    const estimatedP = Math.max(0.01, Math.min(0.99, parsed.estimatedP ?? 0.5));

    return {
      tier,
      estimatedP,
      reasoning: parsed.reasoning ?? "Estimated by Planner",
      modulesAffected: Array.isArray(parsed.modulesAffected) ? parsed.modulesAffected : [],
      historicalSuccessRate: historicalRate,
    };
  } catch {
    return defaultEstimate("JSON parse error in planner response", historicalRate);
  }
}

function defaultEstimate(reason: string, historicalRate: number | null): DifficultyEstimate {
  return {
    tier: 2,
    estimatedP: 0.5,
    reasoning: reason,
    modulesAffected: [],
    historicalSuccessRate: historicalRate,
  };
}

function clampTier(raw: number): ComputeTier {
  if (raw <= 1) return 1;
  if (raw >= 4) return 4;
  return Math.round(raw) as ComputeTier;
}

/**
 * Calibrate LLM estimate against historical data.
 * If historical data is available, blend the LLM estimate with the observed rate.
 */
function calibrateEstimate(
  estimate: DifficultyEstimate,
  historicalRate: number | null,
  config?: PlannerConfig,
): DifficultyEstimate {
  if (historicalRate === null) return estimate;

  // Blend: 60% LLM estimate, 40% historical (historical is ground truth)
  const blendedP = estimate.estimatedP * 0.6 + historicalRate * 0.4;
  const calibratedTier = assignTier(blendedP, config);

  return {
    ...estimate,
    estimatedP: Math.round(blendedP * 100) / 100,
    tier: calibratedTier,
    historicalSuccessRate: historicalRate,
    reasoning: `${estimate.reasoning} [Calibrated: LLM=${estimate.estimatedP.toFixed(2)}, historical=${historicalRate.toFixed(2)}, blended=${blendedP.toFixed(2)}]`,
  };
}
