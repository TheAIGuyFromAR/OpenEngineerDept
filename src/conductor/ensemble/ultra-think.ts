/**
 * Ultra Think — Ensemble Inference Pipeline.
 *
 * The core verification pipeline that makes the Conductor reliable.
 * For each task, fires N parallel Coder generations with varied parameters,
 * reviews each independently, analyzes convergence, and selects the best.
 *
 * Pipeline per round:
 *   1. N Coder generations (parallel, varied temperature + system prompt)
 *   2. N test executions (parallel)
 *   3. N Reviewer evaluations (parallel)
 *   4. Convergence analysis
 *   5. Selection or retry
 *
 * Supports iterative refinement: if no output meets the acceptance threshold,
 * feed critiques back and retry (up to maxRetryRounds).
 */

import type {
  CoderOutput,
  ConvergenceAnalysis,
  InferenceRequest,
  InferenceResponse,
  ReviewResult,
  TaskSpec,
  TestResult,
  UltraThinkConfig,
  UltraThinkResult,
} from "../types.js";
import { TIER_CONFIGS } from "../types.js";
import {
  type CoderConfig,
  executeCoderGeneration,
  runTests,
} from "../agents/coder.js";
import {
  analyzeConvergence,
  reviewOutput,
  selectBestOutput,
} from "../agents/reviewer.js";
import {
  createPermissionGrant,
  type PermissionGrant,
} from "../security/trust-boundary.js";
import type { AgentId } from "../types.js";

export type UltraThinkDeps = {
  /** LLM inference function. */
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
  /** Command execution function. */
  exec: (command: string, cwd: string) => Promise<{ stdout: string; stderr: string; exitCode: number }>;
  /** Coder configuration. */
  coderConfig: CoderConfig;
  /** The Conductor's agent ID (for creating permission grants). */
  conductorId: AgentId;
};

/**
 * Execute the full Ultra Think pipeline for a task.
 */
export async function executeUltraThink(
  task: TaskSpec,
  config: UltraThinkConfig,
  deps: UltraThinkDeps,
): Promise<UltraThinkResult> {
  const startTime = Date.now();
  const allOutputs: CoderOutput[] = [];
  const allTests: TestResult[] = [];
  const allReviews: ReviewResult[] = [];

  let bestOutput: CoderOutput | null = null;
  let bestTest: TestResult | null = null;
  let bestReview: ReviewResult | null = null;
  let convergence: ConvergenceAnalysis | null = null;
  let escalated = false;
  let escalationReason: string | undefined;

  for (let round = 0; round < config.maxRetryRounds; round++) {
    // Build critique context from previous rounds (for iterative refinement)
    const critiqueContext = buildCritiqueContext(allReviews, allTests);
    const roundTask = critiqueContext
      ? augmentTaskWithCritiques(task, critiqueContext)
      : task;

    // Step 1: Fire N parallel Coder generations
    const generations = await fireGenerations(roundTask, config, deps, round);
    allOutputs.push(...generations);

    // Step 2: Run tests on all generations (parallel)
    const testResults = await runAllTests(generations, roundTask, deps);
    allTests.push(...testResults);

    // Step 3: Filter to passing tests, then review (parallel)
    const passingOutputs = generations.filter((_, i) => testResults[i].passed);
    const outputsToReview = passingOutputs.length > 0 ? passingOutputs : generations;

    const reviews = await reviewAll(outputsToReview, roundTask, deps);
    allReviews.push(...reviews);

    // Step 4: Convergence analysis
    convergence = await analyzeConvergence({
      outputs: outputsToReview,
      reviews,
      task: roundTask,
      infer: deps.infer,
    });

    // Step 5: Select best and check threshold
    bestOutput = selectBestOutput(outputsToReview, reviews, convergence);
    bestReview = reviews.find((r) => r.coderAgentId === bestOutput!.agentId) ?? reviews[0];
    bestTest = testResults.find((t) => t.agentId === bestOutput!.agentId) ?? testResults[0];

    // Check if we meet the acceptance threshold
    if (bestReview && bestReview.score.overall >= config.acceptThreshold) {
      break;
    }

    // If this is the last round and we still don't meet threshold, escalate
    if (round === config.maxRetryRounds - 1) {
      escalated = true;
      escalationReason = buildEscalationReason(bestReview, config, allReviews);
    }
  }

  return {
    taskId: task.id,
    tier: config.tier,
    rounds: Math.min(config.maxRetryRounds, allOutputs.length > 0 ? Math.ceil(allOutputs.length / config.n) : 1),
    totalGenerations: allOutputs.length,
    selectedOutput: bestOutput!,
    testResult: bestTest!,
    reviewResult: bestReview!,
    convergence: convergence!,
    allOutputs,
    allTests,
    allReviews,
    wallClockMs: Date.now() - startTime,
    escalated,
    escalationReason,
  };
}

/**
 * Execute Ultra Think at a specific tier using default tier configs.
 */
export async function executeAtTier(
  task: TaskSpec,
  deps: UltraThinkDeps,
): Promise<UltraThinkResult> {
  const config = TIER_CONFIGS[task.difficulty.tier];
  return executeUltraThink(task, config, deps);
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

async function fireGenerations(
  task: TaskSpec,
  config: UltraThinkConfig,
  deps: UltraThinkDeps,
  round: number,
): Promise<CoderOutput[]> {
  const temperatures = distributeTemperatures(
    config.n,
    config.temperatureRange[0],
    config.temperatureRange[1],
  );

  // Create permission grants for each coder
  const grants: PermissionGrant[] = [];
  for (let i = 0; i < config.n; i++) {
    grants.push(
      createPermissionGrant({
        conductorId: deps.conductorId,
        targetAgentId: `coder-gen${round}-${i}`,
        targetRole: "coder",
        taskId: task.id,
        customScope: {
          readPaths: task.contextFiles,
          writePaths: task.writeScope,
          allowedCommands: task.allowedCommands,
        },
      }),
    );
  }

  // Fire all generations in parallel
  const promises = temperatures.map((temp, i) =>
    executeCoderGeneration({
      task,
      config: deps.coderConfig,
      grant: grants[i],
      temperature: temp,
      systemPromptIndex: i + round * config.n,
      infer: deps.infer,
    }),
  );

  const results = await Promise.allSettled(promises);

  return results
    .filter((r): r is PromiseFulfilledResult<CoderOutput> => r.status === "fulfilled")
    .map((r) => r.value);
}

async function runAllTests(
  outputs: CoderOutput[],
  task: TaskSpec,
  deps: UltraThinkDeps,
): Promise<TestResult[]> {
  const grants = outputs.map((output) =>
    createPermissionGrant({
      conductorId: deps.conductorId,
      targetAgentId: output.agentId,
      targetRole: "coder",
      taskId: task.id,
      customScope: {
        readPaths: task.contextFiles,
        writePaths: task.writeScope,
        allowedCommands: task.allowedCommands,
        canExecute: true,
      },
    }),
  );

  const promises = outputs.map((output, i) =>
    runTests({
      output,
      task,
      config: deps.coderConfig,
      grant: grants[i],
      exec: deps.exec,
    }),
  );

  const results = await Promise.allSettled(promises);
  return results.map((r, i) =>
    r.status === "fulfilled"
      ? r.value
      : {
          agentId: outputs[i].agentId,
          taskId: task.id,
          passed: false,
          totalTests: 0,
          passedTests: 0,
          failedTests: 1,
          failures: [{ testName: "execution", error: String((r as PromiseRejectedResult).reason) }],
          executionTimeMs: 0,
        },
  );
}

async function reviewAll(
  outputs: CoderOutput[],
  task: TaskSpec,
  deps: UltraThinkDeps,
): Promise<ReviewResult[]> {
  const promises = outputs.map((output) =>
    reviewOutput({
      output,
      task,
      infer: deps.infer,
    }),
  );

  const results = await Promise.allSettled(promises);
  return results
    .filter((r): r is PromiseFulfilledResult<ReviewResult> => r.status === "fulfilled")
    .map((r) => r.value);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Distribute temperatures evenly across the range for diversity.
 */
function distributeTemperatures(n: number, low: number, high: number): number[] {
  if (n === 1) return [(low + high) / 2];
  return Array.from({ length: n }, (_, i) => {
    const t = low + (high - low) * (i / (n - 1));
    return Math.round(t * 100) / 100;
  });
}

/**
 * Build critique context from previous rounds for iterative refinement.
 */
function buildCritiqueContext(
  reviews: ReviewResult[],
  tests: TestResult[],
): string | null {
  if (reviews.length === 0 && tests.length === 0) return null;

  const lines: string[] = [];

  // Aggregate test failures
  const failedTests = tests.filter((t) => !t.passed);
  if (failedTests.length > 0) {
    lines.push("=== PREVIOUS TEST FAILURES ===");
    for (const t of failedTests.slice(-3)) {
      for (const f of t.failures) {
        lines.push(`- ${f.testName}: ${f.error.slice(0, 200)}`);
      }
    }
    lines.push("");
  }

  // Aggregate critiques
  const critiques = reviews.flatMap((r) => r.score.critiques);
  if (critiques.length > 0) {
    const unique = [...new Set(critiques)];
    lines.push("=== PREVIOUS REVIEWER CRITIQUES ===");
    for (const c of unique.slice(0, 10)) {
      lines.push(`- ${c}`);
    }
    lines.push("");
  }

  return lines.length > 0 ? lines.join("\n") : null;
}

/**
 * Augment a task spec with critique feedback for retry rounds.
 */
function augmentTaskWithCritiques(task: TaskSpec, critiques: string): TaskSpec {
  return {
    ...task,
    instructions: `${task.instructions}\n\n=== FEEDBACK FROM PREVIOUS ATTEMPT ===\nThe previous implementation had issues. Address these before generating:\n${critiques}\n=== END FEEDBACK ===`,
  };
}

/**
 * Build a human-readable escalation reason.
 */
function buildEscalationReason(
  bestReview: ReviewResult | null,
  config: UltraThinkConfig,
  allReviews: ReviewResult[],
): string {
  const bestScore = bestReview?.score.overall ?? 0;
  const avgScore =
    allReviews.length > 0
      ? allReviews.reduce((sum, r) => sum + r.score.overall, 0) / allReviews.length
      : 0;

  return [
    `Failed to meet acceptance threshold (${config.acceptThreshold}) after ${config.maxRetryRounds} rounds.`,
    `Best score: ${bestScore.toFixed(1)}/10, average: ${avgScore.toFixed(1)}/10.`,
    bestReview
      ? `Top critiques: ${bestReview.score.critiques.slice(0, 3).join("; ")}`
      : "No reviews completed.",
  ].join(" ");
}
