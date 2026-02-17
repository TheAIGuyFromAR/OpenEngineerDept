/**
 * Reviewer Agent.
 *
 * Independently evaluates each Coder output against the original task spec.
 * Scores on five dimensions (correctness, style, architecture, robustness,
 * clarity) and produces an overall score with critiques.
 *
 * Also performs convergence analysis across multiple Coder outputs to detect
 * whether generations converge on the same solution, diverge, or form
 * structured clusters.
 *
 * Permissions: read-only access + test execution.
 */

import crypto from "node:crypto";
import type {
  AgentId,
  CoderOutput,
  ConvergenceAnalysis,
  InferenceRequest,
  InferenceResponse,
  ReviewResult,
  ReviewScore,
  TaskSpec,
} from "../types.js";
import { buildLayer0Prompt } from "../memory/layer0-constraints.js";

const REVIEW_SYSTEM_PROMPT = `You are a meticulous code reviewer. Evaluate the implementation against the task specification.

Score each dimension from 0 to 10:
- correctness: Does the code do what was asked? Are edge cases handled?
- style: Does it follow the project's coding standards and constraints?
- architecture: Is the design sound? Are there unnecessary abstractions or missing ones?
- robustness: How well does it handle errors, invalid input, and failure modes?
- clarity: Is the code readable? Are names descriptive? Is the intent obvious?

Respond in JSON format:
{
  "correctness": 0-10,
  "style": 0-10,
  "architecture": 0-10,
  "robustness": 0-10,
  "clarity": 0-10,
  "justification": "2-3 sentence overall assessment",
  "critiques": ["specific issue 1", "specific issue 2"],
  "recommendation": "accept" | "revise" | "reject"
}

Be strict. A score of 7 means "good, production-ready". Above 8 means "excellent".
Below 5 means "significant issues". Be specific in critiques.`;

/**
 * Review a single Coder output against the task spec.
 */
export async function reviewOutput(params: {
  output: CoderOutput;
  task: TaskSpec;
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
}): Promise<ReviewResult> {
  const { output, task, infer } = params;
  const agentId: AgentId = `reviewer-${crypto.randomUUID().slice(0, 8)}`;

  const constraintsPrompt = buildLayer0Prompt(task.constraints);

  const reviewPrompt = [
    constraintsPrompt,
    "",
    "=== TASK SPECIFICATION ===",
    task.instructions,
    `Write scope: ${task.writeScope.join(", ") || "NONE"}`,
    `Difficulty tier: ${task.difficulty.tier} (estimated P: ${task.difficulty.estimatedP})`,
    "=== END TASK ===",
    "",
    "=== IMPLEMENTATION ===",
    output.code || "(no code produced)",
    "=== END IMPLEMENTATION ===",
    "",
    "=== TESTS ===",
    output.tests || "(no tests produced)",
    "=== END TESTS ===",
    "",
    "=== DOCUMENTATION ===",
    output.documentation || "(no documentation produced)",
    "=== END DOCUMENTATION ===",
    "",
    `Coder used temperature=${output.temperature}, system variant: "${output.systemPromptVariant.slice(0, 60)}..."`,
    output.thinkingTrace
      ? `Coder's thinking trace (summary): ${output.thinkingTrace.slice(0, 500)}`
      : "",
  ]
    .filter(Boolean)
    .join("\n");

  const response = await infer({
    systemPrompt: REVIEW_SYSTEM_PROMPT,
    prompt: reviewPrompt,
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 2_000,
    thinkingMode: false,
  });

  const score = parseReviewResponse(response.text);

  return {
    agentId,
    taskId: task.id,
    coderAgentId: output.agentId,
    score,
    recommendation: deriveRecommendation(score),
  };
}

/**
 * Analyze convergence across multiple Coder outputs.
 *
 * Convergent: Most implementations take the same approach → high confidence pick.
 * Divergent: Implementations differ significantly → may need human input.
 * Structured-split: Clear clusters of approaches → pick the best cluster.
 */
export async function analyzeConvergence(params: {
  outputs: CoderOutput[];
  reviews: ReviewResult[];
  task: TaskSpec;
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
}): Promise<ConvergenceAnalysis> {
  const { outputs, reviews, task, infer } = params;

  if (outputs.length <= 1) {
    return {
      pattern: "convergent",
      clusters: [
        {
          approachSummary: "Single generation",
          members: outputs.map((o) => o.agentId),
          averageScore: reviews[0]?.score.overall ?? 0,
        },
      ],
      confidence: reviews[0]?.score.overall ? reviews[0].score.overall / 10 : 0.5,
      recommendation: "Only one generation — accept if score meets threshold.",
    };
  }

  // Build a summary of each output for the LLM to cluster
  const summaries = outputs.map((o, i) => {
    const review = reviews.find((r) => r.coderAgentId === o.agentId);
    return [
      `--- Generation ${i + 1} (${o.agentId}) ---`,
      `Temperature: ${o.temperature}`,
      `Score: ${review?.score.overall ?? "N/A"}/10`,
      `Recommendation: ${review?.recommendation ?? "N/A"}`,
      `Code (first 600 chars): ${o.code.slice(0, 600)}`,
      `Tests: ${o.tests ? "present" : "absent"}`,
    ].join("\n");
  });

  const response = await infer({
    systemPrompt: `You analyze multiple code implementations of the same task.
Identify whether implementations converge on the same approach or diverge.
Group similar implementations into clusters.

Respond in JSON:
{
  "pattern": "convergent" | "divergent" | "structured-split",
  "clusters": [
    {
      "approachSummary": "Brief description of this approach",
      "memberIndices": [0, 2, 3],
      "averageScore": 7.5
    }
  ],
  "confidence": 0.0-1.0,
  "recommendation": "Which cluster/approach to pick and why"
}`,
    prompt: [
      `Task: ${task.instructions}`,
      "",
      "=== IMPLEMENTATIONS ===",
      ...summaries,
      "=== END IMPLEMENTATIONS ===",
      "",
      "Cluster these implementations by approach similarity.",
    ].join("\n"),
    temperature: 0.1,
    topP: 0.9,
    maxTokens: 2_000,
    thinkingMode: false,
  });

  return parseConvergenceResponse(response.text, outputs, reviews);
}

/**
 * Select the best output from reviewed candidates.
 * Strategy: highest-scoring output from the largest convergent cluster.
 */
export function selectBestOutput(
  outputs: CoderOutput[],
  reviews: ReviewResult[],
  convergence: ConvergenceAnalysis,
): CoderOutput {
  // Find the best cluster (largest, then highest average score)
  const bestCluster = [...convergence.clusters].sort((a, b) => {
    // Prefer larger clusters
    if (b.members.length !== a.members.length) {
      return b.members.length - a.members.length;
    }
    // Then higher average score
    return b.averageScore - a.averageScore;
  })[0];

  if (!bestCluster) {
    // Fallback: pick the highest-scored output overall
    return pickHighestScored(outputs, reviews);
  }

  // Pick the highest-scored output within the best cluster
  const clusterOutputs = outputs.filter((o) => bestCluster.members.includes(o.agentId));
  return pickHighestScored(clusterOutputs.length > 0 ? clusterOutputs : outputs, reviews);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pickHighestScored(outputs: CoderOutput[], reviews: ReviewResult[]): CoderOutput {
  let best = outputs[0];
  let bestScore = -1;

  for (const output of outputs) {
    const review = reviews.find((r) => r.coderAgentId === output.agentId);
    const score = review?.score.overall ?? 0;
    if (score > bestScore) {
      bestScore = score;
      best = output;
    }
  }

  return best;
}

function parseReviewResponse(text: string): ReviewScore {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return defaultScore("Failed to parse reviewer response");
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as Partial<ReviewScore> & {
      recommendation?: string;
    };

    const correctness = clampScore(parsed.correctness);
    const style = clampScore(parsed.style);
    const architecture = clampScore(parsed.architecture);
    const robustness = clampScore(parsed.robustness);
    const clarity = clampScore(parsed.clarity);

    // Weighted overall score
    const overall =
      correctness * 0.35 +
      style * 0.1 +
      architecture * 0.25 +
      robustness * 0.2 +
      clarity * 0.1;

    return {
      correctness,
      style,
      architecture,
      robustness,
      clarity,
      overall: Math.round(overall * 10) / 10,
      justification: parsed.justification ?? "",
      critiques: Array.isArray(parsed.critiques) ? parsed.critiques.filter((c) => typeof c === "string") : [],
    };
  } catch {
    return defaultScore("JSON parse error in reviewer response");
  }
}

function clampScore(value: unknown): number {
  if (typeof value !== "number" || isNaN(value)) return 5;
  return Math.max(0, Math.min(10, value));
}

function defaultScore(reason: string): ReviewScore {
  return {
    correctness: 5,
    style: 5,
    architecture: 5,
    robustness: 5,
    clarity: 5,
    overall: 5,
    justification: reason,
    critiques: [reason],
  };
}

function deriveRecommendation(score: ReviewScore): ReviewResult["recommendation"] {
  if (score.overall >= 7.0) return "accept";
  if (score.overall >= 4.0) return "revise";
  return "reject";
}

function parseConvergenceResponse(
  text: string,
  outputs: CoderOutput[],
  reviews: ReviewResult[],
): ConvergenceAnalysis {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return fallbackConvergence(outputs, reviews);
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as {
      pattern?: string;
      clusters?: Array<{
        approachSummary?: string;
        memberIndices?: number[];
        averageScore?: number;
      }>;
      confidence?: number;
      recommendation?: string;
    };

    const pattern = (["convergent", "divergent", "structured-split"].includes(parsed.pattern ?? "")
      ? parsed.pattern
      : "divergent") as ConvergenceAnalysis["pattern"];

    const clusters = (parsed.clusters ?? []).map((c) => ({
      approachSummary: c.approachSummary ?? "Unknown approach",
      members: (c.memberIndices ?? [])
        .filter((i) => i >= 0 && i < outputs.length)
        .map((i) => outputs[i].agentId),
      averageScore: typeof c.averageScore === "number" ? c.averageScore : 5,
    }));

    return {
      pattern,
      clusters: clusters.length > 0 ? clusters : fallbackConvergence(outputs, reviews).clusters,
      confidence: typeof parsed.confidence === "number" ? Math.max(0, Math.min(1, parsed.confidence)) : 0.5,
      recommendation: parsed.recommendation ?? "Select highest-scored output.",
    };
  } catch {
    return fallbackConvergence(outputs, reviews);
  }
}

function fallbackConvergence(
  outputs: CoderOutput[],
  reviews: ReviewResult[],
): ConvergenceAnalysis {
  const avgScore =
    reviews.length > 0
      ? reviews.reduce((sum, r) => sum + r.score.overall, 0) / reviews.length
      : 5;

  return {
    pattern: "divergent",
    clusters: [
      {
        approachSummary: "All outputs (no clustering performed)",
        members: outputs.map((o) => o.agentId),
        averageScore: Math.round(avgScore * 10) / 10,
      },
    ],
    confidence: 0.3,
    recommendation: "Convergence analysis failed — select highest-scored output.",
  };
}
