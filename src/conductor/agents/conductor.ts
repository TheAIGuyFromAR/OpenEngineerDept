/**
 * Conductor.
 *
 * The orchestration brain. Receives structured requests from the Interface
 * Agent. Maintains the memory stack across sessions. Decomposes tasks into
 * sub-agent work orders. Spawns sub-agents with scoped permissions and
 * well-constructed prompts.
 *
 * The Conductor NEVER writes code, NEVER runs commands, NEVER touches files.
 * It plans, delegates, and integrates.
 */

import crypto from "node:crypto";
import type {
  AgentId,
  ChangelogEntry,
  ComputeTier,
  DifficultyEstimate,
  Exemplar,
  HandoffMessage,
  InferenceRequest,
  InferenceResponse,
  TaskSpec,
  UltraThinkResult,
} from "../types.js";
import {
  buildConductorPrompt,
  buildCoderPrompt,
  queryHistoricalSuccess,
  queryKnowledge,
  recordTask,
  recordTurn,
  type MemoryStackState,
} from "../memory/stack.js";
import {
  createPermissionGrant,
  validateHandoff,
  validateTaskSpec,
  type PermissionGrant,
} from "../security/trust-boundary.js";

export type ConductorConfig = {
  /** vLLM endpoint for the orchestrator model. */
  modelEndpoint: string;
  modelId: string;
  /** Maximum concurrent sub-agents. */
  maxConcurrentAgents: number;
  /** Base directory for the project. */
  projectRoot: string;
  /** Best-completion exemplar library. */
  exemplarLibrary: Exemplar[];
};

export type ConductorState = {
  config: ConductorConfig;
  memory: MemoryStackState;
  agentId: AgentId;
  /** Active permission grants. */
  activeGrants: PermissionGrant[];
  /** Running sub-agent tasks. */
  activeTasks: Map<string, { taskSpec: TaskSpec; startedAt: number }>;
  /** Completed results awaiting integration. */
  pendingResults: UltraThinkResult[];
};

/**
 * Process a handoff message from the Interface Agent.
 * This is the main entry point for the Conductor.
 */
export async function processHandoff(
  state: ConductorState,
  message: HandoffMessage,
  infer: (req: InferenceRequest) => Promise<InferenceResponse>,
): Promise<{
  state: ConductorState;
  response: string;
  tasks?: TaskSpec[];
}> {
  // Validate the handoff at the trust boundary
  const validation = validateHandoff(message);
  if (!validation.valid) {
    return {
      state,
      response: `Handoff rejected: ${validation.errors.join(", ")}`,
    };
  }

  // Log sanitization flags if any
  if (message.sanitizationFlags.length > 0) {
    // Record but don't block — the Interface Agent already sanitized
    state = {
      ...state,
      memory: recordTurn(
        state.memory,
        "conductor",
        `[SECURITY] Input sanitization flags: ${message.sanitizationFlags.join(", ")}`,
        state.agentId,
      ),
    };
  }

  // Record the human turn
  state = {
    ...state,
    memory: recordTurn(state.memory, "human", message.content),
  };

  switch (message.intent) {
    case "task":
      return handleTask(state, message, infer);
    case "question":
      return handleQuestion(state, message, infer);
    case "feedback":
      return handleFeedback(state, message, infer);
    case "constraint-update":
      return handleConstraintUpdate(state, message);
    case "clarification":
      return handleClarification(state, message, infer);
  }
}

async function handleTask(
  state: ConductorState,
  message: HandoffMessage,
  infer: (req: InferenceRequest) => Promise<InferenceResponse>,
): Promise<{ state: ConductorState; response: string; tasks?: TaskSpec[] }> {
  // Build the full conductor prompt with all memory layers
  const contextPrompt = buildConductorPrompt(state.memory);

  // Ask the orchestrator model to decompose the task
  const decompositionRequest: InferenceRequest = {
    systemPrompt: `You are the Conductor — an orchestration agent that decomposes engineering tasks.

You NEVER write code. You NEVER run commands. You PLAN and DELEGATE.

For each sub-task, specify:
1. A clear, scoped description
2. Which files the Coder needs to read
3. Which directories the Coder can write to
4. What commands the Coder can run
5. Difficulty estimate (tier 1-4, estimated success probability)

Respond in JSON format:
{
  "plan": "Overall approach in 2-3 sentences",
  "tasks": [
    {
      "description": "...",
      "contextFiles": ["..."],
      "writeScope": ["..."],
      "allowedCommands": ["..."],
      "estimatedTier": 1-4,
      "estimatedP": 0.0-1.0,
      "reasoning": "Why this difficulty level"
    }
  ]
}`,
    prompt: `${contextPrompt}

CURRENT REQUEST:
${message.content}

${message.metadata.targetFiles ? `TARGET FILES: ${message.metadata.targetFiles.join(", ")}` : ""}
${message.metadata.priority ? `PRIORITY: ${message.metadata.priority}` : ""}

Decompose this into concrete, executable sub-tasks for the Coder agents.`,
    temperature: 0.3,
    topP: 0.95,
    maxTokens: 4_000,
    thinkingMode: true,
  };

  const response = await infer(decompositionRequest);

  // Parse the decomposition
  const tasks = parseDecomposition(state, message, response.text);

  // Record the conductor's plan
  state = {
    ...state,
    memory: recordTurn(
      state.memory,
      "conductor",
      `Decomposed task into ${tasks.length} sub-task(s):\n${tasks.map((t) => `- [T${t.difficulty.tier}] ${t.description}`).join("\n")}`,
      state.agentId,
    ),
  };

  return {
    state,
    response: `Task decomposed into ${tasks.length} sub-task(s). Dispatching to Coder agents.`,
    tasks,
  };
}

async function handleQuestion(
  state: ConductorState,
  message: HandoffMessage,
  infer: (req: InferenceRequest) => Promise<InferenceResponse>,
): Promise<{ state: ConductorState; response: string }> {
  const contextPrompt = buildConductorPrompt(state.memory);

  const response = await infer({
    systemPrompt: "You are the Conductor. Answer questions about the project using your memory and knowledge graph. Be concise and factual.",
    prompt: `${contextPrompt}\n\nQUESTION: ${message.content}`,
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 2_000,
    thinkingMode: false,
  });

  state = {
    ...state,
    memory: recordTurn(state.memory, "conductor", response.text, state.agentId),
  };

  return { state, response: response.text };
}

async function handleFeedback(
  state: ConductorState,
  message: HandoffMessage,
  infer: (req: InferenceRequest) => Promise<InferenceResponse>,
): Promise<{ state: ConductorState; response: string }> {
  // Record feedback and update the last pending task's verdict
  state = {
    ...state,
    memory: recordTurn(state.memory, "conductor", `Feedback received: ${message.content}`, state.agentId),
  };

  return {
    state,
    response: "Feedback recorded. It will be used to improve future task execution.",
  };
}

function handleConstraintUpdate(
  state: ConductorState,
  message: HandoffMessage,
): { state: ConductorState; response: string } {
  // Extract the constraint from the message and add to Layer 0
  const { addConstraint } = require("../memory/layer0-constraints.js");
  const nextL0 = addConstraint(state.memory.layer0, message.content, "general", "human");

  state = {
    ...state,
    memory: {
      ...state.memory,
      layer0: nextL0,
    },
  };

  return {
    state,
    response: `Constraint pinned to Layer 0: "${message.content}"`,
  };
}

async function handleClarification(
  state: ConductorState,
  message: HandoffMessage,
  infer: (req: InferenceRequest) => Promise<InferenceResponse>,
): Promise<{ state: ConductorState; response: string }> {
  return handleQuestion(state, message, infer);
}

/**
 * Parse the LLM's decomposition response into TaskSpec objects.
 */
function parseDecomposition(
  state: ConductorState,
  message: HandoffMessage,
  responseText: string,
): TaskSpec[] {
  // Extract JSON from the response (may be wrapped in markdown code blocks)
  const jsonMatch = responseText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    // Fallback: treat the entire message as a single task
    return [buildSingleTask(state, message)];
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as {
      tasks?: Array<{
        description: string;
        contextFiles?: string[];
        writeScope?: string[];
        allowedCommands?: string[];
        estimatedTier?: number;
        estimatedP?: number;
        reasoning?: string;
      }>;
    };

    if (!parsed.tasks || !Array.isArray(parsed.tasks) || parsed.tasks.length === 0) {
      return [buildSingleTask(state, message)];
    }

    return parsed.tasks.map((t) => {
      const tier = clampTier(t.estimatedTier ?? 2);
      const estimatedP = Math.max(0, Math.min(1, t.estimatedP ?? 0.5));

      const contextFiles = t.contextFiles ?? message.metadata.targetFiles ?? [];
      const knowledgeContext = queryKnowledge(state.memory, contextFiles);
      const historicalRate = queryHistoricalSuccess(state.memory, t.description);

      // Select relevant exemplars
      const exemplars = selectExemplars(state.config.exemplarLibrary, t.description, 3);

      const spec: TaskSpec = {
        id: `task-${crypto.randomUUID().slice(0, 8)}`,
        description: t.description,
        difficulty: {
          tier,
          estimatedP,
          reasoning: t.reasoning ?? "Estimated by Conductor",
          modulesAffected: contextFiles,
          historicalSuccessRate: historicalRate,
        },
        contextFiles,
        writeScope: t.writeScope ?? [],
        allowedCommands: t.allowedCommands ?? ["npm test", "pnpm test"],
        constraints: state.memory.layer0,
        knowledgeContext,
        exemplars,
        instructions: t.description,
        createdAt: Date.now(),
      };

      // Validate at the trust boundary
      const validation = validateTaskSpec(spec, state.agentId);
      if (!validation.valid) {
        // Log the validation failure but still create the task with empty scope
        spec.writeScope = [];
        spec.allowedCommands = [];
      }

      return spec;
    });
  } catch {
    return [buildSingleTask(state, message)];
  }
}

function buildSingleTask(state: ConductorState, message: HandoffMessage): TaskSpec {
  const contextFiles = message.metadata.targetFiles ?? [];
  return {
    id: `task-${crypto.randomUUID().slice(0, 8)}`,
    description: message.content,
    difficulty: {
      tier: 2,
      estimatedP: 0.5,
      reasoning: "Default estimate — single undecomposed task",
      modulesAffected: contextFiles,
      historicalSuccessRate: queryHistoricalSuccess(state.memory, message.content),
    },
    contextFiles,
    writeScope: [],
    allowedCommands: ["npm test", "pnpm test"],
    constraints: state.memory.layer0,
    knowledgeContext: queryKnowledge(state.memory, contextFiles),
    exemplars: selectExemplars(state.config.exemplarLibrary, message.content, 3),
    instructions: message.content,
    createdAt: Date.now(),
  };
}

function clampTier(raw: number): ComputeTier {
  if (raw <= 1) return 1;
  if (raw >= 4) return 4;
  return Math.round(raw) as ComputeTier;
}

function selectExemplars(library: Exemplar[], taskDescription: string, count: number): Exemplar[] {
  if (library.length === 0) return [];

  // Score exemplars by word overlap with the task description
  const taskWords = new Set(taskDescription.toLowerCase().split(/\s+/));
  const scored = library.map((ex) => {
    const exWords = new Set(ex.taskDescription.toLowerCase().split(/\s+/));
    let overlap = 0;
    for (const w of taskWords) {
      if (exWords.has(w)) overlap++;
    }
    return { exemplar: ex, score: overlap / taskWords.size };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, count).map((s) => s.exemplar);
}

/**
 * Integrate a completed Ultra Think result into the memory stack.
 */
export function integrateResult(
  state: ConductorState,
  result: UltraThinkResult,
): ConductorState {
  const entry: Omit<ChangelogEntry, "taskHash"> = {
    ts: Date.now(),
    agentId: result.selectedOutput.agentId,
    taskDescription: state.activeTasks.get(result.taskId)?.taskSpec.description ?? "Unknown task",
    filesModified: [], // Will be populated by the actual file diff
    testsRun: result.testResult.failures.length > 0
      ? result.testResult.failures.map((f) => ({ name: f.testName, passed: false }))
      : [{ name: "all", passed: true }],
    reviewerScore: result.reviewResult.score.overall,
    humanVerdict: "pending",
    tier: result.tier,
    generationsUsed: result.totalGenerations,
    wallClockMs: result.wallClockMs,
  };

  const nextMemory = recordTask(state.memory, entry);
  const nextActive = new Map(state.activeTasks);
  nextActive.delete(result.taskId);

  return {
    ...state,
    memory: nextMemory,
    activeTasks: nextActive,
    pendingResults: [...state.pendingResults, result],
  };
}
