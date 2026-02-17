/**
 * Core type definitions for the Conductor architecture.
 *
 * The Conductor is a multi-agent orchestration system with six agent types
 * operating across two trust boundaries. These types define the contracts
 * between agents, the memory stack, and the verification pipeline.
 */

// ---------------------------------------------------------------------------
// Agent Identity & Permissions
// ---------------------------------------------------------------------------

export type AgentRole =
  | "interface"
  | "conductor"
  | "planner"
  | "coder"
  | "reviewer"
  | "scout";

export type AgentId = `${AgentRole}-${string}`;

export type PermissionScope = {
  /** Directories the agent can read from (glob patterns). */
  readPaths: string[];
  /** Directories the agent can write to (glob patterns). */
  writePaths: string[];
  /** Whether the agent can execute commands. */
  canExecute: boolean;
  /** Specific commands the agent is allowed to run (when canExecute is true). */
  allowedCommands?: string[];
  /** Maximum output tokens per generation. */
  maxOutputTokens: number;
  /** Maximum wall-clock time in ms before the agent is killed. */
  timeoutMs: number;
};

/** Default permission scopes per agent role. */
export const DEFAULT_PERMISSIONS: Record<AgentRole, PermissionScope> = {
  interface: {
    readPaths: [],
    writePaths: [],
    canExecute: false,
    maxOutputTokens: 2_000,
    timeoutMs: 10_000,
  },
  conductor: {
    readPaths: ["**/*"],
    writePaths: [],
    canExecute: false,
    maxOutputTokens: 8_000,
    timeoutMs: 120_000,
  },
  planner: {
    readPaths: ["**/*"],
    writePaths: [],
    canExecute: false,
    maxOutputTokens: 4_000,
    timeoutMs: 60_000,
  },
  coder: {
    readPaths: [],  // Scoped per task by Conductor
    writePaths: [],  // Scoped per task by Conductor
    canExecute: true,
    allowedCommands: [],  // Scoped per task by Conductor
    maxOutputTokens: 16_000,
    timeoutMs: 300_000,
  },
  reviewer: {
    readPaths: ["**/*"],
    writePaths: [],
    canExecute: true,
    allowedCommands: ["npm test", "pnpm test", "vitest", "jest", "pytest"],
    maxOutputTokens: 4_000,
    timeoutMs: 120_000,
  },
  scout: {
    readPaths: ["**/*"],
    writePaths: [],
    canExecute: false,
    maxOutputTokens: 4_000,
    timeoutMs: 60_000,
  },
};

// ---------------------------------------------------------------------------
// Memory Stack
// ---------------------------------------------------------------------------

export type ConstraintEntry = {
  id: string;
  rule: string;
  category: "style" | "architecture" | "testing" | "security" | "api" | "general";
  addedAt: number;
  source: "human" | "repetition-detector";
};

export type WorkingMemoryTurn = {
  role: "human" | "conductor" | "agent";
  agentId?: AgentId;
  content: string;
  ts: number;
};

export type CompressedEntry = {
  originalTurnRange: [number, number];
  summary: string;
  decisions: string[];
  outcomes: string[];
  compressedAt: number;
};

export type ChangelogEntry = {
  taskHash: string;
  ts: number;
  agentId: AgentId;
  taskDescription: string;
  filesModified: string[];
  testsRun: { name: string; passed: boolean }[];
  reviewerScore: number | null;
  humanVerdict: "accepted" | "rejected" | "revised" | "pending";
  tier: ComputeTier;
  generationsUsed: number;
  wallClockMs: number;
};

export type KnowledgeGraphNode = {
  id: string;
  type: "module" | "file" | "function" | "class" | "api" | "dependency";
  name: string;
  path?: string;
  description?: string;
  dependencies: string[];
  dependents: string[];
  lastUpdated: number;
  metadata?: Record<string, unknown>;
};

export type KnowledgeGraph = {
  version: number;
  nodes: Map<string, KnowledgeGraphNode>;
  lastFullScan: number;
};

export type MemoryStack = {
  layer0: ConstraintEntry[];
  layer1: WorkingMemoryTurn[];
  layer2: CompressedEntry[];
  layer3: ChangelogEntry[];
  layer4: KnowledgeGraph;
};

// ---------------------------------------------------------------------------
// Task Decomposition & Compute Allocation
// ---------------------------------------------------------------------------

export type ComputeTier = 1 | 2 | 3 | 4;

export type DifficultyEstimate = {
  tier: ComputeTier;
  estimatedP: number;
  reasoning: string;
  modulesAffected: string[];
  historicalSuccessRate: number | null;
};

export type TaskSpec = {
  id: string;
  parentId?: string;
  description: string;
  difficulty: DifficultyEstimate;
  /** Files the Coder should read for context. */
  contextFiles: string[];
  /** Specific directories the Coder can write to. */
  writeScope: string[];
  /** Commands the Coder can execute. */
  allowedCommands: string[];
  /** Layer 0 constraints relevant to this task. */
  constraints: ConstraintEntry[];
  /** Knowledge graph slices relevant to this task. */
  knowledgeContext: KnowledgeGraphNode[];
  /** Best-completion exemplars for few-shot prompting. */
  exemplars: Exemplar[];
  /** Explicit instructions from the Conductor. */
  instructions: string;
  createdAt: number;
};

export type Exemplar = {
  taskDescription: string;
  implementation: string;
  reviewerScore: number;
  humanVerdict: "accepted";
  taskCategory: string;
};

// ---------------------------------------------------------------------------
// Ultra Think
// ---------------------------------------------------------------------------

export type UltraThinkConfig = {
  tier: ComputeTier;
  /** Number of parallel generations. */
  n: number;
  /** Temperature range for diversity. */
  temperatureRange: [number, number];
  /** Maximum retry rounds. */
  maxRetryRounds: number;
  /** Minimum Reviewer score to accept. */
  acceptThreshold: number;
  /** Whether to use thinking mode. */
  thinkingMode: boolean;
};

export const TIER_CONFIGS: Record<ComputeTier, UltraThinkConfig> = {
  1: {
    tier: 1,
    n: 1,
    temperatureRange: [0.3, 0.3],
    maxRetryRounds: 1,
    acceptThreshold: 6.0,
    thinkingMode: false,
  },
  2: {
    tier: 2,
    n: 3,
    temperatureRange: [0.4, 0.7],
    maxRetryRounds: 1,
    acceptThreshold: 6.5,
    thinkingMode: false,
  },
  3: {
    tier: 3,
    n: 5,
    temperatureRange: [0.5, 0.9],
    maxRetryRounds: 3,
    acceptThreshold: 7.0,
    thinkingMode: true,
  },
  4: {
    tier: 4,
    n: 10,
    temperatureRange: [0.3, 0.9],
    maxRetryRounds: 5,
    acceptThreshold: 7.5,
    thinkingMode: true,
  },
};

export type CoderOutput = {
  agentId: AgentId;
  taskId: string;
  code: string;
  tests: string;
  documentation: string;
  thinkingTrace?: string;
  temperature: number;
  systemPromptVariant: string;
  generationTimeMs: number;
  tokenCount: number;
};

export type TestResult = {
  agentId: AgentId;
  taskId: string;
  passed: boolean;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  failures: { testName: string; error: string }[];
  executionTimeMs: number;
};

export type ReviewScore = {
  correctness: number;
  style: number;
  architecture: number;
  robustness: number;
  clarity: number;
  overall: number;
  justification: string;
  critiques: string[];
};

export type ReviewResult = {
  agentId: AgentId;
  taskId: string;
  coderAgentId: AgentId;
  score: ReviewScore;
  recommendation: "accept" | "revise" | "reject";
};

export type ConvergenceAnalysis = {
  pattern: "convergent" | "divergent" | "structured-split";
  /** Clusters of implementations that share the same approach. */
  clusters: {
    approachSummary: string;
    members: AgentId[];
    averageScore: number;
  }[];
  confidence: number;
  recommendation: string;
};

export type UltraThinkResult = {
  taskId: string;
  tier: ComputeTier;
  rounds: number;
  totalGenerations: number;
  selectedOutput: CoderOutput;
  testResult: TestResult;
  reviewResult: ReviewResult;
  convergence: ConvergenceAnalysis;
  allOutputs: CoderOutput[];
  allTests: TestResult[];
  allReviews: ReviewResult[];
  wallClockMs: number;
  escalated: boolean;
  escalationReason?: string;
};

// ---------------------------------------------------------------------------
// Structured Handoff Protocol (Interface → Conductor)
// ---------------------------------------------------------------------------

export type HandoffMessage = {
  id: string;
  source: "obsidian" | "openwebui";
  /** Sanitized user intent classification. */
  intent: "task" | "question" | "clarification" | "feedback" | "constraint-update";
  /** Sanitized content — never raw user input. */
  content: string;
  /** Extracted metadata from frontmatter or chat context. */
  metadata: {
    priority?: "low" | "normal" | "high" | "urgent";
    category?: string;
    targetFiles?: string[];
    referencedTasks?: string[];
  };
  /** Suspicious patterns detected in raw input (for logging). */
  sanitizationFlags: string[];
  ts: number;
};

// ---------------------------------------------------------------------------
// Pareto Metrics
// ---------------------------------------------------------------------------

export type ParetoMetrics = {
  taskId: string;
  tierAssigned: ComputeTier;
  tierConsumed: ComputeTier;
  generationsFired: number;
  generationAccepted: number;
  testPassRatePerGeneration: number[];
  reviewerScorePerGeneration: number[];
  humanAcceptedFirstPresentation: boolean;
  wallClockMs: number;
  taskCategory: string;
  modelId: string;
  loraVersion: string | null;
};

// ---------------------------------------------------------------------------
// vLLM Integration
// ---------------------------------------------------------------------------

export type VllmEndpoint = {
  baseUrl: string;
  modelId: string;
  loraAdapter?: string;
};

export type InferenceRequest = {
  prompt: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
  thinkingMode: boolean;
  stopSequences?: string[];
};

export type InferenceResponse = {
  text: string;
  thinkingTrace?: string;
  tokenCount: number;
  finishReason: "stop" | "length" | "error";
  latencyMs: number;
};
