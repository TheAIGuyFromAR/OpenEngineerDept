/**
 * Conductor — Entry Point.
 *
 * Wires up all components: memory stack, agents, ensemble inference,
 * and input adapters into a running system.
 *
 * This module is the top-level API for the Conductor architecture.
 * It can be used standalone or integrated into the existing Maistro
 * gateway infrastructure.
 */

// Core types
export type {
  AgentId,
  AgentRole,
  ChangelogEntry,
  CoderOutput,
  CompressedEntry,
  ComputeTier,
  ConstraintEntry,
  ConvergenceAnalysis,
  DifficultyEstimate,
  Exemplar,
  HandoffMessage,
  InferenceRequest,
  InferenceResponse,
  KnowledgeGraph,
  KnowledgeGraphNode,
  MemoryStack,
  ParetoMetrics,
  PermissionScope,
  ReviewResult,
  ReviewScore,
  TaskSpec,
  TestResult,
  UltraThinkConfig,
  UltraThinkResult,
  VllmEndpoint,
  WorkingMemoryTurn,
} from "./types.js";

export { DEFAULT_PERMISSIONS, TIER_CONFIGS } from "./types.js";

// Memory stack
export {
  buildConductorPrompt,
  buildCoderPrompt,
  createMemoryStack,
  estimateStackTokens,
  persistState,
  queryHistoricalSuccess,
  queryKnowledge,
  recordTask,
  recordTurn,
  type MemoryStackConfig,
  type MemoryStackState,
} from "./memory/stack.js";

// Agents
export { parseObsidianInput, parseOpenWebUIInput } from "./agents/interface-agent.js";
export {
  integrateResult,
  processHandoff,
  type ConductorConfig,
  type ConductorState,
} from "./agents/conductor.js";
export { executeCoderGeneration, runTests } from "./agents/coder.js";
export {
  analyzeConvergence,
  reviewOutput,
  selectBestOutput,
} from "./agents/reviewer.js";
export {
  allocateBudget,
  assignTier,
  computeOptimalN,
  estimateDifficulty,
  expectedCost,
  getComputeConfig,
  type PlannerConfig,
} from "./agents/planner.js";

// Ensemble
export {
  executeAtTier,
  executeUltraThink,
  type UltraThinkDeps,
} from "./ensemble/ultra-think.js";

// Security
export {
  checkPermission,
  createPermissionGrant,
  validateHandoff,
  validateTaskSpec,
  type PermissionGrant,
} from "./security/trust-boundary.js";

// Adapters
export {
  scanOnce,
  startWatcher,
  type ObsidianWatcherConfig,
  type WatcherCallback,
  type WatcherHandle,
} from "./adapters/obsidian-watcher.js";
