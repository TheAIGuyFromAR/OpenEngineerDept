/**
 * Memory Stack Manager.
 *
 * Coordinates all five memory layers. Handles the continuous flow of
 * information from Layer 1 → Layer 2 (rolling compaction), repetition
 * detection in Layer 3, and selective knowledge graph queries from Layer 4.
 *
 * This is the central memory interface that the Conductor uses.
 */

import type {
  AgentId,
  ChangelogEntry,
  ConstraintEntry,
  KnowledgeGraphNode,
  MemoryStack,
  WorkingMemoryTurn,
} from "../types.js";
import {
  addConstraint,
  buildLayer0Prompt,
  loadConstraints,
  type Layer0Config,
} from "./layer0-constraints.js";
import {
  addTurn,
  buildLayer1Prompt,
  createWorkingMemory,
  estimateTotalTokens,
  type Layer1Config,
  type Layer1State,
} from "./layer1-working.js";
import {
  applyCompression,
  buildCompressionPrompt,
  buildLayer2Prompt,
  compressLocally,
  createCompressedHistory,
  needsCompaction,
  receiveTurns,
  type Layer2Config,
  type Layer2State,
} from "./layer2-compressed.js";
import {
  addEntry as addChangelogEntry,
  buildLayer3Prompt,
  createChangelog,
  detectRepetitions,
  getHistoricalSuccessRate,
  repetitionsToConstraints,
  serializeChangelog,
  type Layer3Config,
  type Layer3State,
} from "./layer3-changelog.js";
import {
  buildLayer4Prompt,
  createKnowledgeGraph,
  loadGraph,
  queryByFiles,
  saveGraph,
  type Layer4Config,
} from "./layer4-knowledge.js";

export type MemoryStackConfig = {
  layer0: Layer0Config;
  layer1?: Partial<Layer1Config>;
  layer2?: Partial<Layer2Config>;
  layer3?: Partial<Layer3Config>;
  layer4?: Partial<Layer4Config>;
};

export type MemoryStackState = {
  layer0: ConstraintEntry[];
  layer1: Layer1State;
  layer2: Layer2State;
  layer3: Layer3State;
  layer4: {
    config: Layer4Config;
    graph: MemoryStack["layer4"];
  };
};

export function createMemoryStack(config: MemoryStackConfig): MemoryStackState {
  const layer0 = loadConstraints(config.layer0);
  const layer1 = createWorkingMemory(config.layer1);
  const layer2 = createCompressedHistory(config.layer2);
  const layer3 = createChangelog(config.layer3);
  const layer4 = createKnowledgeGraph(config.layer4);

  // Try to load persisted knowledge graph
  const persisted = loadGraph(layer4.config.graphPath);
  if (persisted) {
    layer4.graph = persisted;
  }

  return { layer0, layer1, layer2, layer3, layer4 };
}

/**
 * Record a conversation turn. Handles L1 overflow → L2 compaction.
 */
export function recordTurn(
  state: MemoryStackState,
  role: WorkingMemoryTurn["role"],
  content: string,
  agentId?: AgentId,
): MemoryStackState {
  const { state: nextL1, overflow } = addTurn(state.layer1, role, content, agentId);
  let nextL2 = state.layer2;

  if (overflow.length > 0) {
    nextL2 = receiveTurns(nextL2, overflow);
    // Auto-compact if enough turns have accumulated
    if (needsCompaction(nextL2)) {
      nextL2 = compressLocally(nextL2);
    }
  }

  return { ...state, layer1: nextL1, layer2: nextL2 };
}

/**
 * Record a completed task in the changelog.
 */
export function recordTask(
  state: MemoryStackState,
  entry: Omit<ChangelogEntry, "taskHash">,
): MemoryStackState {
  const nextL3 = addChangelogEntry(state.layer3, entry);

  // Check for repetitions and auto-pin constraints
  const repetitions = detectRepetitions(nextL3);
  let nextL0 = state.layer0;
  for (const constraint of repetitionsToConstraints(repetitions)) {
    nextL0 = addConstraint(nextL0, constraint.rule, constraint.category, constraint.source);
  }

  return { ...state, layer0: nextL0, layer3: nextL3 };
}

/**
 * Build a complete prompt for the Conductor (all layers).
 * Target: 16,000–64,000 tokens depending on project maturity.
 */
export function buildConductorPrompt(state: MemoryStackState): string {
  const parts = [
    buildLayer0Prompt(state.layer0),
    buildLayer2Prompt(state.layer2),
    buildLayer3Prompt(state.layer3),
    buildLayer1Prompt(state.layer1),
  ].filter(Boolean);

  return parts.join("\n\n");
}

/**
 * Build a scoped prompt for a Coder agent.
 * Target: 4,000–12,000 tokens of context.
 */
export function buildCoderPrompt(
  state: MemoryStackState,
  contextFiles: string[],
): string {
  const knowledgeNodes = queryByFiles(
    state.layer4.graph,
    contextFiles,
    state.layer4.config,
  );

  const parts = [
    buildLayer0Prompt(state.layer0),
    buildLayer4Prompt(knowledgeNodes),
  ].filter(Boolean);

  return parts.join("\n\n");
}

/**
 * Get historical success rate for a task description.
 */
export function queryHistoricalSuccess(
  state: MemoryStackState,
  taskDescription: string,
): number | null {
  return getHistoricalSuccessRate(state.layer3, taskDescription);
}

/**
 * Get relevant knowledge graph nodes for specific files.
 */
export function queryKnowledge(
  state: MemoryStackState,
  files: string[],
): KnowledgeGraphNode[] {
  return queryByFiles(state.layer4.graph, files, state.layer4.config);
}

/**
 * Estimate the total token count of the memory stack.
 */
export function estimateStackTokens(state: MemoryStackState): {
  layer0: number;
  layer1: number;
  layer2: number;
  layer3: number;
  total: number;
} {
  const l0 = Math.ceil(buildLayer0Prompt(state.layer0).length / 4);
  const l1 = estimateTotalTokens(state.layer1.turns);
  const l2 = Math.ceil(buildLayer2Prompt(state.layer2).length / 4);
  const l3 = Math.ceil(buildLayer3Prompt(state.layer3).length / 4);
  return { layer0: l0, layer1: l1, layer2: l2, layer3: l3, total: l0 + l1 + l2 + l3 };
}

/**
 * Persist the current state to disk.
 */
export function persistState(state: MemoryStackState): void {
  saveGraph(state.layer4.graph, state.layer4.config.graphPath);
  // Changelog is serialized to markdown for Obsidian
  const changelogMd = serializeChangelog(state.layer3);
  const changelogPath = state.layer3.config.changelogPath;
  const dir = require("node:path").dirname(changelogPath);
  if (!require("node:fs").existsSync(dir)) {
    require("node:fs").mkdirSync(dir, { recursive: true });
  }
  require("node:fs").writeFileSync(changelogPath, changelogMd, "utf-8");
}
