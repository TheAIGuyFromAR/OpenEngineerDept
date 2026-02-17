/**
 * Layer 3: Annotated Changelog.
 *
 * Sequential, factual, diagnostic record of what was done, when, by which
 * agent, and what the outcome was. Supports the threefold repetition
 * detector: if the same approach to the same problem appears three times,
 * the pattern is flagged.
 *
 * Lives as a markdown file in the Obsidian vault.
 * Typical size: 2,000–8,000 tokens (most recent entries loaded).
 */

import crypto from "node:crypto";
import fs from "node:fs";
import type { AgentId, ChangelogEntry, ComputeTier, ConstraintEntry } from "../types.js";

export type Layer3Config = {
  /** Path to the changelog file. */
  changelogPath: string;
  /** Maximum recent entries to include in prompts. */
  maxPromptEntries: number;
  /** Threshold for repetition detection. */
  repetitionThreshold: number;
};

const DEFAULT_CONFIG: Layer3Config = {
  changelogPath: "changelog.md",
  maxPromptEntries: 50,
  repetitionThreshold: 3,
};

export type Layer3State = {
  config: Layer3Config;
  entries: ChangelogEntry[];
};

export function createChangelog(config?: Partial<Layer3Config>): Layer3State {
  return {
    config: { ...DEFAULT_CONFIG, ...config },
    entries: [],
  };
}

export function addEntry(
  state: Layer3State,
  entry: Omit<ChangelogEntry, "taskHash">,
): Layer3State {
  const taskHash = crypto
    .createHash("sha256")
    .update(`${entry.taskDescription}:${entry.ts}:${entry.agentId}`)
    .digest("hex")
    .slice(0, 16);

  const fullEntry: ChangelogEntry = { ...entry, taskHash };
  return {
    ...state,
    entries: [...state.entries, fullEntry],
  };
}

/**
 * Threefold Repetition Detector.
 *
 * Identifies when the same approach to the same problem has been tried
 * three or more times. Uses fuzzy matching on task description to detect
 * similar tasks, then checks if the approaches (files modified, agent
 * patterns) are similar.
 *
 * Returns entries that should be escalated — either pinned to Layer 0
 * as a constraint or flagged for human attention.
 */
export type RepetitionDetection = {
  detected: boolean;
  pattern: string;
  occurrences: ChangelogEntry[];
  recommendation: "pin-constraint" | "escalate-human";
  suggestedConstraint?: string;
};

export function detectRepetitions(state: Layer3State): RepetitionDetection[] {
  const detections: RepetitionDetection[] = [];
  const buckets = new Map<string, ChangelogEntry[]>();

  // Group entries by normalized task description
  for (const entry of state.entries) {
    const key = normalizeTaskDescription(entry.taskDescription);
    const list = buckets.get(key) ?? [];
    list.push(entry);
    buckets.set(key, list);
  }

  for (const [pattern, entries] of buckets) {
    if (entries.length < state.config.repetitionThreshold) {
      continue;
    }

    // Check if the failed approaches are similar
    const failedEntries = entries.filter(
      (e) => e.humanVerdict === "rejected" || e.humanVerdict === "revised",
    );

    if (failedEntries.length >= state.config.repetitionThreshold) {
      // Same problem, same failures — need a constraint
      const filesInCommon = findCommonFiles(failedEntries);
      detections.push({
        detected: true,
        pattern,
        occurrences: failedEntries,
        recommendation: "pin-constraint",
        suggestedConstraint: `When working on "${pattern}", avoid approaches that modify: ${filesInCommon.join(", ")}. Previous ${failedEntries.length} attempts with this approach were rejected.`,
      });
    } else if (entries.length >= state.config.repetitionThreshold) {
      detections.push({
        detected: true,
        pattern,
        occurrences: entries,
        recommendation: "escalate-human",
      });
    }
  }

  return detections;
}

/**
 * Convert repetition detections into Layer 0 constraints.
 */
export function repetitionsToConstraints(
  detections: RepetitionDetection[],
): ConstraintEntry[] {
  return detections
    .filter((d) => d.recommendation === "pin-constraint" && d.suggestedConstraint)
    .map((d) => ({
      id: crypto
        .createHash("sha256")
        .update(d.suggestedConstraint!)
        .digest("hex")
        .slice(0, 12),
      rule: d.suggestedConstraint!,
      category: "general" as const,
      addedAt: Date.now(),
      source: "repetition-detector" as const,
    }));
}

function normalizeTaskDescription(desc: string): string {
  return desc
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 100);
}

function findCommonFiles(entries: ChangelogEntry[]): string[] {
  if (entries.length === 0) return [];

  const fileCounts = new Map<string, number>();
  for (const entry of entries) {
    for (const file of entry.filesModified) {
      fileCounts.set(file, (fileCounts.get(file) ?? 0) + 1);
    }
  }

  // Files that appear in at least half of the entries
  const threshold = Math.ceil(entries.length / 2);
  return Array.from(fileCounts.entries())
    .filter(([, count]) => count >= threshold)
    .map(([file]) => file);
}

/**
 * Query changelog for historical success rates on similar tasks.
 */
export function getHistoricalSuccessRate(
  state: Layer3State,
  taskDescription: string,
): number | null {
  const normalized = normalizeTaskDescription(taskDescription);
  const similar = state.entries.filter((e) => {
    const entryNorm = normalizeTaskDescription(e.taskDescription);
    return stringSimilarity(normalized, entryNorm) > 0.6;
  });

  if (similar.length < 3) {
    return null; // Not enough data
  }

  const accepted = similar.filter((e) => e.humanVerdict === "accepted").length;
  return accepted / similar.length;
}

function stringSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  if (a.length === 0 || b.length === 0) return 0;

  const words1 = new Set(a.split(" "));
  const words2 = new Set(b.split(" "));
  let intersection = 0;
  for (const w of words1) {
    if (words2.has(w)) intersection++;
  }
  return (2 * intersection) / (words1.size + words2.size);
}

/**
 * Serialize changelog to markdown for Obsidian persistence.
 */
export function serializeChangelog(state: Layer3State): string {
  const lines = ["# Changelog", ""];

  for (const entry of state.entries) {
    const date = new Date(entry.ts).toISOString();
    const verdict = entry.humanVerdict.toUpperCase();
    const score = entry.reviewerScore !== null ? `Score: ${entry.reviewerScore}/10` : "Unreviewed";
    const testSummary =
      entry.testsRun.length > 0
        ? `${entry.testsRun.filter((t) => t.passed).length}/${entry.testsRun.length} passed`
        : "No tests";

    lines.push(`## [${verdict}] ${entry.taskDescription}`);
    lines.push(`- **Date**: ${date}`);
    lines.push(`- **Agent**: ${entry.agentId}`);
    lines.push(`- **Tier**: ${entry.tier} | **Generations**: ${entry.generationsUsed}`);
    lines.push(`- **Tests**: ${testSummary} | ${score}`);
    lines.push(`- **Wall clock**: ${(entry.wallClockMs / 1000).toFixed(1)}s`);
    if (entry.filesModified.length > 0) {
      lines.push(`- **Files**: ${entry.filesModified.join(", ")}`);
    }
    lines.push(`- **Hash**: \`${entry.taskHash}\``);
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Build the prompt fragment for Layer 3 inclusion (recent entries only).
 */
export function buildLayer3Prompt(state: Layer3State): string {
  const recent = state.entries.slice(-state.config.maxPromptEntries);
  if (recent.length === 0) {
    return "";
  }

  const lines = ["=== CHANGELOG (Layer 3) ===", ""];

  for (const entry of recent) {
    const verdict = entry.humanVerdict;
    const testStatus =
      entry.testsRun.length > 0
        ? `${entry.testsRun.filter((t) => t.passed).length}/${entry.testsRun.length} tests`
        : "no tests";

    lines.push(
      `[${verdict}] ${entry.taskDescription} | T${entry.tier} | ${testStatus} | ${entry.filesModified.length} files`,
    );
  }

  lines.push("", "=== END CHANGELOG ===");
  return lines.join("\n");
}
