/**
 * Journals — Cross-Session Reasoning Memory.
 *
 * Inspired by Claude Code's journals feature. While our Layer 3 changelog
 * tracks WHAT happened (files modified, tests run, scores), journals
 * capture WHY decisions were made and what was learned.
 *
 * Journals sit between Layer 2 (compressed history) and Layer 3 (changelog)
 * in importance. They persist across sessions and provide the Conductor
 * with institutional knowledge about the project.
 *
 * Journal entries are:
 *   - Auto-generated after each task completion (what worked, what didn't)
 *   - Human-authored (via Obsidian "journal" frontmatter type)
 *   - System-generated (when repetition detector fires, when escalation occurs)
 *
 * The journal is queryable by topic, recency, and relevance to current task.
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import type { AgentId, ComputeTier } from "../types.js";

export type JournalEntry = {
  id: string;
  ts: number;
  /** Who created this entry. */
  source: "conductor" | "human" | "reviewer" | "system";
  agentId?: AgentId;
  /** What kind of knowledge this captures. */
  category:
    | "decision"        // Why a particular approach was chosen
    | "lesson"          // What was learned from a success or failure
    | "pattern"         // Recurring pattern observed in the codebase
    | "constraint-rationale" // Why a constraint was added
    | "escalation"      // Why a task was escalated to human
    | "architecture"    // Architectural insight about the project
    | "debug"           // Debugging insight (root cause analysis)
    | "human-note";     // Direct human annotation
  /** Short title for search/display. */
  title: string;
  /** Full content — the reasoning, analysis, or note. */
  content: string;
  /** Related task IDs. */
  relatedTasks: string[];
  /** Related file paths. */
  relatedFiles: string[];
  /** Tags for topic-based retrieval. */
  tags: string[];
  /** Relevance score (higher = more important). Decays over time. */
  importance: number;
  /** How many times this entry has been surfaced in prompts. */
  retrievalCount: number;
};

export type JournalStore = {
  version: number;
  entries: JournalEntry[];
  lastPersisted: number;
};

/**
 * Create an empty journal store.
 */
export function createJournalStore(): JournalStore {
  return {
    version: 1,
    entries: [],
    lastPersisted: 0,
  };
}

/**
 * Record a decision — why a particular approach was chosen.
 */
export function recordDecision(
  store: JournalStore,
  params: {
    title: string;
    content: string;
    taskId?: string;
    files?: string[];
    tags?: string[];
    agentId?: AgentId;
  },
): JournalStore {
  return addEntry(store, {
    source: "conductor",
    agentId: params.agentId,
    category: "decision",
    title: params.title,
    content: params.content,
    relatedTasks: params.taskId ? [params.taskId] : [],
    relatedFiles: params.files ?? [],
    tags: params.tags ?? [],
    importance: 7,
  });
}

/**
 * Record a lesson learned from a completed task.
 * Auto-generated after Ultra Think completes.
 */
export function recordLesson(
  store: JournalStore,
  params: {
    taskDescription: string;
    succeeded: boolean;
    tier: ComputeTier;
    generationsUsed: number;
    reviewerScore: number;
    critiques: string[];
    taskId: string;
    files: string[];
  },
): JournalStore {
  const status = params.succeeded ? "succeeded" : "failed";
  const efficiency = params.generationsUsed === 1 ? "first try" : `${params.generationsUsed} generations`;

  const content = [
    `Task: ${params.taskDescription}`,
    `Outcome: ${status} (T${params.tier}, ${efficiency}, score: ${params.reviewerScore}/10)`,
    "",
    params.critiques.length > 0
      ? `Key critiques:\n${params.critiques.map((c) => `- ${c}`).join("\n")}`
      : "No significant critiques.",
    "",
    params.succeeded
      ? `Takeaway: T${params.tier} was ${params.generationsUsed <= params.tier ? "well-calibrated" : "under-estimated"} for this task type.`
      : `Takeaway: This task type may need tier escalation or constraint clarification.`,
  ].join("\n");

  return addEntry(store, {
    source: "system",
    category: "lesson",
    title: `${status}: ${params.taskDescription.slice(0, 60)}`,
    content,
    relatedTasks: [params.taskId],
    relatedFiles: params.files,
    tags: [status, `tier-${params.tier}`],
    importance: params.succeeded ? 5 : 8, // Failures are more important to remember
  });
}

/**
 * Record an escalation event — why the system gave up and asked a human.
 */
export function recordEscalation(
  store: JournalStore,
  params: {
    taskDescription: string;
    reason: string;
    taskId: string;
    attempts: number;
    bestScore: number;
  },
): JournalStore {
  return addEntry(store, {
    source: "system",
    category: "escalation",
    title: `Escalated: ${params.taskDescription.slice(0, 60)}`,
    content: [
      `Task: ${params.taskDescription}`,
      `Reason: ${params.reason}`,
      `Attempts: ${params.attempts}, best score: ${params.bestScore}/10`,
      "",
      "This task exceeded the system's current capability. Human intervention needed.",
    ].join("\n"),
    relatedTasks: [params.taskId],
    relatedFiles: [],
    tags: ["escalation", "needs-human"],
    importance: 9,
  });
}

/**
 * Record a human note (from Obsidian journal entry).
 */
export function recordHumanNote(
  store: JournalStore,
  params: {
    title: string;
    content: string;
    tags?: string[];
    files?: string[];
  },
): JournalStore {
  return addEntry(store, {
    source: "human",
    category: "human-note",
    title: params.title,
    content: params.content,
    relatedTasks: [],
    relatedFiles: params.files ?? [],
    tags: params.tags ?? [],
    importance: 8, // Human notes are inherently important
  });
}

/**
 * Query journals relevant to a task description.
 * Uses keyword overlap + recency + importance for ranking.
 */
export function queryRelevant(
  store: JournalStore,
  query: string,
  maxResults: number = 5,
): JournalEntry[] {
  const queryWords = new Set(
    query
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2),
  );

  const scored = store.entries.map((entry) => {
    // Keyword relevance
    const entryText = `${entry.title} ${entry.content} ${entry.tags.join(" ")}`.toLowerCase();
    const entryWords = new Set(entryText.split(/\s+/));
    let keywordScore = 0;
    for (const w of queryWords) {
      if (entryWords.has(w)) keywordScore++;
    }
    const keywordRelevance = queryWords.size > 0 ? keywordScore / queryWords.size : 0;

    // Recency decay (half-life of 7 days)
    const ageMs = Date.now() - entry.ts;
    const ageDays = ageMs / (1000 * 60 * 60 * 24);
    const recencyScore = Math.pow(0.5, ageDays / 7);

    // Importance (normalized to 0-1)
    const importanceScore = entry.importance / 10;

    // Combined score
    const score = keywordRelevance * 0.5 + recencyScore * 0.2 + importanceScore * 0.3;

    return { entry, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxResults).map((s) => s.entry);
}

/**
 * Query journals by file path — find all entries related to specific files.
 */
export function queryByFiles(
  store: JournalStore,
  files: string[],
): JournalEntry[] {
  const fileSet = new Set(files);
  return store.entries.filter((entry) =>
    entry.relatedFiles.some((f) => fileSet.has(f)),
  );
}

/**
 * Query journals by category.
 */
export function queryByCategory(
  store: JournalStore,
  category: JournalEntry["category"],
  limit: number = 10,
): JournalEntry[] {
  return store.entries
    .filter((e) => e.category === category)
    .sort((a, b) => b.ts - a.ts)
    .slice(0, limit);
}

/**
 * Build a journal prompt section for inclusion in Conductor/Coder prompts.
 * Selects the most relevant entries and formats them.
 */
export function buildJournalPrompt(
  store: JournalStore,
  query: string,
  maxEntries: number = 3,
): string {
  const relevant = queryRelevant(store, query, maxEntries);
  if (relevant.length === 0) return "";

  const lines = ["=== JOURNAL (cross-session memory) ===", ""];

  for (const entry of relevant) {
    lines.push(`[${entry.category.toUpperCase()}] ${entry.title}`);
    lines.push(entry.content.slice(0, 300));
    if (entry.tags.length > 0) {
      lines.push(`Tags: ${entry.tags.join(", ")}`);
    }
    lines.push("");

    // Increment retrieval count
    entry.retrievalCount++;
  }

  lines.push("=== END JOURNAL ===");
  return lines.join("\n");
}

/**
 * Decay importance scores over time.
 * Called periodically (e.g., once per session start).
 */
export function decayImportance(store: JournalStore): JournalStore {
  const now = Date.now();
  const entries = store.entries.map((entry) => {
    const ageDays = (now - entry.ts) / (1000 * 60 * 60 * 24);
    // Entries older than 30 days start losing importance
    if (ageDays > 30) {
      const decayFactor = Math.max(0.3, 1 - (ageDays - 30) / 180);
      return { ...entry, importance: Math.round(entry.importance * decayFactor * 10) / 10 };
    }
    return entry;
  });

  return { ...store, entries };
}

/**
 * Prune low-importance entries to keep the journal from growing unbounded.
 */
export function pruneJournal(
  store: JournalStore,
  maxEntries: number = 500,
): JournalStore {
  if (store.entries.length <= maxEntries) return store;

  // Keep human notes and high-importance entries, prune the rest
  const sorted = [...store.entries].sort((a, b) => {
    // Human notes always kept
    if (a.source === "human" && b.source !== "human") return -1;
    if (b.source === "human" && a.source !== "human") return 1;
    // Then by importance
    if (b.importance !== a.importance) return b.importance - a.importance;
    // Then by recency
    return b.ts - a.ts;
  });

  return { ...store, entries: sorted.slice(0, maxEntries) };
}

/**
 * Persist the journal to disk as JSON.
 */
export function persistJournal(store: JournalStore, filePath: string): void {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, JSON.stringify({ ...store, lastPersisted: Date.now() }, null, 2));
}

/**
 * Load a journal from disk.
 */
export function loadJournal(filePath: string): JournalStore {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw) as JournalStore;
    if (!data.entries || !Array.isArray(data.entries)) {
      return createJournalStore();
    }
    return data;
  } catch {
    return createJournalStore();
  }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function addEntry(
  store: JournalStore,
  params: Omit<JournalEntry, "id" | "ts" | "retrievalCount">,
): JournalStore {
  const entry: JournalEntry = {
    id: `journal-${crypto.randomUUID().slice(0, 8)}`,
    ts: Date.now(),
    retrievalCount: 0,
    ...params,
  };

  return {
    ...store,
    entries: [...store.entries, entry],
  };
}
