/**
 * Layer 0: Pinned Constraints.
 *
 * Constitutional rules that never compress. Included verbatim in every
 * Conductor and Coder prompt. Lives as a _constraints.md file in the
 * Obsidian vault, editable by the human.
 *
 * Typical size: 500–2,000 tokens.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { ConstraintEntry } from "../types.js";

const CONSTRAINTS_FILENAME = "_constraints.md";

export type Layer0Config = {
  /** Path to the constraints file (or directory containing it). */
  constraintsPath: string;
};

/**
 * Parse a constraints markdown file into structured entries.
 *
 * Expected format:
 * ```markdown
 * ## Style
 * - Use camelCase for variables
 * - Maximum file size: 500 lines
 *
 * ## Architecture
 * - All API endpoints must use Zod validation
 * ```
 */
export function parseConstraintsFile(content: string): ConstraintEntry[] {
  const entries: ConstraintEntry[] = [];
  let currentCategory: ConstraintEntry["category"] = "general";

  const categoryMap: Record<string, ConstraintEntry["category"]> = {
    style: "style",
    architecture: "architecture",
    testing: "testing",
    security: "security",
    api: "api",
    general: "general",
  };

  for (const line of content.split("\n")) {
    const trimmed = line.trim();

    // Detect category headers
    const headerMatch = trimmed.match(/^#{1,3}\s+(.+)$/);
    if (headerMatch) {
      const headerText = headerMatch[1].toLowerCase().trim();
      for (const [key, value] of Object.entries(categoryMap)) {
        if (headerText.includes(key)) {
          currentCategory = value;
          break;
        }
      }
      continue;
    }

    // Parse constraint rules (lines starting with - or *)
    const ruleMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ruleMatch) {
      const rule = ruleMatch[1].trim();
      if (rule.length > 0) {
        entries.push({
          id: crypto.createHash("sha256").update(rule).digest("hex").slice(0, 12),
          rule,
          category: currentCategory,
          addedAt: Date.now(),
          source: "human",
        });
      }
    }
  }

  return entries;
}

export function loadConstraints(config: Layer0Config): ConstraintEntry[] {
  const filePath = config.constraintsPath.endsWith(".md")
    ? config.constraintsPath
    : path.join(config.constraintsPath, CONSTRAINTS_FILENAME);

  if (!fs.existsSync(filePath)) {
    return [];
  }

  const content = fs.readFileSync(filePath, "utf-8");
  return parseConstraintsFile(content);
}

export function addConstraint(
  constraints: ConstraintEntry[],
  rule: string,
  category: ConstraintEntry["category"],
  source: ConstraintEntry["source"] = "human",
): ConstraintEntry[] {
  const entry: ConstraintEntry = {
    id: crypto.createHash("sha256").update(rule).digest("hex").slice(0, 12),
    rule,
    category,
    addedAt: Date.now(),
    source,
  };

  // Deduplicate by rule content
  if (constraints.some((c) => c.id === entry.id)) {
    return constraints;
  }

  return [...constraints, entry];
}

export function removeConstraint(
  constraints: ConstraintEntry[],
  id: string,
): ConstraintEntry[] {
  return constraints.filter((c) => c.id !== id);
}

/**
 * Serialize constraints back to markdown format for persistence.
 */
export function serializeConstraints(constraints: ConstraintEntry[]): string {
  const grouped = new Map<string, ConstraintEntry[]>();
  for (const entry of constraints) {
    const list = grouped.get(entry.category) ?? [];
    list.push(entry);
    grouped.set(entry.category, list);
  }

  const sections: string[] = ["# Project Constraints", ""];
  const categoryOrder: ConstraintEntry["category"][] = [
    "architecture",
    "style",
    "testing",
    "security",
    "api",
    "general",
  ];

  for (const category of categoryOrder) {
    const entries = grouped.get(category);
    if (!entries || entries.length === 0) continue;

    sections.push(`## ${category.charAt(0).toUpperCase() + category.slice(1)}`);
    for (const entry of entries) {
      sections.push(`- ${entry.rule}`);
    }
    sections.push("");
  }

  return sections.join("\n");
}

/**
 * Build the prompt fragment for Layer 0 inclusion.
 */
export function buildLayer0Prompt(constraints: ConstraintEntry[]): string {
  if (constraints.length === 0) {
    return "";
  }

  const lines = [
    "=== PINNED CONSTRAINTS (Layer 0) ===",
    "These rules are absolute. Never violate them.",
    "",
  ];

  for (const entry of constraints) {
    lines.push(`[${entry.category.toUpperCase()}] ${entry.rule}`);
  }

  lines.push("=== END CONSTRAINTS ===");
  return lines.join("\n");
}
