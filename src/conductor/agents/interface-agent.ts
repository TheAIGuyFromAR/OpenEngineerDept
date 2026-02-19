/**
 * Interface Agent.
 *
 * The ONLY agent that receives raw user input. Near-zero permissions.
 * Parses Obsidian frontmatter or OpenAI-compatible chat messages,
 * classifies intent, sanitizes input, and produces structured handoff
 * messages for the Conductor.
 *
 * Cannot read, write, or execute anything in the project.
 */

import crypto from "node:crypto";
import type { HandoffMessage } from "../types.js";

// Reuse the existing security infrastructure from the Maistro hardening
import { detectSuspiciousPatterns } from "../../security/external-content.js";

/**
 * Obsidian frontmatter schema for work orders.
 *
 * Example:
 * ```yaml
 * ---
 * type: task
 * priority: high
 * category: feature
 * files:
 *   - src/auth/login.ts
 *   - src/auth/session.ts
 * ---
 * Implement rate limiting on the login endpoint...
 * ```
 */
export type ObsidianFrontmatter = {
  type?: string;
  priority?: string;
  category?: string;
  files?: string[];
  references?: string[];
};

/**
 * Parse Obsidian markdown with frontmatter into a structured handoff message.
 * This is the primary input path for asynchronous task dispatch.
 */
export function parseObsidianInput(
  markdown: string,
  filename: string,
): HandoffMessage {
  const { frontmatter, body } = extractFrontmatter(markdown);
  const sanitized = sanitizeContent(body);
  const intent = classifyIntent(frontmatter, sanitized.content);
  const flags = detectSuspiciousPatterns(body);

  return {
    id: crypto.randomUUID(),
    source: "obsidian",
    intent,
    content: sanitized.content,
    metadata: {
      priority: normalizePriority(frontmatter.priority),
      category: frontmatter.category?.trim(),
      targetFiles: sanitizeFilePaths(frontmatter.files),
      referencedTasks: frontmatter.references,
    },
    sanitizationFlags: [...flags, ...sanitized.flags],
    ts: Date.now(),
  };
}

/**
 * Parse an OpenWebUI/OpenAI-compatible chat message into a handoff.
 * This is the input path for real-time interactive conversation.
 */
export function parseOpenWebUIInput(
  message: { role: string; content: string },
): HandoffMessage {
  if (message.role !== "user") {
    return {
      id: crypto.randomUUID(),
      source: "openwebui",
      intent: "clarification",
      content: "",
      metadata: {},
      sanitizationFlags: ["non-user-role-ignored"],
      ts: Date.now(),
    };
  }

  const sanitized = sanitizeContent(message.content);
  const intent = classifyIntent({}, sanitized.content);
  const flags = detectSuspiciousPatterns(message.content);

  return {
    id: crypto.randomUUID(),
    source: "openwebui",
    intent,
    content: sanitized.content,
    metadata: {
      targetFiles: extractFileReferences(sanitized.content),
    },
    sanitizationFlags: [...flags, ...sanitized.flags],
    ts: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function extractFrontmatter(markdown: string): {
  frontmatter: ObsidianFrontmatter;
  body: string;
} {
  const fmMatch = markdown.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!fmMatch) {
    return { frontmatter: {}, body: markdown };
  }

  const fmRaw = fmMatch[1];
  const body = fmMatch[2].trim();

  // Simple YAML-like parsing (no dependency on yaml parser)
  const frontmatter: ObsidianFrontmatter = {};
  let currentArrayKey: string | null = null;
  const currentArray: string[] = [];

  for (const line of fmRaw.split("\n")) {
    const trimmed = line.trim();

    // Array item continuation
    if (trimmed.startsWith("- ") && currentArrayKey) {
      currentArray.push(trimmed.slice(2).trim());
      continue;
    }

    // Flush previous array
    if (currentArrayKey && currentArray.length > 0) {
      (frontmatter as Record<string, unknown>)[currentArrayKey] = [...currentArray];
      currentArray.length = 0;
      currentArrayKey = null;
    }

    const kvMatch = trimmed.match(/^(\w+):\s*(.*)$/);
    if (kvMatch) {
      const key = kvMatch[1];
      const value = kvMatch[2].trim();

      if (value === "") {
        // Array indicator (value on next lines)
        currentArrayKey = key;
      } else {
        (frontmatter as Record<string, unknown>)[key] = value;
      }
    }
  }

  // Flush last array
  if (currentArrayKey && currentArray.length > 0) {
    (frontmatter as Record<string, unknown>)[currentArrayKey] = [...currentArray];
  }

  return { frontmatter, body };
}

function sanitizeContent(raw: string): { content: string; flags: string[] } {
  const flags: string[] = [];
  let content = raw;

  // Strip any system/assistant role markers that could be injected
  const roleInjection = /\[(system|assistant)\]:/gi;
  if (roleInjection.test(content)) {
    content = content.replace(roleInjection, "[SANITIZED]:");
    flags.push("role-injection-sanitized");
  }

  // Strip XML-style system tags
  const systemTags = /<\/?system>/gi;
  if (systemTags.test(content)) {
    content = content.replace(systemTags, "");
    flags.push("system-tags-stripped");
  }

  // Normalize Unicode to prevent homoglyph attacks
  content = content.normalize("NFKC");

  // Strip zero-width characters
  content = content.replace(
    /[\u200B\u200C\u200D\u200E\u200F\u2060\u2061\u2062\u2063\u2064\uFEFF\u00AD\u034F\u061C]/g,
    "",
  );

  // Limit content length
  if (content.length > 50_000) {
    content = content.slice(0, 50_000);
    flags.push("content-truncated");
  }

  return { content: content.trim(), flags };
}

function classifyIntent(
  frontmatter: ObsidianFrontmatter,
  content: string,
): HandoffMessage["intent"] {
  // Frontmatter type takes precedence
  if (frontmatter.type) {
    const typeMap: Record<string, HandoffMessage["intent"]> = {
      task: "task",
      question: "question",
      clarification: "clarification",
      feedback: "feedback",
      constraint: "constraint-update",
    };
    const mapped = typeMap[frontmatter.type.toLowerCase()];
    if (mapped) return mapped;
  }

  // Heuristic classification from content
  const lower = content.toLowerCase();

  if (
    lower.startsWith("add constraint") ||
    lower.startsWith("new rule") ||
    lower.startsWith("always ") ||
    lower.startsWith("never ")
  ) {
    return "constraint-update";
  }

  if (
    lower.includes("?") &&
    (lower.startsWith("what") ||
      lower.startsWith("how") ||
      lower.startsWith("why") ||
      lower.startsWith("where") ||
      lower.startsWith("when") ||
      lower.startsWith("can") ||
      lower.startsWith("does") ||
      lower.startsWith("is "))
  ) {
    return "question";
  }

  if (
    lower.startsWith("good") ||
    lower.startsWith("bad") ||
    lower.startsWith("the issue") ||
    lower.startsWith("actually") ||
    lower.startsWith("no,") ||
    lower.startsWith("yes,")
  ) {
    return "feedback";
  }

  // Default to task
  return "task";
}

function normalizePriority(raw?: string): HandoffMessage["metadata"]["priority"] {
  if (!raw) return undefined;
  const lower = raw.toLowerCase().trim();
  if (["low", "normal", "high", "urgent"].includes(lower)) {
    return lower as "low" | "normal" | "high" | "urgent";
  }
  return "normal";
}

function sanitizeFilePaths(files?: string[]): string[] | undefined {
  if (!files || !Array.isArray(files)) return undefined;
  return files
    .filter((f) => typeof f === "string")
    .map((f) => f.trim())
    .filter((f) => !f.includes("..") && !f.startsWith("/"))
    .filter((f) => f.length > 0 && f.length < 500);
}

function extractFileReferences(content: string): string[] | undefined {
  // Extract file paths mentioned in backticks
  const matches = content.match(/`([^`]+\.[a-z]+)`/g);
  if (!matches) return undefined;
  return sanitizeFilePaths(
    matches.map((m) => m.replace(/`/g, "")),
  );
}
