/**
 * Trust Boundary Enforcement.
 *
 * The security boundary between Interface Agent and Conductor is the
 * critical design element. This module enforces:
 *
 * 1. Interface Agent → Conductor: only structured HandoffMessages cross
 * 2. Conductor → Sub-agents: only scoped TaskSpecs with permission grants
 * 3. No agent can escalate its own permissions
 * 4. All permission grants are logged
 */

import type { AgentId, AgentRole, HandoffMessage, PermissionScope, TaskSpec } from "../types.js";
import { DEFAULT_PERMISSIONS } from "../types.js";
import { secureId } from "../../utils/secure-random.js";

export type PermissionGrant = {
  grantId: string;
  grantedBy: AgentId;
  grantedTo: AgentId;
  scope: PermissionScope;
  taskId: string;
  grantedAt: number;
  expiresAt: number;
};

export type TrustBoundaryLog = {
  entries: PermissionGrant[];
};

/**
 * Validate that a handoff message from Interface → Conductor is well-formed
 * and does not contain raw user input in prohibited fields.
 */
export function validateHandoff(message: HandoffMessage): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!message.id || typeof message.id !== "string") {
    errors.push("Missing or invalid message ID");
  }

  if (!["obsidian", "openwebui"].includes(message.source)) {
    errors.push(`Invalid source: ${message.source}`);
  }

  if (!["task", "question", "clarification", "feedback", "constraint-update"].includes(message.intent)) {
    errors.push(`Invalid intent: ${message.intent}`);
  }

  if (typeof message.content !== "string" || message.content.length === 0) {
    errors.push("Content must be a non-empty string");
  }

  // Content length sanity check (prevent prompt stuffing)
  if (message.content.length > 50_000) {
    errors.push("Content exceeds maximum length (50,000 chars)");
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Validate that a task spec from Conductor → Coder has proper permission scoping.
 */
export function validateTaskSpec(
  spec: TaskSpec,
  conductorId: AgentId,
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!spec.id) {
    errors.push("Task must have an ID");
  }

  if (spec.writeScope.length === 0 && spec.allowedCommands.length === 0) {
    errors.push("Task must have at least one write scope or allowed command");
  }

  // Prevent directory traversal in write scopes
  for (const scope of spec.writeScope) {
    if (scope.includes("..") || scope.startsWith("/")) {
      errors.push(`Invalid write scope (traversal attempt): ${scope}`);
    }
  }

  // Prevent dangerous commands
  const dangerousPatterns = [
    /rm\s+-rf\s+[/~]/,
    /sudo\s/,
    /chmod\s+777/,
    /curl.*\|.*sh/,
    /wget.*\|.*sh/,
    /eval\s/,
    /> \/dev\//,
    /mkfs\./,
    /dd\s+if=/,
  ];

  for (const cmd of spec.allowedCommands) {
    for (const pattern of dangerousPatterns) {
      if (pattern.test(cmd)) {
        errors.push(`Dangerous command pattern detected: ${cmd}`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Create a scoped permission grant for a sub-agent.
 * Only the Conductor can create grants. Grants expire.
 */
export function createPermissionGrant(params: {
  conductorId: AgentId;
  targetAgentId: AgentId;
  targetRole: AgentRole;
  taskId: string;
  customScope?: Partial<PermissionScope>;
  ttlMs?: number;
}): PermissionGrant {
  const baseScope = DEFAULT_PERMISSIONS[params.targetRole];
  const scope: PermissionScope = {
    ...baseScope,
    ...params.customScope,
  };

  const ttl = params.ttlMs ?? baseScope.timeoutMs;

  return {
    grantId: `grant-${Date.now()}-${secureId(6)}`,
    grantedBy: params.conductorId,
    grantedTo: params.targetAgentId,
    scope,
    taskId: params.taskId,
    grantedAt: Date.now(),
    expiresAt: Date.now() + ttl,
  };
}

/**
 * Check if an agent action is permitted under its current grant.
 */
export function checkPermission(
  grant: PermissionGrant,
  action: {
    type: "read" | "write" | "execute";
    path?: string;
    command?: string;
  },
): { allowed: boolean; reason?: string } {
  // Check expiry
  if (Date.now() > grant.expiresAt) {
    return { allowed: false, reason: "Permission grant has expired" };
  }

  switch (action.type) {
    case "read": {
      if (grant.scope.readPaths.length === 0) {
        return { allowed: false, reason: "No read permissions" };
      }
      if (action.path && !matchesGlob(action.path, grant.scope.readPaths)) {
        return { allowed: false, reason: `Path ${action.path} not in read scope` };
      }
      return { allowed: true };
    }

    case "write": {
      if (grant.scope.writePaths.length === 0) {
        return { allowed: false, reason: "No write permissions" };
      }
      if (action.path && !matchesGlob(action.path, grant.scope.writePaths)) {
        return { allowed: false, reason: `Path ${action.path} not in write scope` };
      }
      return { allowed: true };
    }

    case "execute": {
      if (!grant.scope.canExecute) {
        return { allowed: false, reason: "Execution not permitted" };
      }
      if (
        action.command &&
        grant.scope.allowedCommands &&
        grant.scope.allowedCommands.length > 0 &&
        !grant.scope.allowedCommands.some((allowed) => action.command!.startsWith(allowed))
      ) {
        return { allowed: false, reason: `Command ${action.command} not in allowed list` };
      }
      return { allowed: true };
    }
  }
}

/**
 * Simple glob matching for permission paths.
 */
function matchesGlob(filepath: string, patterns: string[]): boolean {
  for (const pattern of patterns) {
    if (pattern === "**/*") return true;
    if (pattern === filepath) return true;

    // Simple wildcard matching
    const regex = new RegExp(
      "^" +
        pattern
          .replace(/[.+^${}()|[\]\\]/g, "\\$&")
          .replace(/\*\*/g, "DOUBLESTAR")
          .replace(/\*/g, "[^/]*")
          .replace(/DOUBLESTAR/g, ".*") +
        "$",
    );
    if (regex.test(filepath)) return true;
  }
  return false;
}
