/**
 * Lifecycle Hooks — Pre/Post Execution Triggers.
 *
 * Inspired by Claude Code's hooks system. Users define shell commands
 * or callback functions that fire at specific points in the pipeline.
 *
 * Hook points:
 *   - pre-task:    Before a task is dispatched to Coder agents
 *   - post-task:   After Ultra Think completes (success or failure)
 *   - pre-write:   Before a file is written/modified
 *   - post-write:  After a file is written/modified
 *   - pre-exec:    Before a command is executed
 *   - post-exec:   After a command completes
 *   - on-escalation: When a task is escalated to human
 *   - on-constraint: When a new constraint is pinned
 *   - session-start: When the Conductor boots up
 *   - session-end:   When the Conductor shuts down
 *
 * Hooks can:
 *   - Block execution (pre-hooks return { allow: false })
 *   - Log/notify (fire-and-forget)
 *   - Transform data (pre-hooks return modified input)
 *
 * Configuration via `.conductor/hooks.json` in the project root.
 */

import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export type HookPoint =
  | "pre-task"
  | "post-task"
  | "pre-write"
  | "post-write"
  | "pre-exec"
  | "post-exec"
  | "on-escalation"
  | "on-constraint"
  | "session-start"
  | "session-end";

export type HookDefinition = {
  /** Unique ID for this hook. */
  id: string;
  /** When to fire. */
  point: HookPoint;
  /** Shell command to execute. Receives context as JSON on stdin. */
  command?: string;
  /** Inline handler (for programmatic use). */
  handler?: (context: HookContext) => Promise<HookResult>;
  /** Whether this hook can block execution. Default: false. */
  blocking: boolean;
  /** Timeout in ms. Default: 10000. */
  timeoutMs: number;
  /** Whether the hook is enabled. Default: true. */
  enabled: boolean;
  /** Optional description. */
  description?: string;
  /** File pattern filter (only fire for matching files, for write hooks). */
  filePattern?: string;
};

export type HookContext = {
  point: HookPoint;
  taskId?: string;
  taskDescription?: string;
  filePath?: string;
  command?: string;
  exitCode?: number;
  score?: number;
  escalationReason?: string;
  constraint?: string;
  metadata?: Record<string, unknown>;
};

export type HookResult = {
  /** Whether to allow the operation to proceed. */
  allow: boolean;
  /** Message to log. */
  message?: string;
  /** Modified context (for pre-hooks that transform input). */
  modifiedContext?: Partial<HookContext>;
};

export type HookRegistry = {
  hooks: HookDefinition[];
  executionLog: HookExecution[];
};

export type HookExecution = {
  hookId: string;
  point: HookPoint;
  ts: number;
  durationMs: number;
  result: HookResult;
  error?: string;
};

/**
 * Create an empty hook registry.
 */
export function createHookRegistry(): HookRegistry {
  return {
    hooks: [],
    executionLog: [],
  };
}

/**
 * Load hooks from `.conductor/hooks.json` in the project root.
 */
export function loadHooks(projectRoot: string): HookRegistry {
  const hooksFile = path.join(projectRoot, ".conductor", "hooks.json");
  const registry = createHookRegistry();

  if (!fs.existsSync(hooksFile)) return registry;

  try {
    const raw = fs.readFileSync(hooksFile, "utf-8");
    const data = JSON.parse(raw) as {
      hooks?: Array<Partial<HookDefinition>>;
    };

    if (Array.isArray(data.hooks)) {
      for (const hook of data.hooks) {
        if (!hook.point || !hook.id) continue;
        registry.hooks.push({
          id: hook.id,
          point: hook.point as HookPoint,
          command: hook.command,
          blocking: hook.blocking ?? false,
          timeoutMs: hook.timeoutMs ?? 10_000,
          enabled: hook.enabled ?? true,
          description: hook.description,
          filePattern: hook.filePattern,
        });
      }
    }
  } catch {
    // Invalid hooks file — start with empty registry
  }

  return registry;
}

/**
 * Register a hook programmatically.
 */
export function registerHook(
  registry: HookRegistry,
  hook: HookDefinition,
): HookRegistry {
  return {
    ...registry,
    hooks: [...registry.hooks, hook],
  };
}

/**
 * Remove a hook by ID.
 */
export function removeHook(
  registry: HookRegistry,
  hookId: string,
): HookRegistry {
  return {
    ...registry,
    hooks: registry.hooks.filter((h) => h.id !== hookId),
  };
}

/**
 * Execute all hooks for a given point. Returns whether to proceed.
 *
 * For blocking hooks: if ANY blocking hook returns { allow: false },
 * the operation should be aborted.
 *
 * For non-blocking hooks: they run fire-and-forget.
 */
export async function executeHooks(
  registry: HookRegistry,
  point: HookPoint,
  context: HookContext,
): Promise<{ proceed: boolean; messages: string[]; registry: HookRegistry }> {
  const matchingHooks = registry.hooks.filter(
    (h) => h.point === point && h.enabled && matchesFilePattern(h, context),
  );

  if (matchingHooks.length === 0) {
    return { proceed: true, messages: [], registry };
  }

  const messages: string[] = [];
  const newExecutions: HookExecution[] = [];
  let proceed = true;

  // Execute blocking hooks first (sequentially)
  const blockingHooks = matchingHooks.filter((h) => h.blocking);
  for (const hook of blockingHooks) {
    const startTime = Date.now();
    try {
      const result = await executeHook(hook, context);
      newExecutions.push({
        hookId: hook.id,
        point,
        ts: Date.now(),
        durationMs: Date.now() - startTime,
        result,
      });

      if (result.message) messages.push(`[${hook.id}] ${result.message}`);

      if (!result.allow) {
        proceed = false;
        messages.push(`[${hook.id}] BLOCKED operation`);
        break; // Stop on first block
      }
    } catch (err) {
      newExecutions.push({
        hookId: hook.id,
        point,
        ts: Date.now(),
        durationMs: Date.now() - startTime,
        result: { allow: true },
        error: String(err),
      });
    }
  }

  // Execute non-blocking hooks (parallel, fire-and-forget)
  if (proceed) {
    const nonBlockingHooks = matchingHooks.filter((h) => !h.blocking);
    const nonBlockingPromises = nonBlockingHooks.map(async (hook) => {
      const startTime = Date.now();
      try {
        const result = await executeHook(hook, context);
        newExecutions.push({
          hookId: hook.id,
          point,
          ts: Date.now(),
          durationMs: Date.now() - startTime,
          result,
        });
        if (result.message) messages.push(`[${hook.id}] ${result.message}`);
      } catch (err) {
        newExecutions.push({
          hookId: hook.id,
          point,
          ts: Date.now(),
          durationMs: Date.now() - startTime,
          result: { allow: true },
          error: String(err),
        });
      }
    });

    // Don't await — fire and forget
    Promise.allSettled(nonBlockingPromises);
  }

  return {
    proceed,
    messages,
    registry: {
      ...registry,
      executionLog: [...registry.executionLog, ...newExecutions],
    },
  };
}

/**
 * Get recent hook execution log.
 */
export function getRecentExecutions(
  registry: HookRegistry,
  limit: number = 20,
): HookExecution[] {
  return registry.executionLog.slice(-limit);
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

async function executeHook(
  hook: HookDefinition,
  context: HookContext,
): Promise<HookResult> {
  // Inline handler takes precedence
  if (hook.handler) {
    return hook.handler(context);
  }

  // Shell command execution
  if (hook.command) {
    return executeShellHook(hook.command, context, hook.timeoutMs);
  }

  return { allow: true, message: "No handler or command defined" };
}

function executeShellHook(
  command: string,
  context: HookContext,
  timeoutMs: number,
): Promise<HookResult> {
  return new Promise((resolve) => {
    const contextJson = JSON.stringify(context);

    // Split command into executable and args
    const parts = command.split(/\s+/);
    const executable = parts[0];
    const args = parts.slice(1);

    const proc = execFile(executable, args, {
      timeout: timeoutMs,
      env: {
        ...process.env,
        CONDUCTOR_HOOK_POINT: context.point,
        CONDUCTOR_TASK_ID: context.taskId ?? "",
        CONDUCTOR_FILE_PATH: context.filePath ?? "",
      },
    }, (error, stdout, stderr) => {
      if (error) {
        // Timeout or execution error — allow by default (fail-open for non-blocking)
        resolve({
          allow: true,
          message: `Hook error: ${error.message}`,
        });
        return;
      }

      // Try to parse stdout as JSON result
      try {
        const result = JSON.parse(stdout.trim()) as Partial<HookResult>;
        resolve({
          allow: result.allow !== false,
          message: result.message ?? stderr.trim() || undefined,
          modifiedContext: result.modifiedContext,
        });
      } catch {
        // Non-JSON output — treat as success message
        resolve({
          allow: true,
          message: stdout.trim() || stderr.trim() || undefined,
        });
      }
    });

    // Send context on stdin
    if (proc.stdin) {
      proc.stdin.write(contextJson);
      proc.stdin.end();
    }
  });
}

function matchesFilePattern(hook: HookDefinition, context: HookContext): boolean {
  if (!hook.filePattern) return true;
  if (!context.filePath) return true;

  // Simple glob matching
  const pattern = hook.filePattern;
  if (pattern === "**/*") return true;
  if (context.filePath.endsWith(pattern)) return true;

  const regex = new RegExp(
    "^" +
      pattern
        .replace(/[.+^${}()|[\]\\]/g, "\\$&")
        .replace(/\*\*/g, "DOUBLESTAR")
        .replace(/\*/g, "[^/]*")
        .replace(/DOUBLESTAR/g, ".*") +
      "$",
  );

  return regex.test(context.filePath);
}
