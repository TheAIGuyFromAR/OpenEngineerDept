/**
 * Coder Agent.
 *
 * The workhorse. Receives a narrowly scoped task specification from the
 * Conductor, along with relevant file contents, module context from the
 * knowledge graph, Layer 0 constraints, and exemplars.
 *
 * Produces code, tests, and documentation.
 * Multiple Coders execute in parallel during Ultra Think cycles.
 *
 * Permissions: read/write within specified scope, execute test suites.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type {
  AgentId,
  CoderOutput,
  InferenceRequest,
  InferenceResponse,
  TaskSpec,
  TestResult,
} from "../types.js";
import { buildLayer0Prompt } from "../memory/layer0-constraints.js";
import { buildLayer4Prompt } from "../memory/layer4-knowledge.js";
import {
  checkPermission,
  type PermissionGrant,
} from "../security/trust-boundary.js";

const SYSTEM_PROMPT_VARIANTS = [
  "You are a precise software engineer. Write clean, correct, minimal code. Prioritize readability and correctness over cleverness.",
  "You are a pragmatic developer. Focus on getting the implementation right with solid test coverage. Prefer simple, direct solutions.",
  "You are a defensive programmer. Think about edge cases, error handling, and failure modes. Write robust code that handles unexpected input gracefully.",
];

export type CoderConfig = {
  projectRoot: string;
  modelEndpoint: string;
  modelId: string;
};

/**
 * Execute a single Coder generation for a task.
 *
 * This is the atomic unit of work in the Ultra Think pipeline.
 * Multiple of these run in parallel with varied parameters.
 */
export async function executeCoderGeneration(params: {
  task: TaskSpec;
  config: CoderConfig;
  grant: PermissionGrant;
  temperature: number;
  systemPromptIndex: number;
  infer: (req: InferenceRequest) => Promise<InferenceResponse>;
}): Promise<CoderOutput> {
  const { task, config, grant, temperature, systemPromptIndex, infer } = params;
  const agentId: AgentId = `coder-${crypto.randomUUID().slice(0, 8)}`;
  const startTime = Date.now();

  // Read context files (with permission check)
  const fileContents = readContextFiles(task.contextFiles, config.projectRoot, grant);

  // Build the prompt
  const constraintsPrompt = buildLayer0Prompt(task.constraints);
  const knowledgePrompt = buildLayer4Prompt(task.knowledgeContext);
  const exemplarPrompt = buildExemplarPrompt(task.exemplars);
  const systemVariant = SYSTEM_PROMPT_VARIANTS[systemPromptIndex % SYSTEM_PROMPT_VARIANTS.length];

  const fullPrompt = [
    constraintsPrompt,
    knowledgePrompt,
    exemplarPrompt,
    "=== FILE CONTENTS ===",
    ...fileContents.map(({ path: p, content }) => `--- ${p} ---\n${content}`),
    "=== END FILE CONTENTS ===",
    "",
    "=== TASK ===",
    task.instructions,
    "",
    "Write scope: " + (task.writeScope.length > 0 ? task.writeScope.join(", ") : "NONE (propose files)"),
    "=== END TASK ===",
    "",
    "Respond with:",
    "1. The implementation code (in fenced code blocks with file paths)",
    "2. Tests for the implementation",
    "3. Brief documentation of what was changed and why",
  ]
    .filter(Boolean)
    .join("\n");

  const response = await infer({
    systemPrompt: systemVariant,
    prompt: fullPrompt,
    temperature,
    topP: 0.95,
    maxTokens: grant.scope.maxOutputTokens,
    thinkingMode: task.difficulty.tier >= 3,
  });

  // Parse the response into structured output
  const { code, tests, documentation } = parseCoderResponse(response.text);

  return {
    agentId,
    taskId: task.id,
    code,
    tests,
    documentation,
    thinkingTrace: response.thinkingTrace,
    temperature,
    systemPromptVariant: systemVariant,
    generationTimeMs: Date.now() - startTime,
    tokenCount: response.tokenCount,
  };
}

/**
 * Run the test suite against a Coder's output.
 */
export async function runTests(params: {
  output: CoderOutput;
  task: TaskSpec;
  config: CoderConfig;
  grant: PermissionGrant;
  exec: (command: string, cwd: string) => Promise<{ stdout: string; stderr: string; exitCode: number }>;
}): Promise<TestResult> {
  const { output, task, config, grant } = params;
  const startTime = Date.now();

  // Check execution permission
  for (const cmd of task.allowedCommands) {
    const perm = checkPermission(grant, { type: "execute", command: cmd });
    if (!perm.allowed) {
      return {
        agentId: output.agentId,
        taskId: task.id,
        passed: false,
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        failures: [{ testName: "permission-check", error: perm.reason ?? "Execution not permitted" }],
        executionTimeMs: Date.now() - startTime,
      };
    }
  }

  // Run tests
  try {
    const testCommand = task.allowedCommands.find(
      (cmd) => cmd.includes("test") || cmd.includes("vitest") || cmd.includes("jest"),
    );

    if (!testCommand) {
      return {
        agentId: output.agentId,
        taskId: task.id,
        passed: true,
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        failures: [],
        executionTimeMs: Date.now() - startTime,
      };
    }

    const result = await params.exec(testCommand, config.projectRoot);
    const parsed = parseTestOutput(result.stdout + "\n" + result.stderr, result.exitCode);

    return {
      agentId: output.agentId,
      taskId: task.id,
      ...parsed,
      executionTimeMs: Date.now() - startTime,
    };
  } catch (err) {
    return {
      agentId: output.agentId,
      taskId: task.id,
      passed: false,
      totalTests: 0,
      passedTests: 0,
      failedTests: 1,
      failures: [{ testName: "execution", error: String(err) }],
      executionTimeMs: Date.now() - startTime,
    };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readContextFiles(
  files: string[],
  projectRoot: string,
  grant: PermissionGrant,
): { path: string; content: string }[] {
  const results: { path: string; content: string }[] = [];

  for (const file of files) {
    const perm = checkPermission(grant, { type: "read", path: file });
    if (!perm.allowed) continue;

    const fullPath = path.resolve(projectRoot, file);
    // Prevent path traversal
    if (!fullPath.startsWith(path.resolve(projectRoot))) continue;

    try {
      const content = fs.readFileSync(fullPath, "utf-8");
      // Limit file size to prevent prompt stuffing
      results.push({
        path: file,
        content: content.length > 10_000 ? content.slice(0, 10_000) + "\n... (truncated)" : content,
      });
    } catch {
      // File not found — skip silently
    }
  }

  return results;
}

function buildExemplarPrompt(exemplars: TaskSpec["exemplars"]): string {
  if (exemplars.length === 0) return "";

  const lines = ["=== EXEMPLARS (best previous completions) ===", ""];
  for (const ex of exemplars) {
    lines.push(`Task: ${ex.taskDescription}`);
    lines.push(`Score: ${ex.reviewerScore}/10`);
    lines.push("```");
    lines.push(ex.implementation.slice(0, 2_000));
    lines.push("```");
    lines.push("");
  }
  lines.push("=== END EXEMPLARS ===");
  return lines.join("\n");
}

function parseCoderResponse(text: string): {
  code: string;
  tests: string;
  documentation: string;
} {
  // Extract code blocks
  const codeBlocks: string[] = [];
  const codeRegex = /```[\w]*\n([\s\S]*?)```/g;
  let match;
  while ((match = codeRegex.exec(text)) !== null) {
    codeBlocks.push(match[1].trim());
  }

  // Heuristic: first code block(s) are implementation, test-related blocks are tests
  let code = "";
  let tests = "";

  for (const block of codeBlocks) {
    if (
      block.includes("describe(") ||
      block.includes("it(") ||
      block.includes("test(") ||
      block.includes("expect(") ||
      block.includes("assert") ||
      block.includes("def test_") ||
      block.includes("@pytest")
    ) {
      tests += block + "\n\n";
    } else {
      code += block + "\n\n";
    }
  }

  // Everything outside code blocks is documentation
  const documentation = text.replace(/```[\w]*\n[\s\S]*?```/g, "").trim();

  return { code: code.trim(), tests: tests.trim(), documentation };
}

function parseTestOutput(
  output: string,
  exitCode: number,
): {
  passed: boolean;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  failures: { testName: string; error: string }[];
} {
  const failures: { testName: string; error: string }[] = [];

  // Parse vitest/jest output
  const summaryMatch = output.match(/Tests:\s+(\d+)\s+passed(?:,\s+(\d+)\s+failed)?/);
  if (summaryMatch) {
    const passedTests = parseInt(summaryMatch[1], 10);
    const failedTests = parseInt(summaryMatch[2] ?? "0", 10);
    return {
      passed: failedTests === 0 && exitCode === 0,
      totalTests: passedTests + failedTests,
      passedTests,
      failedTests,
      failures,
    };
  }

  // Fallback: just use exit code
  return {
    passed: exitCode === 0,
    totalTests: 0,
    passedTests: exitCode === 0 ? 1 : 0,
    failedTests: exitCode === 0 ? 0 : 1,
    failures: exitCode !== 0 ? [{ testName: "unknown", error: output.slice(-500) }] : [],
  };
}
