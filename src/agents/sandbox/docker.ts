import { spawn } from "node:child_process";

type ExecDockerRawOptions = {
  allowFailure?: boolean;
  input?: Buffer | string;
  signal?: AbortSignal;
};

export type ExecDockerRawResult = {
  stdout: Buffer;
  stderr: Buffer;
  code: number;
};

type ExecDockerRawError = Error & {
  code: number;
  stdout: Buffer;
  stderr: Buffer;
};

function createAbortError(): Error {
  const err = new Error("Aborted");
  err.name = "AbortError";
  return err;
}

export function execDockerRaw(
  args: string[],
  opts?: ExecDockerRawOptions,
): Promise<ExecDockerRawResult> {
  return new Promise<ExecDockerRawResult>((resolve, reject) => {
    const child = spawn("docker", args, {
      stdio: ["pipe", "pipe", "pipe"],
    });
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    let aborted = false;

    const signal = opts?.signal;
    const handleAbort = () => {
      if (aborted) {
        return;
      }
      aborted = true;
      child.kill("SIGTERM");
    };
    if (signal) {
      if (signal.aborted) {
        handleAbort();
      } else {
        signal.addEventListener("abort", handleAbort);
      }
    }

    child.stdout?.on("data", (chunk) => {
      stdoutChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    child.stderr?.on("data", (chunk) => {
      stderrChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });

    child.on("error", (error) => {
      if (signal) {
        signal.removeEventListener("abort", handleAbort);
      }
      reject(error);
    });

    child.on("close", (code) => {
      if (signal) {
        signal.removeEventListener("abort", handleAbort);
      }
      const stdout = Buffer.concat(stdoutChunks);
      const stderr = Buffer.concat(stderrChunks);
      if (aborted || signal?.aborted) {
        reject(createAbortError());
        return;
      }
      const exitCode = code ?? 0;
      if (exitCode !== 0 && !opts?.allowFailure) {
        const message = stderr.length > 0 ? stderr.toString("utf8").trim() : "";
        const error: ExecDockerRawError = Object.assign(
          new Error(message || `docker ${args.join(" ")} failed`),
          {
            code: exitCode,
            stdout,
            stderr,
          },
        );
        reject(error);
        return;
      }
      resolve({ stdout, stderr, code: exitCode });
    });

    const stdin = child.stdin;
    if (stdin) {
      if (opts?.input !== undefined) {
        stdin.end(opts.input);
      } else {
        stdin.end();
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Environment variable sanitization for sandbox containers
// ---------------------------------------------------------------------------

/** Exact env var names that must never be forwarded into a sandbox. */
const SENSITIVE_ENV_EXACT = new Set([
  "ANTHROPIC_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_ORG_ID",
  "GEMINI_API_KEY",
  "OPENROUTER_API_KEY",
  "AZURE_OPENAI_API_KEY",
  "AZURE_API_KEY",
  "COHERE_API_KEY",
  "HUGGINGFACE_API_KEY",
  "HF_TOKEN",
  "REPLICATE_API_TOKEN",
  "TOGETHER_API_KEY",
  "GROQ_API_KEY",
  "MISTRAL_API_KEY",
  "PERPLEXITY_API_KEY",
  "AI_GATEWAY_API_KEY",
  "FIRECRAWL_API_KEY",
  "BRAVE_API_KEY",
  "ELEVENLABS_API_KEY",
  "XI_API_KEY",
  "DEEPGRAM_API_KEY",
  "ZAI_API_KEY",
  "MINIMAX_API_KEY",
  "SYNTHETIC_API_KEY",
  "TELEGRAM_BOT_TOKEN",
  "DISCORD_BOT_TOKEN",
  "SLACK_BOT_TOKEN",
  "SLACK_APP_TOKEN",
  "MATTERMOST_BOT_TOKEN",
  "ZALO_BOT_TOKEN",
  "GITHUB_TOKEN",
  "GH_TOKEN",
  "GITLAB_TOKEN",
  "NPM_TOKEN",
  "NODE_AUTH_TOKEN",
  "DOCKER_AUTH_TOKEN",
  "DOCKER_PASSWORD",
  "DATABASE_URL",
  "REDIS_URL",
  "MONGO_URI",
  "SENTRY_DSN",
  "DATADOG_API_KEY",
  "DD_API_KEY",
  "CLOUDFLARE_API_TOKEN",
  "CLOUDFLARE_API_KEY",
  "VERCEL_TOKEN",
  "NETLIFY_AUTH_TOKEN",
  "HEROKU_API_KEY",
  "DIGITALOCEAN_TOKEN",
  "LINODE_TOKEN",
  "VULTR_API_KEY",
  "FLY_API_TOKEN",
  "NODE_OPTIONS",
  "NODE_EXTRA_CA_CERTS",
  "JAVA_TOOL_OPTIONS",
  "_JAVA_OPTIONS",
]);

/** Suffix patterns that indicate a secret. Matched case-insensitively. */
const SENSITIVE_ENV_SUFFIXES = [
  "_SECRET",
  "_TOKEN",
  "_PASSWORD",
  "_PASS",
  "_KEY",
  "_API_KEY",
  "_PRIVATE_KEY",
  "_CREDENTIALS",
  "_AUTH",
];

/** Prefix patterns that indicate an entire namespace of secrets. */
const SENSITIVE_ENV_PREFIXES = [
  "AWS_",
  "SSH_",
  "GPG_",
  "VAULT_",
  "MAISTRO_GATEWAY_",
  "NPM_CONFIG_",
];

/**
 * Returns true if the given env var name matches a known sensitive pattern.
 * The check is case-insensitive for suffix/prefix matching.
 */
function isSensitiveEnvVar(name: string): boolean {
  const upper = name.toUpperCase();
  if (SENSITIVE_ENV_EXACT.has(upper)) {
    return true;
  }
  for (const prefix of SENSITIVE_ENV_PREFIXES) {
    if (upper.startsWith(prefix)) {
      return true;
    }
  }
  for (const suffix of SENSITIVE_ENV_SUFFIXES) {
    if (upper.endsWith(suffix)) {
      return true;
    }
  }
  return false;
}

/**
 * Filter env vars for sandbox containers. Blocks known-sensitive vars unless
 * they appear in the explicit allowlist.
 */
export function sanitizeSandboxEnv(
  env: Record<string, string>,
  allowlist?: Set<string>,
): { sanitized: Record<string, string>; blocked: string[] } {
  const sanitized: Record<string, string> = {};
  const blocked: string[] = [];
  for (const [key, value] of Object.entries(env)) {
    if (isSensitiveEnvVar(key) && !allowlist?.has(key)) {
      blocked.push(key);
    } else {
      sanitized[key] = value;
    }
  }
  return { sanitized, blocked };
}

import { formatCliCommand } from "../../cli/command-format.js";
import { defaultRuntime } from "../../runtime.js";
import { computeSandboxConfigHash } from "./config-hash.js";
import { DEFAULT_SANDBOX_IMAGE, SANDBOX_AGENT_WORKSPACE_MOUNT } from "./constants.js";
import { readRegistry, updateRegistry } from "./registry.js";
import { resolveSandboxAgentId, resolveSandboxScopeKey, slugifySessionKey } from "./shared.js";
import type { SandboxConfig, SandboxDockerConfig, SandboxWorkspaceAccess } from "./types.js";
import { validateSandboxSecurity } from "./validate-sandbox-security.js";

const HOT_CONTAINER_WINDOW_MS = 5 * 60 * 1000;

export type ExecDockerOptions = ExecDockerRawOptions;

export async function execDocker(args: string[], opts?: ExecDockerOptions) {
  const result = await execDockerRaw(args, opts);
  return {
    stdout: result.stdout.toString("utf8"),
    stderr: result.stderr.toString("utf8"),
    code: result.code,
  };
}

export async function readDockerContainerLabel(
  containerName: string,
  label: string,
): Promise<string | null> {
  const result = await execDocker(
    ["inspect", "-f", `{{ index .Config.Labels "${label}" }}`, containerName],
    { allowFailure: true },
  );
  if (result.code !== 0) {
    return null;
  }
  const raw = result.stdout.trim();
  if (!raw || raw === "<no value>") {
    return null;
  }
  return raw;
}

export async function readDockerPort(containerName: string, port: number) {
  const result = await execDocker(["port", containerName, `${port}/tcp`], {
    allowFailure: true,
  });
  if (result.code !== 0) {
    return null;
  }
  const line = result.stdout.trim().split(/\r?\n/)[0] ?? "";
  const match = line.match(/:(\d+)\s*$/);
  if (!match) {
    return null;
  }
  const mapped = Number.parseInt(match[1] ?? "", 10);
  return Number.isFinite(mapped) ? mapped : null;
}

async function dockerImageExists(image: string) {
  const result = await execDocker(["image", "inspect", image], {
    allowFailure: true,
  });
  if (result.code === 0) {
    return true;
  }
  const stderr = result.stderr.trim();
  if (stderr.includes("No such image")) {
    return false;
  }
  throw new Error(`Failed to inspect sandbox image: ${stderr}`);
}

export async function ensureDockerImage(image: string) {
  const exists = await dockerImageExists(image);
  if (exists) {
    return;
  }
  if (image === DEFAULT_SANDBOX_IMAGE) {
    await execDocker(["pull", "debian:bookworm-slim"]);
    await execDocker(["tag", "debian:bookworm-slim", DEFAULT_SANDBOX_IMAGE]);
    return;
  }
  throw new Error(`Sandbox image not found: ${image}. Build or pull it first.`);
}

export async function dockerContainerState(name: string) {
  const result = await execDocker(["inspect", "-f", "{{.State.Running}}", name], {
    allowFailure: true,
  });
  if (result.code !== 0) {
    return { exists: false, running: false };
  }
  return { exists: true, running: result.stdout.trim() === "true" };
}

function normalizeDockerLimit(value?: string | number) {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : undefined;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

function formatUlimitValue(
  name: string,
  value: string | number | { soft?: number; hard?: number },
) {
  if (!name.trim()) {
    return null;
  }
  if (typeof value === "number" || typeof value === "string") {
    const raw = String(value).trim();
    return raw ? `${name}=${raw}` : null;
  }
  const soft = typeof value.soft === "number" ? Math.max(0, value.soft) : undefined;
  const hard = typeof value.hard === "number" ? Math.max(0, value.hard) : undefined;
  if (soft === undefined && hard === undefined) {
    return null;
  }
  if (soft === undefined) {
    return `${name}=${hard}`;
  }
  if (hard === undefined) {
    return `${name}=${soft}`;
  }
  return `${name}=${soft}:${hard}`;
}

export function buildSandboxCreateArgs(params: {
  name: string;
  cfg: SandboxDockerConfig;
  scopeKey: string;
  createdAtMs?: number;
  labels?: Record<string, string>;
  configHash?: string;
}) {
  // Runtime security validation: blocks dangerous bind mounts, network modes, and profiles.
  validateSandboxSecurity(params.cfg);

  const createdAtMs = params.createdAtMs ?? Date.now();
  const args = ["create", "--name", params.name];
  args.push("--label", "maistro.sandbox=1");
  args.push("--label", `maistro.sessionKey=${params.scopeKey}`);
  args.push("--label", `maistro.createdAtMs=${createdAtMs}`);
  if (params.configHash) {
    args.push("--label", `maistro.configHash=${params.configHash}`);
  }
  for (const [key, value] of Object.entries(params.labels ?? {})) {
    if (key && value) {
      args.push("--label", `${key}=${value}`);
    }
  }
  if (params.cfg.readOnlyRoot) {
    args.push("--read-only");
  }
  for (const entry of params.cfg.tmpfs) {
    args.push("--tmpfs", entry);
  }
  if (params.cfg.network) {
    args.push("--network", params.cfg.network);
  }
  if (params.cfg.user) {
    args.push("--user", params.cfg.user);
  }
  // Sanitize env vars: block known secrets from leaking into the sandbox.
  const rawEnv = params.cfg.env ?? {};
  const { sanitized: safeEnv, blocked } = sanitizeSandboxEnv(
    rawEnv,
    params.cfg.envAllowlist ? new Set(params.cfg.envAllowlist) : undefined,
  );
  if (blocked.length > 0) {
    defaultRuntime.log(
      `Sandbox env: blocked ${blocked.length} sensitive var(s): ${blocked.join(", ")}`,
    );
  }
  for (const [key, value] of Object.entries(safeEnv)) {
    if (!key.trim()) {
      continue;
    }
    args.push("--env", key + "=" + value);
  }
  for (const cap of params.cfg.capDrop) {
    args.push("--cap-drop", cap);
  }
  args.push("--security-opt", "no-new-privileges");
  if (params.cfg.seccompProfile) {
    args.push("--security-opt", `seccomp=${params.cfg.seccompProfile}`);
  }
  if (params.cfg.apparmorProfile) {
    args.push("--security-opt", `apparmor=${params.cfg.apparmorProfile}`);
  }
  for (const entry of params.cfg.dns ?? []) {
    if (entry.trim()) {
      args.push("--dns", entry);
    }
  }
  for (const entry of params.cfg.extraHosts ?? []) {
    if (entry.trim()) {
      args.push("--add-host", entry);
    }
  }
  if (typeof params.cfg.pidsLimit === "number" && params.cfg.pidsLimit > 0) {
    args.push("--pids-limit", String(params.cfg.pidsLimit));
  }
  const memory = normalizeDockerLimit(params.cfg.memory);
  if (memory) {
    args.push("--memory", memory);
  }
  const memorySwap = normalizeDockerLimit(params.cfg.memorySwap);
  if (memorySwap) {
    args.push("--memory-swap", memorySwap);
  }
  if (typeof params.cfg.cpus === "number" && params.cfg.cpus > 0) {
    args.push("--cpus", String(params.cfg.cpus));
  }
  for (const [name, value] of Object.entries(params.cfg.ulimits ?? {})) {
    const formatted = formatUlimitValue(name, value);
    if (formatted) {
      args.push("--ulimit", formatted);
    }
  }
  if (params.cfg.binds?.length) {
    for (const bind of params.cfg.binds) {
      args.push("-v", bind);
    }
  }
  return args;
}

async function createSandboxContainer(params: {
  name: string;
  cfg: SandboxDockerConfig;
  workspaceDir: string;
  workspaceAccess: SandboxWorkspaceAccess;
  agentWorkspaceDir: string;
  scopeKey: string;
  configHash?: string;
}) {
  const { name, cfg, workspaceDir, scopeKey } = params;
  await ensureDockerImage(cfg.image);

  const args = buildSandboxCreateArgs({
    name,
    cfg,
    scopeKey,
    configHash: params.configHash,
  });
  args.push("--workdir", cfg.workdir);
  const mainMountSuffix =
    params.workspaceAccess === "ro" && workspaceDir === params.agentWorkspaceDir ? ":ro" : "";
  args.push("-v", `${workspaceDir}:${cfg.workdir}${mainMountSuffix}`);
  if (params.workspaceAccess !== "none" && workspaceDir !== params.agentWorkspaceDir) {
    const agentMountSuffix = params.workspaceAccess === "ro" ? ":ro" : "";
    args.push(
      "-v",
      `${params.agentWorkspaceDir}:${SANDBOX_AGENT_WORKSPACE_MOUNT}${agentMountSuffix}`,
    );
  }
  args.push(cfg.image, "sleep", "infinity");

  await execDocker(args);
  await execDocker(["start", name]);

  if (cfg.setupCommand?.trim()) {
    await execDocker(["exec", "-i", name, "sh", "-lc", cfg.setupCommand]);
  }
}

async function readContainerConfigHash(containerName: string): Promise<string | null> {
  return await readDockerContainerLabel(containerName, "maistro.configHash");
}

function formatSandboxRecreateHint(params: { scope: SandboxConfig["scope"]; sessionKey: string }) {
  if (params.scope === "session") {
    return formatCliCommand(`maistro sandbox recreate --session ${params.sessionKey}`);
  }
  if (params.scope === "agent") {
    const agentId = resolveSandboxAgentId(params.sessionKey) ?? "main";
    return formatCliCommand(`maistro sandbox recreate --agent ${agentId}`);
  }
  return formatCliCommand("maistro sandbox recreate --all");
}

export async function ensureSandboxContainer(params: {
  sessionKey: string;
  workspaceDir: string;
  agentWorkspaceDir: string;
  cfg: SandboxConfig;
}) {
  const scopeKey = resolveSandboxScopeKey(params.cfg.scope, params.sessionKey);
  const slug = params.cfg.scope === "shared" ? "shared" : slugifySessionKey(scopeKey);
  const name = `${params.cfg.docker.containerPrefix}${slug}`;
  const containerName = name.slice(0, 63);
  const expectedHash = computeSandboxConfigHash({
    docker: params.cfg.docker,
    workspaceAccess: params.cfg.workspaceAccess,
    workspaceDir: params.workspaceDir,
    agentWorkspaceDir: params.agentWorkspaceDir,
  });
  const now = Date.now();
  const state = await dockerContainerState(containerName);
  let hasContainer = state.exists;
  let running = state.running;
  let currentHash: string | null = null;
  let hashMismatch = false;
  let registryEntry:
    | {
        lastUsedAtMs: number;
        configHash?: string;
      }
    | undefined;
  if (hasContainer) {
    const registry = await readRegistry();
    registryEntry = registry.entries.find((entry) => entry.containerName === containerName);
    currentHash = await readContainerConfigHash(containerName);
    if (!currentHash) {
      currentHash = registryEntry?.configHash ?? null;
    }
    hashMismatch = !currentHash || currentHash !== expectedHash;
    if (hashMismatch) {
      const lastUsedAtMs = registryEntry?.lastUsedAtMs;
      const isHot =
        running &&
        (typeof lastUsedAtMs !== "number" || now - lastUsedAtMs < HOT_CONTAINER_WINDOW_MS);
      if (isHot) {
        const hint = formatSandboxRecreateHint({ scope: params.cfg.scope, sessionKey: scopeKey });
        defaultRuntime.log(
          `Sandbox config changed for ${containerName} (recently used). Recreate to apply: ${hint}`,
        );
      } else {
        await execDocker(["rm", "-f", containerName], { allowFailure: true });
        hasContainer = false;
        running = false;
      }
    }
  }
  if (!hasContainer) {
    await createSandboxContainer({
      name: containerName,
      cfg: params.cfg.docker,
      workspaceDir: params.workspaceDir,
      workspaceAccess: params.cfg.workspaceAccess,
      agentWorkspaceDir: params.agentWorkspaceDir,
      scopeKey,
      configHash: expectedHash,
    });
  } else if (!running) {
    await execDocker(["start", containerName]);
  }
  await updateRegistry({
    containerName,
    sessionKey: scopeKey,
    createdAtMs: now,
    lastUsedAtMs: now,
    image: params.cfg.docker.image,
    configHash: hashMismatch && running ? (currentHash ?? undefined) : expectedHash,
  });
  return containerName;
}
