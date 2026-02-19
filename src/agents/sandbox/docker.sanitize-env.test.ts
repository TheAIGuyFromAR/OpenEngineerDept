import { describe, expect, it } from "vitest";
import { sanitizeSandboxEnv } from "./docker.js";

describe("sanitizeSandboxEnv", () => {
  // -------------------------------------------------------------------------
  // Exact match blocking
  // -------------------------------------------------------------------------

  describe("blocks exact-match sensitive vars", () => {
    const sensitiveExact = [
      "ANTHROPIC_API_KEY",
      "OPENAI_API_KEY",
      "GITHUB_TOKEN",
      "GH_TOKEN",
      "DATABASE_URL",
      "REDIS_URL",
      "SENTRY_DSN",
      "SLACK_BOT_TOKEN",
      "DISCORD_BOT_TOKEN",
      "TELEGRAM_BOT_TOKEN",
      "CLOUDFLARE_API_TOKEN",
      "FLY_API_TOKEN",
      "VERCEL_TOKEN",
      "HEROKU_API_KEY",
      "NPM_TOKEN",
      "DOCKER_PASSWORD",
    ];

    for (const key of sensitiveExact) {
      it(`blocks ${key}`, () => {
        const { sanitized, blocked } = sanitizeSandboxEnv({ [key]: "secret-value" });
        expect(sanitized).not.toHaveProperty(key);
        expect(blocked).toContain(key);
      });
    }
  });

  // -------------------------------------------------------------------------
  // Code injection env vars
  // -------------------------------------------------------------------------

  describe("blocks code injection env vars", () => {
    const injectionVars = [
      "NODE_OPTIONS",
      "NODE_EXTRA_CA_CERTS",
      "JAVA_TOOL_OPTIONS",
      "_JAVA_OPTIONS",
    ];

    for (const key of injectionVars) {
      it(`blocks ${key}`, () => {
        const { sanitized, blocked } = sanitizeSandboxEnv({
          [key]: "--require=/tmp/evil.js",
        });
        expect(sanitized).not.toHaveProperty(key);
        expect(blocked).toContain(key);
      });
    }
  });

  // -------------------------------------------------------------------------
  // Suffix pattern blocking
  // -------------------------------------------------------------------------

  describe("blocks vars matching suffix patterns", () => {
    const suffixCases = [
      ["MY_APP_SECRET", "_SECRET"],
      ["CUSTOM_TOKEN", "_TOKEN"],
      ["DB_PASSWORD", "_PASSWORD"],
      ["ADMIN_PASS", "_PASS"],
      ["STRIPE_API_KEY", "_API_KEY"],
      ["SIGNING_PRIVATE_KEY", "_PRIVATE_KEY"],
      ["GCP_CREDENTIALS", "_CREDENTIALS"],
      ["OAUTH_AUTH", "_AUTH"],
    ];

    for (const [key, suffix] of suffixCases) {
      it(`blocks ${key} (suffix: ${suffix})`, () => {
        const { sanitized, blocked } = sanitizeSandboxEnv({ [key]: "value" });
        expect(sanitized).not.toHaveProperty(key);
        expect(blocked).toContain(key);
      });
    }

    it("matches suffixes case-insensitively", () => {
      const { sanitized, blocked } = sanitizeSandboxEnv({
        my_app_secret: "value",
        Custom_Token: "value",
      });
      expect(sanitized).not.toHaveProperty("my_app_secret");
      expect(sanitized).not.toHaveProperty("Custom_Token");
      expect(blocked).toHaveLength(2);
    });
  });

  // -------------------------------------------------------------------------
  // Prefix pattern blocking
  // -------------------------------------------------------------------------

  describe("blocks vars matching prefix patterns", () => {
    const prefixCases = [
      ["AWS_ACCESS_KEY_ID", "AWS_"],
      ["AWS_SECRET_ACCESS_KEY", "AWS_"],
      ["SSH_AUTH_SOCK", "SSH_"],
      ["GPG_AGENT_INFO", "GPG_"],
      ["VAULT_TOKEN", "VAULT_"],
      ["MAISTRO_GATEWAY_SECRET", "MAISTRO_GATEWAY_"],
      ["NPM_CONFIG_REGISTRY", "NPM_CONFIG_"],
    ];

    for (const [key, prefix] of prefixCases) {
      it(`blocks ${key} (prefix: ${prefix})`, () => {
        const { sanitized, blocked } = sanitizeSandboxEnv({ [key]: "value" });
        expect(sanitized).not.toHaveProperty(key);
        expect(blocked).toContain(key);
      });
    }

    it("matches prefixes case-insensitively", () => {
      const { sanitized, blocked } = sanitizeSandboxEnv({
        aws_session_token: "value",
      });
      expect(sanitized).not.toHaveProperty("aws_session_token");
      expect(blocked).toHaveLength(1);
    });
  });

  // -------------------------------------------------------------------------
  // Safe vars pass through
  // -------------------------------------------------------------------------

  describe("allows safe environment variables", () => {
    const safeVars: Record<string, string> = {
      PATH: "/usr/bin",
      HOME: "/home/user",
      LANG: "en_US.UTF-8",
      TERM: "xterm-256color",
      NODE_ENV: "development",
      CI: "true",
      EDITOR: "vim",
      TZ: "UTC",
      USER: "sandbox",
      SHELL: "/bin/bash",
      MY_APP_PORT: "3000",
      DEBUG: "app:*",
    };

    for (const [key, value] of Object.entries(safeVars)) {
      it(`allows ${key}`, () => {
        const { sanitized, blocked } = sanitizeSandboxEnv({ [key]: value });
        expect(sanitized).toHaveProperty(key, value);
        expect(blocked).toHaveLength(0);
      });
    }
  });

  // -------------------------------------------------------------------------
  // Mixed input
  // -------------------------------------------------------------------------

  it("filters mixed safe and sensitive vars", () => {
    const env = {
      PATH: "/usr/bin",
      NODE_ENV: "production",
      ANTHROPIC_API_KEY: "sk-ant-xxx",
      GITHUB_TOKEN: "ghp_xxx",
      MY_APP_PORT: "8080",
      AWS_SECRET_ACCESS_KEY: "wJalrXUtnFEMI",
    };
    const { sanitized, blocked } = sanitizeSandboxEnv(env);

    expect(Object.keys(sanitized)).toEqual(["PATH", "NODE_ENV", "MY_APP_PORT"]);
    expect(blocked).toContain("ANTHROPIC_API_KEY");
    expect(blocked).toContain("GITHUB_TOKEN");
    expect(blocked).toContain("AWS_SECRET_ACCESS_KEY");
    expect(blocked).toHaveLength(3);
  });

  it("returns empty sanitized and blocked for empty input", () => {
    const { sanitized, blocked } = sanitizeSandboxEnv({});
    expect(sanitized).toEqual({});
    expect(blocked).toEqual([]);
  });

  // -------------------------------------------------------------------------
  // Allowlist override
  // -------------------------------------------------------------------------

  describe("allowlist", () => {
    it("permits explicitly allowlisted sensitive vars", () => {
      const allowlist = new Set(["GITHUB_TOKEN"]);
      const { sanitized, blocked } = sanitizeSandboxEnv(
        { GITHUB_TOKEN: "ghp_xxx", OPENAI_API_KEY: "sk-xxx" },
        allowlist,
      );
      expect(sanitized).toHaveProperty("GITHUB_TOKEN", "ghp_xxx");
      expect(blocked).not.toContain("GITHUB_TOKEN");
      // Non-allowlisted sensitive var is still blocked
      expect(sanitized).not.toHaveProperty("OPENAI_API_KEY");
      expect(blocked).toContain("OPENAI_API_KEY");
    });

    it("allowlist does not affect safe vars", () => {
      const allowlist = new Set(["PATH"]);
      const { sanitized } = sanitizeSandboxEnv({ PATH: "/usr/bin" }, allowlist);
      expect(sanitized).toHaveProperty("PATH", "/usr/bin");
    });

    it("allowlist with undefined behaves as no allowlist", () => {
      const { blocked } = sanitizeSandboxEnv({ ANTHROPIC_API_KEY: "sk-xxx" }, undefined);
      expect(blocked).toContain("ANTHROPIC_API_KEY");
    });
  });
});
