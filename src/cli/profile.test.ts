import path from "node:path";
import { describe, expect, it } from "vitest";
import { formatCliCommand } from "./command-format.js";
import { applyCliProfileEnv, parseCliProfileArgs } from "./profile.js";

describe("parseCliProfileArgs", () => {
  it("leaves gateway --dev for subcommands", () => {
    const res = parseCliProfileArgs([
      "node",
      "maistro",
      "gateway",
      "--dev",
      "--allow-unconfigured",
    ]);
    if (!res.ok) {
      throw new Error(res.error);
    }
    expect(res.profile).toBeNull();
    expect(res.argv).toEqual(["node", "maistro", "gateway", "--dev", "--allow-unconfigured"]);
  });

  it("still accepts global --dev before subcommand", () => {
    const res = parseCliProfileArgs(["node", "maistro", "--dev", "gateway"]);
    if (!res.ok) {
      throw new Error(res.error);
    }
    expect(res.profile).toBe("dev");
    expect(res.argv).toEqual(["node", "maistro", "gateway"]);
  });

  it("parses --profile value and strips it", () => {
    const res = parseCliProfileArgs(["node", "maistro", "--profile", "work", "status"]);
    if (!res.ok) {
      throw new Error(res.error);
    }
    expect(res.profile).toBe("work");
    expect(res.argv).toEqual(["node", "maistro", "status"]);
  });

  it("rejects missing profile value", () => {
    const res = parseCliProfileArgs(["node", "maistro", "--profile"]);
    expect(res.ok).toBe(false);
  });

  it("rejects combining --dev with --profile (dev first)", () => {
    const res = parseCliProfileArgs(["node", "maistro", "--dev", "--profile", "work", "status"]);
    expect(res.ok).toBe(false);
  });

  it("rejects combining --dev with --profile (profile first)", () => {
    const res = parseCliProfileArgs(["node", "maistro", "--profile", "work", "--dev", "status"]);
    expect(res.ok).toBe(false);
  });
});

describe("applyCliProfileEnv", () => {
  it("fills env defaults for dev profile", () => {
    const env: Record<string, string | undefined> = {};
    applyCliProfileEnv({
      profile: "dev",
      env,
      homedir: () => "/home/peter",
    });
    const expectedStateDir = path.join(path.resolve("/home/peter"), ".maistro-dev");
    expect(env.MAISTRO_PROFILE).toBe("dev");
    expect(env.MAISTRO_STATE_DIR).toBe(expectedStateDir);
    expect(env.MAISTRO_CONFIG_PATH).toBe(path.join(expectedStateDir, "maistro.json"));
    expect(env.MAISTRO_GATEWAY_PORT).toBe("19001");
  });

  it("does not override explicit env values", () => {
    const env: Record<string, string | undefined> = {
      MAISTRO_STATE_DIR: "/custom",
      MAISTRO_GATEWAY_PORT: "19099",
    };
    applyCliProfileEnv({
      profile: "dev",
      env,
      homedir: () => "/home/peter",
    });
    expect(env.MAISTRO_STATE_DIR).toBe("/custom");
    expect(env.MAISTRO_GATEWAY_PORT).toBe("19099");
    expect(env.MAISTRO_CONFIG_PATH).toBe(path.join("/custom", "maistro.json"));
  });

  it("uses MAISTRO_HOME when deriving profile state dir", () => {
    const env: Record<string, string | undefined> = {
      MAISTRO_HOME: "/srv/maistro-home",
      HOME: "/home/other",
    };
    applyCliProfileEnv({
      profile: "work",
      env,
      homedir: () => "/home/fallback",
    });

    const resolvedHome = path.resolve("/srv/maistro-home");
    expect(env.MAISTRO_STATE_DIR).toBe(path.join(resolvedHome, ".maistro-work"));
    expect(env.MAISTRO_CONFIG_PATH).toBe(
      path.join(resolvedHome, ".maistro-work", "maistro.json"),
    );
  });
});

describe("formatCliCommand", () => {
  it("returns command unchanged when no profile is set", () => {
    expect(formatCliCommand("maistro doctor --fix", {})).toBe("maistro doctor --fix");
  });

  it("returns command unchanged when profile is default", () => {
    expect(formatCliCommand("maistro doctor --fix", { MAISTRO_PROFILE: "default" })).toBe(
      "maistro doctor --fix",
    );
  });

  it("returns command unchanged when profile is Default (case-insensitive)", () => {
    expect(formatCliCommand("maistro doctor --fix", { MAISTRO_PROFILE: "Default" })).toBe(
      "maistro doctor --fix",
    );
  });

  it("returns command unchanged when profile is invalid", () => {
    expect(formatCliCommand("maistro doctor --fix", { MAISTRO_PROFILE: "bad profile" })).toBe(
      "maistro doctor --fix",
    );
  });

  it("returns command unchanged when --profile is already present", () => {
    expect(
      formatCliCommand("maistro --profile work doctor --fix", { MAISTRO_PROFILE: "work" }),
    ).toBe("maistro --profile work doctor --fix");
  });

  it("returns command unchanged when --dev is already present", () => {
    expect(formatCliCommand("maistro --dev doctor", { MAISTRO_PROFILE: "dev" })).toBe(
      "maistro --dev doctor",
    );
  });

  it("inserts --profile flag when profile is set", () => {
    expect(formatCliCommand("maistro doctor --fix", { MAISTRO_PROFILE: "work" })).toBe(
      "maistro --profile work doctor --fix",
    );
  });

  it("trims whitespace from profile", () => {
    expect(formatCliCommand("maistro doctor --fix", { MAISTRO_PROFILE: "  jbmaistro  " })).toBe(
      "maistro --profile jbmaistro doctor --fix",
    );
  });

  it("handles command with no args after maistro", () => {
    expect(formatCliCommand("maistro", { MAISTRO_PROFILE: "test" })).toBe(
      "maistro --profile test",
    );
  });

  it("handles pnpm wrapper", () => {
    expect(formatCliCommand("pnpm maistro doctor", { MAISTRO_PROFILE: "work" })).toBe(
      "pnpm maistro --profile work doctor",
    );
  });
});
