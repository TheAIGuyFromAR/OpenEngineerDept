import { describe, expect, it } from "vitest";
import {
  buildParseArgv,
  getFlagValue,
  getCommandPath,
  getPrimaryCommand,
  getPositiveIntFlagValue,
  getVerboseFlag,
  hasHelpOrVersion,
  hasFlag,
  shouldMigrateState,
  shouldMigrateStateFromPath,
} from "./argv.js";

describe("argv helpers", () => {
  it("detects help/version flags", () => {
    expect(hasHelpOrVersion(["node", "maistro", "--help"])).toBe(true);
    expect(hasHelpOrVersion(["node", "maistro", "-V"])).toBe(true);
    expect(hasHelpOrVersion(["node", "maistro", "status"])).toBe(false);
  });

  it("extracts command path ignoring flags and terminator", () => {
    expect(getCommandPath(["node", "maistro", "status", "--json"], 2)).toEqual(["status"]);
    expect(getCommandPath(["node", "maistro", "agents", "list"], 2)).toEqual(["agents", "list"]);
    expect(getCommandPath(["node", "maistro", "status", "--", "ignored"], 2)).toEqual(["status"]);
  });

  it("returns primary command", () => {
    expect(getPrimaryCommand(["node", "maistro", "agents", "list"])).toBe("agents");
    expect(getPrimaryCommand(["node", "maistro"])).toBeNull();
  });

  it("parses boolean flags and ignores terminator", () => {
    expect(hasFlag(["node", "maistro", "status", "--json"], "--json")).toBe(true);
    expect(hasFlag(["node", "maistro", "--", "--json"], "--json")).toBe(false);
  });

  it("extracts flag values with equals and missing values", () => {
    expect(getFlagValue(["node", "maistro", "status", "--timeout", "5000"], "--timeout")).toBe(
      "5000",
    );
    expect(getFlagValue(["node", "maistro", "status", "--timeout=2500"], "--timeout")).toBe(
      "2500",
    );
    expect(getFlagValue(["node", "maistro", "status", "--timeout"], "--timeout")).toBeNull();
    expect(getFlagValue(["node", "maistro", "status", "--timeout", "--json"], "--timeout")).toBe(
      null,
    );
    expect(getFlagValue(["node", "maistro", "--", "--timeout=99"], "--timeout")).toBeUndefined();
  });

  it("parses verbose flags", () => {
    expect(getVerboseFlag(["node", "maistro", "status", "--verbose"])).toBe(true);
    expect(getVerboseFlag(["node", "maistro", "status", "--debug"])).toBe(false);
    expect(getVerboseFlag(["node", "maistro", "status", "--debug"], { includeDebug: true })).toBe(
      true,
    );
  });

  it("parses positive integer flag values", () => {
    expect(getPositiveIntFlagValue(["node", "maistro", "status"], "--timeout")).toBeUndefined();
    expect(
      getPositiveIntFlagValue(["node", "maistro", "status", "--timeout"], "--timeout"),
    ).toBeNull();
    expect(
      getPositiveIntFlagValue(["node", "maistro", "status", "--timeout", "5000"], "--timeout"),
    ).toBe(5000);
    expect(
      getPositiveIntFlagValue(["node", "maistro", "status", "--timeout", "nope"], "--timeout"),
    ).toBeUndefined();
  });

  it("builds parse argv from raw args", () => {
    const nodeArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node", "maistro", "status"],
    });
    expect(nodeArgv).toEqual(["node", "maistro", "status"]);

    const versionedNodeArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node-22", "maistro", "status"],
    });
    expect(versionedNodeArgv).toEqual(["node-22", "maistro", "status"]);

    const versionedNodeWindowsArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node-22.2.0.exe", "maistro", "status"],
    });
    expect(versionedNodeWindowsArgv).toEqual(["node-22.2.0.exe", "maistro", "status"]);

    const versionedNodePatchlessArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node-22.2", "maistro", "status"],
    });
    expect(versionedNodePatchlessArgv).toEqual(["node-22.2", "maistro", "status"]);

    const versionedNodeWindowsPatchlessArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node-22.2.exe", "maistro", "status"],
    });
    expect(versionedNodeWindowsPatchlessArgv).toEqual(["node-22.2.exe", "maistro", "status"]);

    const versionedNodeWithPathArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["/usr/bin/node-22.2.0", "maistro", "status"],
    });
    expect(versionedNodeWithPathArgv).toEqual(["/usr/bin/node-22.2.0", "maistro", "status"]);

    const nodejsArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["nodejs", "maistro", "status"],
    });
    expect(nodejsArgv).toEqual(["nodejs", "maistro", "status"]);

    const nonVersionedNodeArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["node-dev", "maistro", "status"],
    });
    expect(nonVersionedNodeArgv).toEqual(["node", "maistro", "node-dev", "maistro", "status"]);

    const directArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["maistro", "status"],
    });
    expect(directArgv).toEqual(["node", "maistro", "status"]);

    const bunArgv = buildParseArgv({
      programName: "maistro",
      rawArgs: ["bun", "src/entry.ts", "status"],
    });
    expect(bunArgv).toEqual(["bun", "src/entry.ts", "status"]);
  });

  it("builds parse argv from fallback args", () => {
    const fallbackArgv = buildParseArgv({
      programName: "maistro",
      fallbackArgv: ["status"],
    });
    expect(fallbackArgv).toEqual(["node", "maistro", "status"]);
  });

  it("decides when to migrate state", () => {
    expect(shouldMigrateState(["node", "maistro", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "health"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "sessions"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "config", "get", "update"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "config", "unset", "update"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "models", "list"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "models", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "memory", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "agent", "--message", "hi"])).toBe(false);
    expect(shouldMigrateState(["node", "maistro", "agents", "list"])).toBe(true);
    expect(shouldMigrateState(["node", "maistro", "message", "send"])).toBe(true);
  });

  it("reuses command path for migrate state decisions", () => {
    expect(shouldMigrateStateFromPath(["status"])).toBe(false);
    expect(shouldMigrateStateFromPath(["config", "get"])).toBe(false);
    expect(shouldMigrateStateFromPath(["models", "status"])).toBe(false);
    expect(shouldMigrateStateFromPath(["agents", "list"])).toBe(true);
  });
});
