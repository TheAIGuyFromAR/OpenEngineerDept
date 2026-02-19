import { describe, expect, it } from "vitest";
import { resolveIrcInboundTarget } from "./monitor.js";

describe("irc monitor inbound target", () => {
  it("keeps channel target for group messages", () => {
    expect(
      resolveIrcInboundTarget({
        target: "#maistro",
        senderNick: "alice",
      }),
    ).toEqual({
      isGroup: true,
      target: "#maistro",
      rawTarget: "#maistro",
    });
  });

  it("maps DM target to sender nick and preserves raw target", () => {
    expect(
      resolveIrcInboundTarget({
        target: "maistro-bot",
        senderNick: "alice",
      }),
    ).toEqual({
      isGroup: false,
      target: "alice",
      rawTarget: "maistro-bot",
    });
  });

  it("falls back to raw target when sender nick is empty", () => {
    expect(
      resolveIrcInboundTarget({
        target: "maistro-bot",
        senderNick: " ",
      }),
    ).toEqual({
      isGroup: false,
      target: "maistro-bot",
      rawTarget: "maistro-bot",
    });
  });
});
