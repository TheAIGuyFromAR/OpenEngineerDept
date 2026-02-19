import { describe, expect, it } from "vitest";
import { checkBrowserOrigin } from "./origin-check.js";

describe("checkBrowserOrigin", () => {
  it("accepts same-origin host matches", () => {
    const result = checkBrowserOrigin({
      requestHost: "127.0.0.1:18789",
      origin: "http://127.0.0.1:18789",
    });
    expect(result.ok).toBe(true);
  });

  it("accepts loopback host mismatches for dev", () => {
    const result = checkBrowserOrigin({
      requestHost: "127.0.0.1:18789",
      origin: "http://localhost:5173",
    });
    expect(result.ok).toBe(true);
  });

  it("accepts allowlisted origins", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
      origin: "https://control.example.com",
      allowedOrigins: ["https://control.example.com"],
    });
    expect(result.ok).toBe(true);
  });

  it("rejects missing origin", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
      origin: "",
    });
    expect(result.ok).toBe(false);
  });

  it("rejects mismatched origins", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
      origin: "https://attacker.example.com",
    });
    expect(result.ok).toBe(false);
  });

  it("rejects origin string 'null'", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
      origin: "null",
    });
    expect(result.ok).toBe(false);
  });

  it("rejects undefined origin", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
    });
    expect(result.ok).toBe(false);
  });

  it("rejects malformed origin URL", () => {
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com:18789",
      origin: "not-a-url",
    });
    expect(result.ok).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// IDN / Punycode homoglyph attack prevention
// ---------------------------------------------------------------------------

describe("IDN homoglyph prevention", () => {
  it("rejects Cyrillic homoglyph of allowed origin", () => {
    // Cyrillic 'а' (U+0430) looks identical to Latin 'a' (U+0061)
    // "аpple.com" with Cyrillic а vs "apple.com" with Latin a
    const cyrillicA = "\u0430"; // Cyrillic small letter a
    const result = checkBrowserOrigin({
      requestHost: "apple.com",
      origin: `https://${cyrillicA}pple.com`,
    });
    // Should reject because punycode normalizes Cyrillic а → xn-- prefix
    expect(result.ok).toBe(false);
  });

  it("normalizes IDN origins to punycode for comparison", () => {
    // "münchen.de" should normalize to "xn--mnchen-3ya.de"
    const result = checkBrowserOrigin({
      requestHost: "xn--mnchen-3ya.de:18789",
      origin: "https://m\u00FCnchen.de:18789",
    });
    expect(result.ok).toBe(true);
  });

  it("normalizes IDN allowlist entries to punycode", () => {
    // Allowlist contains Unicode, origin contains punycode — should match
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com",
      origin: "https://xn--mnchen-3ya.de",
      allowedOrigins: ["https://m\u00FCnchen.de"],
    });
    expect(result.ok).toBe(true);
  });

  it("normalizes both sides for consistent comparison", () => {
    // Both origin and allowlist in Unicode
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com",
      origin: "https://m\u00FCnchen.de",
      allowedOrigins: ["https://m\u00FCnchen.de"],
    });
    expect(result.ok).toBe(true);
  });

  it("rejects mixed-script homoglyph in allowlist check", () => {
    // Allowlist has real "example.com", attacker uses Cyrillic е (U+0435)
    const cyrillicE = "\u0435";
    const result = checkBrowserOrigin({
      requestHost: "gateway.example.com",
      origin: `https://${cyrillicE}xample.com`,
      allowedOrigins: ["https://example.com"],
    });
    expect(result.ok).toBe(false);
  });

  it("handles pure ASCII domains without modification", () => {
    const result = checkBrowserOrigin({
      requestHost: "example.com:18789",
      origin: "https://example.com:18789",
    });
    expect(result.ok).toBe(true);
  });

  it("case-insensitive matching after normalization", () => {
    const result = checkBrowserOrigin({
      requestHost: "Example.COM:18789",
      origin: "https://EXAMPLE.com:18789",
    });
    expect(result.ok).toBe(true);
  });
});
