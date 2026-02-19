import { describe, expect, it } from "vitest";
import { secureBase36, secureId, secureRandomInt } from "./secure-random.js";

describe("secureId", () => {
  it("returns a hex string of default length (32 chars = 16 bytes)", () => {
    const id = secureId();
    expect(id).toHaveLength(32);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  it("returns hex string with specified byte count", () => {
    const id = secureId(4);
    expect(id).toHaveLength(8);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  it("returns hex string for single byte", () => {
    const id = secureId(1);
    expect(id).toHaveLength(2);
    expect(id).toMatch(/^[0-9a-f]+$/);
  });

  it("generates unique IDs across calls", () => {
    const ids = new Set(Array.from({ length: 1000 }, () => secureId()));
    expect(ids.size).toBe(1000);
  });

  it("generates unique IDs even with small byte count", () => {
    // 4 bytes = 2^32 possibilities — 100 calls should never collide
    const ids = new Set(Array.from({ length: 100 }, () => secureId(4)));
    expect(ids.size).toBe(100);
  });
});

describe("secureRandomInt", () => {
  it("returns integer within specified range", () => {
    for (let i = 0; i < 100; i++) {
      const n = secureRandomInt(10, 20);
      expect(n).toBeGreaterThanOrEqual(10);
      expect(n).toBeLessThan(20);
      expect(Number.isInteger(n)).toBe(true);
    }
  });

  it("returns min when range is 1", () => {
    const n = secureRandomInt(42, 43);
    expect(n).toBe(42);
  });

  it("covers the full range", () => {
    const seen = new Set<number>();
    for (let i = 0; i < 500; i++) {
      seen.add(secureRandomInt(0, 5));
    }
    // With 500 rolls of [0,5), all 5 values should appear
    expect(seen.size).toBe(5);
  });

  it("handles large ranges", () => {
    const n = secureRandomInt(0, 1_000_000);
    expect(n).toBeGreaterThanOrEqual(0);
    expect(n).toBeLessThan(1_000_000);
  });
});

describe("secureBase36", () => {
  it("returns string of default length (8)", () => {
    const s = secureBase36();
    expect(s).toHaveLength(8);
  });

  it("returns string of specified length", () => {
    const s = secureBase36(16);
    expect(s).toHaveLength(16);
  });

  it("returns only hex characters", () => {
    // Current implementation uses hex encoding, so output is [0-9a-f]
    const s = secureBase36(32);
    expect(s).toMatch(/^[0-9a-f]+$/);
  });

  it("generates unique strings", () => {
    const strs = new Set(Array.from({ length: 100 }, () => secureBase36(12)));
    expect(strs.size).toBe(100);
  });

  it("handles length of 1", () => {
    const s = secureBase36(1);
    expect(s).toHaveLength(1);
  });
});
