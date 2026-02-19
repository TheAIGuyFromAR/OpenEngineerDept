/**
 * Cryptographically secure random utilities.
 *
 * Replaces Math.random()-based ID generation in security-sensitive contexts.
 * Uses Node.js crypto module backed by the OS CSPRNG (/dev/urandom).
 */

import { randomBytes, randomInt } from "node:crypto";

/**
 * Generate a cryptographically secure random hex string.
 * Default 16 bytes = 32 hex chars, providing 128 bits of entropy.
 */
export function secureId(bytes = 16): string {
  return randomBytes(bytes).toString("hex");
}

/**
 * Generate a cryptographically secure random integer in [min, max).
 * Replacement for `Math.floor(Math.random() * (max - min)) + min`.
 */
export function secureRandomInt(min: number, max: number): number {
  return randomInt(min, max);
}

/**
 * Generate a cryptographically secure random base36 string of the given length.
 * Useful as a drop-in for `Math.random().toString(36).slice(2, n)`.
 */
export function secureBase36(length = 8): string {
  // Generate enough random bytes to produce the desired base36 length.
  // Each byte gives ~1.3 base36 chars, so we overshoot and truncate.
  const bytes = Math.ceil(length * 0.8) + 2;
  return randomBytes(bytes).toString("hex").slice(0, length);
}
