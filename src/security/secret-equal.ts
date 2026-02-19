import { createHmac, timingSafeEqual } from "node:crypto";

/**
 * Constant-time secret comparison that does NOT leak length information.
 *
 * Previous implementation returned early when buffer lengths differed,
 * which allowed attackers to determine secret length via timing analysis.
 *
 * This version HMAC-hashes both values with a fixed key before comparison,
 * producing fixed-length digests regardless of input length. The comparison
 * is always performed in constant time via `timingSafeEqual`.
 */
const HMAC_KEY = Buffer.from("maistro-secret-comparison-v2");

export function safeEqualSecret(
  provided: string | undefined | null,
  expected: string | undefined | null,
): boolean {
  if (typeof provided !== "string" || typeof expected !== "string") {
    // Ensure non-string inputs still consume constant time against
    // valid-string inputs by performing a dummy comparison.
    const dummy = Buffer.alloc(32);
    timingSafeEqual(dummy, dummy);
    return false;
  }
  const providedHash = createHmac("sha256", HMAC_KEY).update(provided).digest();
  const expectedHash = createHmac("sha256", HMAC_KEY).update(expected).digest();
  return timingSafeEqual(providedHash, expectedHash);
}
