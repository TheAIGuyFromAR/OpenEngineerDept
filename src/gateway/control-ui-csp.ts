import crypto from "node:crypto";

/**
 * Build a nonce for CSP style-src to replace 'unsafe-inline'.
 * The nonce is cryptographically random and unique per response.
 */
export function generateCspNonce(): string {
  return crypto.randomBytes(16).toString("base64");
}

export function buildControlUiCspHeader(nonce?: string): string {
  // Control UI: block framing, block inline scripts.
  // Style-src uses a nonce instead of 'unsafe-inline' to prevent style injection XSS.
  const styleDirective = nonce
    ? `style-src 'self' 'nonce-${nonce}'`
    : "style-src 'self'";
  return [
    "default-src 'self'",
    "base-uri 'none'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "script-src 'self'",
    styleDirective,
    "img-src 'self' data: https:",
    "font-src 'self'",
    "connect-src 'self' ws: wss:",
    "form-action 'self'",
    "upgrade-insecure-requests",
  ].join("; ");
}
