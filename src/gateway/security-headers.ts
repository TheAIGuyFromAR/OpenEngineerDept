import type { IncomingMessage, ServerResponse } from "node:http";

/**
 * Enterprise security headers applied to every HTTP response.
 *
 * These headers defend against:
 * - MIME type sniffing (X-Content-Type-Options)
 * - Clickjacking (X-Frame-Options, CSP frame-ancestors)
 * - Referrer information leakage (Referrer-Policy)
 * - Feature/permission abuse (Permissions-Policy)
 * - Caching of sensitive responses (Cache-Control, Pragma)
 * - Cross-origin information leakage (Cross-Origin-* headers)
 * - Protocol downgrade attacks (Strict-Transport-Security)
 */

export type SecurityHeadersOptions = {
  /** Enable HSTS header. Only set when TLS termination is confirmed. */
  enableHsts?: boolean;
  /** HSTS max-age in seconds. @default 31536000 (1 year) */
  hstsMaxAge?: number;
};

const DEFAULT_HSTS_MAX_AGE = 31_536_000; // 1 year

export function applySecurityHeaders(
  _req: IncomingMessage,
  res: ServerResponse,
  options?: SecurityHeadersOptions,
): void {
  // Prevent MIME type sniffing — browsers must respect Content-Type
  res.setHeader("X-Content-Type-Options", "nosniff");

  // Prevent clickjacking — deny all framing (defense-in-depth alongside CSP frame-ancestors)
  res.setHeader("X-Frame-Options", "DENY");

  // Control referrer information leakage
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");

  // Restrict browser feature access
  res.setHeader(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()",
  );

  // Prevent caching of authenticated responses
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, private");
  res.setHeader("Pragma", "no-cache");

  // Cross-origin isolation headers
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");

  // HSTS — only when TLS is confirmed (either direct TLS or behind a trusted proxy)
  if (options?.enableHsts) {
    const maxAge = options.hstsMaxAge ?? DEFAULT_HSTS_MAX_AGE;
    res.setHeader("Strict-Transport-Security", `max-age=${maxAge}; includeSubDomains`);
  }

  // Remove server identification header
  res.removeHeader("X-Powered-By");
}
