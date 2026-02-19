import { domainToASCII } from "node:url";
import { isLoopbackHost, normalizeHostHeader, resolveHostName } from "./net.js";

type OriginCheckResult = { ok: true } | { ok: false; reason: string };

/**
 * Normalize a hostname to its ASCII/punycode form to prevent IDN homoglyph
 * bypass attacks (e.g. Cyrillic 'а' in place of Latin 'a').
 * Returns the lowercased punycode hostname, or the original lowercased input
 * if conversion fails (e.g. already ASCII or invalid).
 */
function normalizeHostnameToASCII(hostname: string): string {
  const lower = hostname.toLowerCase();
  try {
    const ascii = domainToASCII(lower);
    // domainToASCII returns empty string on failure
    return ascii || lower;
  } catch {
    return lower;
  }
}

function parseOrigin(
  originRaw?: string,
): { origin: string; host: string; hostname: string } | null {
  const trimmed = (originRaw ?? "").trim();
  if (!trimmed || trimmed === "null") {
    return null;
  }
  try {
    const url = new URL(trimmed);
    // Normalize hostname through punycode to prevent IDN homoglyph attacks
    const normalizedHostname = normalizeHostnameToASCII(url.hostname);
    const port = url.port ? `:${url.port}` : "";
    return {
      origin: `${url.protocol}//${normalizedHostname}${port}`.toLowerCase(),
      host: `${normalizedHostname}${port}`.toLowerCase(),
      hostname: normalizedHostname,
    };
  } catch {
    return null;
  }
}

export function checkBrowserOrigin(params: {
  requestHost?: string;
  origin?: string;
  allowedOrigins?: string[];
}): OriginCheckResult {
  const parsedOrigin = parseOrigin(params.origin);
  if (!parsedOrigin) {
    return { ok: false, reason: "origin missing or invalid" };
  }

  // Normalize allowlist entries through punycode as well for consistent comparison
  const allowlist = (params.allowedOrigins ?? [])
    .map((value) => {
      const trimmed = value.trim().toLowerCase();
      // If it looks like an origin (has protocol), parse and normalize it
      try {
        const url = new URL(trimmed);
        const normalizedHost = normalizeHostnameToASCII(url.hostname);
        const port = url.port ? `:${url.port}` : "";
        return `${url.protocol}//${normalizedHost}${port}`;
      } catch {
        return trimmed;
      }
    })
    .filter(Boolean);
  if (allowlist.includes(parsedOrigin.origin)) {
    return { ok: true };
  }

  const requestHost = normalizeHostHeader(params.requestHost);
  if (requestHost && parsedOrigin.host === requestHost) {
    return { ok: true };
  }

  const requestHostname = resolveHostName(requestHost);
  if (isLoopbackHost(parsedOrigin.hostname) && isLoopbackHost(requestHostname)) {
    return { ok: true };
  }

  return { ok: false, reason: "origin not allowed" };
}
