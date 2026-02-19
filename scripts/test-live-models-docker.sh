#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${MAISTRO_IMAGE:-${MAISTRO_IMAGE:-maistro:local}}"
CONFIG_DIR="${MAISTRO_CONFIG_DIR:-${MAISTRO_CONFIG_DIR:-$HOME/.maistro}}"
WORKSPACE_DIR="${MAISTRO_WORKSPACE_DIR:-${MAISTRO_WORKSPACE_DIR:-$HOME/.maistro/workspace}}"
PROFILE_FILE="${MAISTRO_PROFILE_FILE:-${MAISTRO_PROFILE_FILE:-$HOME/.profile}}"

PROFILE_MOUNT=()
if [[ -f "$PROFILE_FILE" ]]; then
  PROFILE_MOUNT=(-v "$PROFILE_FILE":/home/node/.profile:ro)
fi

echo "==> Build image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

echo "==> Run live model tests (profile keys)"
docker run --rm -t \
  --entrypoint bash \
  -e COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
  -e HOME=/home/node \
  -e NODE_OPTIONS=--disable-warning=ExperimentalWarning \
  -e MAISTRO_LIVE_TEST=1 \
  -e MAISTRO_LIVE_MODELS="${MAISTRO_LIVE_MODELS:-${MAISTRO_LIVE_MODELS:-all}}" \
  -e MAISTRO_LIVE_PROVIDERS="${MAISTRO_LIVE_PROVIDERS:-${MAISTRO_LIVE_PROVIDERS:-}}" \
  -e MAISTRO_LIVE_MODEL_TIMEOUT_MS="${MAISTRO_LIVE_MODEL_TIMEOUT_MS:-${MAISTRO_LIVE_MODEL_TIMEOUT_MS:-}}" \
  -e MAISTRO_LIVE_REQUIRE_PROFILE_KEYS="${MAISTRO_LIVE_REQUIRE_PROFILE_KEYS:-${MAISTRO_LIVE_REQUIRE_PROFILE_KEYS:-}}" \
  -v "$CONFIG_DIR":/home/node/.maistro \
  -v "$WORKSPACE_DIR":/home/node/.maistro/workspace \
  "${PROFILE_MOUNT[@]}" \
  "$IMAGE_NAME" \
  -lc "set -euo pipefail; [ -f \"$HOME/.profile\" ] && source \"$HOME/.profile\" || true; cd /app && pnpm test:live"
