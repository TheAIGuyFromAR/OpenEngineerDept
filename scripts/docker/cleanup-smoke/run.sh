#!/usr/bin/env bash
set -euo pipefail

cd /repo

export MAISTRO_STATE_DIR="/tmp/maistro-test"
export MAISTRO_CONFIG_PATH="${MAISTRO_STATE_DIR}/maistro.json"

echo "==> Build"
pnpm build

echo "==> Seed state"
mkdir -p "${MAISTRO_STATE_DIR}/credentials"
mkdir -p "${MAISTRO_STATE_DIR}/agents/main/sessions"
echo '{}' >"${MAISTRO_CONFIG_PATH}"
echo 'creds' >"${MAISTRO_STATE_DIR}/credentials/marker.txt"
echo 'session' >"${MAISTRO_STATE_DIR}/agents/main/sessions/sessions.json"

echo "==> Reset (config+creds+sessions)"
pnpm maistro reset --scope config+creds+sessions --yes --non-interactive

test ! -f "${MAISTRO_CONFIG_PATH}"
test ! -d "${MAISTRO_STATE_DIR}/credentials"
test ! -d "${MAISTRO_STATE_DIR}/agents/main/sessions"

echo "==> Recreate minimal config"
mkdir -p "${MAISTRO_STATE_DIR}/credentials"
echo '{}' >"${MAISTRO_CONFIG_PATH}"

echo "==> Uninstall (state only)"
pnpm maistro uninstall --state --yes --non-interactive

test ! -d "${MAISTRO_STATE_DIR}"

echo "OK"
