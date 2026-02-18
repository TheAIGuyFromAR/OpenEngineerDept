#!/usr/bin/env bash
# Conductor — Cron-able trace and conversation review
#
# Add to crontab for automated periodic reviews:
#   # Daily review at 6am
#   0 6 * * * /path/to/conductor/cron-review.sh --since 24h
#
#   # Weekly deep review on Mondays
#   0 7 * * 1 /path/to/conductor/cron-review.sh --since 7d
#
# Reviews are written to data/reviews/ as both JSON and Markdown.
# If an Obsidian vault is configured, reviews are also copied there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [[ -d "$SCRIPT_DIR/.venv" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Parse args (pass through to trace_reviewer)
SINCE="${1:---since}"
SINCE_VAL="${2:-24h}"

if [[ "$SINCE" == "--since" ]]; then
  ARGS="--since $SINCE_VAL"
else
  ARGS="--since 24h"
fi

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Running trace review ($ARGS)..."

python3 -m orchestrator.agents.trace_reviewer $ARGS \
  --training-dir "$SCRIPT_DIR/data/training" \
  --metrics-dir "$SCRIPT_DIR/data/metrics" \
  --output "$SCRIPT_DIR/data/reviews"

# Copy latest review to Obsidian vault if configured
if [[ -f "$SCRIPT_DIR/projects/example/conductor.yaml" ]]; then
  VAULT_PATH=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$SCRIPT_DIR/projects/example/conductor.yaml'))
print(cfg.get('obsidian_vault', ''))
" 2>/dev/null)

  if [[ -n "$VAULT_PATH" ]] && [[ -d "$VAULT_PATH" ]] && [[ "$VAULT_PATH" != "/path/to/obsidian-vault" ]]; then
    REVIEW_DIR="$VAULT_PATH/conductor/reviews"
    mkdir -p "$REVIEW_DIR"
    # Copy latest markdown review
    LATEST_MD=$(ls -1t "$SCRIPT_DIR/data/reviews/"review-*.md 2>/dev/null | head -1)
    if [[ -n "$LATEST_MD" ]]; then
      cp "$LATEST_MD" "$REVIEW_DIR/"
      echo "[review] Copied to Obsidian: $REVIEW_DIR/$(basename "$LATEST_MD")"
    fi
  fi
fi

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Review complete."
