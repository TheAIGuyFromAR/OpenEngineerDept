#!/usr/bin/env bash
# Conductor Phase 0 — Launch full stack
# Starts: [inference engine →] gateway → orchestrator
# Inference engine is only started for local provider; API providers skip it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║       Conductor Phase 0 — Launch         ║"
echo "╚══════════════════════════════════════════╝"

# ── Detect inference provider from config ──────────────────────────
CONFIG="${1:-projects/example/conductor.yaml}"
INFERENCE_PROVIDER="${CONDUCTOR_INFERENCE_PROVIDER:-local}"
# Try to read from YAML config if not set via env var
if [[ "$INFERENCE_PROVIDER" == "local" ]] && [[ -f "$CONFIG" ]]; then
  yaml_provider=$(grep -oP '^\s*inference_provider:\s*"\K[^"]+' "$CONFIG" 2>/dev/null || true)
  if [[ -n "$yaml_provider" ]]; then
    INFERENCE_PROVIDER="$yaml_provider"
  fi
fi

echo "[preflight] Inference provider: $INFERENCE_PROVIDER"

# ── Preflight checks ───────────────────────────────────────────────
READY=true

# Check venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    echo "[preflight] Activating .venv..."
    source "$SCRIPT_DIR/.venv/bin/activate"
  else
    echo "[ERROR] No virtual environment found. Run ./setup.sh first."
    exit 1
  fi
fi

# Check Python deps
python3 -c "import fastapi, uvicorn, httpx, pydantic, watchdog, yaml, rich" 2>/dev/null || {
  echo "[ERROR] Missing Python dependencies. Run: pip install -e .[dev]"
  exit 1
}

# Local-only checks: binary and model
if [[ "$INFERENCE_PROVIDER" == "local" ]]; then
  LLAMA_BIN="${CONDUCTOR_LLAMA_BIN:-./build/bin/llama-server}"
  if [[ ! -x "$LLAMA_BIN" ]] && [[ ! -L "$LLAMA_BIN" ]]; then
    echo "[ERROR] llama-server not found at: $LLAMA_BIN"
    echo "        Run ./setup.sh to build it."
    READY=false
  fi

  MODEL_PATH="${CONDUCTOR_MODEL_PATH:-./models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model not found at: $MODEL_PATH"
    echo "        Run ./setup.sh to download it."
    READY=false
  fi
else
  # API provider: check that API key is configured
  API_KEY="${CONDUCTOR_INFERENCE_API_KEY:-}"
  if [[ -z "$API_KEY" ]]; then
    yaml_key=$(grep -oP '^\s*inference_api_key:\s*"\K[^"]+' "$CONFIG" 2>/dev/null || true)
    if [[ -z "$yaml_key" ]]; then
      echo "[ERROR] No API key configured for provider '$INFERENCE_PROVIDER'."
      echo "        Set CONDUCTOR_INFERENCE_API_KEY or add inference_api_key to $CONFIG"
      READY=false
    fi
  fi
  echo "[preflight] Skipping llama-server/model checks (using $INFERENCE_PROVIDER API)"
fi

# Check project config
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Config not found: $CONFIG"
  echo "        Copy and edit projects/example/conductor.yaml"
  READY=false
elif grep -q '/path/to/' "$CONFIG"; then
  echo "[ERROR] Config still has placeholder paths: $CONFIG"
  echo "        Edit project_dir and obsidian_vault before launching."
  READY=false
fi

if [[ "$READY" != true ]]; then
  echo ""
  echo "[ABORT] Fix the above errors and retry."
  exit 1
fi

# Extract project name from config path
PROJECT=$(basename "$(dirname "$CONFIG")")

echo ""
echo "[preflight] All checks passed."
echo ""

# ── Cleanup trap ───────────────────────────────────────────────────
PIDS=()
cleanup() {
  echo ""
  echo "[shutdown] Stopping all processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
  echo "[shutdown] Done."
}
trap cleanup EXIT INT TERM

# ── Start inference engine (local only) ───────────────────────────
if [[ "$INFERENCE_PROVIDER" == "local" ]]; then
  echo "[launch] Starting inference engine..."
  ./start-inference.sh &
  PIDS+=($!)
  echo "[launch]   inference PID=${PIDS[-1]} (port ${CONDUCTOR_INFERENCE_PORT:-8080})"

  # Wait for inference to be ready
  echo "[launch] Waiting for inference engine to load model..."
  INFERENCE_URL="http://localhost:${CONDUCTOR_INFERENCE_PORT:-8080}"
  for i in $(seq 1 120); do
    if curl -sf "$INFERENCE_URL/health" >/dev/null 2>&1; then
      echo "[launch]   Inference engine ready (${i}s)"
      break
    fi
    if [[ $i -eq 120 ]]; then
      echo "[ERROR] Inference engine didn't start within 120s."
      exit 1
    fi
    sleep 1
  done
else
  echo "[launch] Skipping local inference engine (using $INFERENCE_PROVIDER API)"
fi

# ── Start gateway ──────────────────────────────────────────────────
echo "[launch] Starting gateway..."
uvicorn gateway.server:app --host 0.0.0.0 --port 9090 &
PIDS+=($!)
echo "[launch]   gateway PID=${PIDS[-1]} (port 9090)"
sleep 2

# ── Start orchestrator ─────────────────────────────────────────────
echo "[launch] Starting orchestrator (project: $PROJECT)..."
python3 -m orchestrator.conductor \
  --project "$PROJECT" \
  --config "$CONFIG" &
PIDS+=($!)
echo "[launch]   orchestrator PID=${PIDS[-1]}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║            Conductor Running             ║"
echo "╠══════════════════════════════════════════╣"
if [[ "$INFERENCE_PROVIDER" == "local" ]]; then
echo "║  inference:    http://localhost:${CONDUCTOR_INFERENCE_PORT:-8080}       ║"
else
echo "║  inference:    $INFERENCE_PROVIDER API               ║"
fi
echo "║  gateway:      http://localhost:9090       ║"
echo "║  orchestrator: project '$PROJECT'          "
echo "╠══════════════════════════════════════════╣"
echo "║  Drop .md files into conductor/inbox/    ║"
echo "║  Press Ctrl+C to stop all processes.     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

wait
