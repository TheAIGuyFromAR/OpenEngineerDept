#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Conductor Phase 0 ==="

echo "Starting inference engine..."
./start-inference.sh &
INFERENCE_PID=$!
sleep 10

echo "Starting gateway..."
uvicorn gateway.server:app --host 0.0.0.0 --port 9090 &
GATEWAY_PID=$!
sleep 2

echo "Starting orchestrator..."
python -m orchestrator.conductor \
  --project example \
  --config projects/example/conductor.yaml &
CONDUCTOR_PID=$!

echo ""
echo "Running:"
echo "  inference:    PID $INFERENCE_PID (port 8080)"
echo "  gateway:      PID $GATEWAY_PID (port 9090)"
echo "  orchestrator: PID $CONDUCTOR_PID"
echo ""
echo "Drop .md files into your Obsidian vault's conductor/inbox/ to submit tasks."

cleanup() {
  echo "Shutting down..."
  kill "$CONDUCTOR_PID" "$GATEWAY_PID" "$INFERENCE_PID" 2>/dev/null || true
  wait
}
trap cleanup EXIT INT TERM

wait
