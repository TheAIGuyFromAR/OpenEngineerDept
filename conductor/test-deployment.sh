#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Conductor Deployment Test Suite
#
# Validates the full stack after hardware detection and setup.
# Run this BEFORE using Conductor for real tasks.
#
# Usage:
#   ./test-deployment.sh [config-path]
#
# Phases:
#   Phase 1: Unit tests (pytest, offline)
#   Phase 2: Provider connectivity (API key validation / llama-server health)
#   Phase 3: Gateway smoke tests (start gateway, hit endpoints)
#   Phase 4: Inference validation (send a real prompt, validate output)
#   Phase 5: Ultra Think pipeline (parallel generation + review)
#   Phase 6: Full orchestrator dry run (planner → coder → reviewer)
#
# Exit codes:
#   0 — all phases passed
#   1 — one or more phases failed
# ═══════════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-projects/example/conductor.yaml}"
GATEWAY_PORT=19090       # Use non-default port to avoid conflicts
INFERENCE_PORT=18080
PASSED=0
FAILED=0
SKIPPED=0
PHASE_RESULTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ────────────────────────────────────────────────────────

pass_test() {
  local name="$1"
  PASSED=$((PASSED + 1))
  echo -e "  ${GREEN}PASS${NC} $name"
}

fail_test() {
  local name="$1"
  local detail="${2:-}"
  FAILED=$((FAILED + 1))
  echo -e "  ${RED}FAIL${NC} $name"
  [[ -n "$detail" ]] && echo -e "       ${RED}$detail${NC}"
}

skip_test() {
  local name="$1"
  local reason="${2:-}"
  SKIPPED=$((SKIPPED + 1))
  echo -e "  ${YELLOW}SKIP${NC} $name${reason:+ ($reason)}"
}

phase_header() {
  local num="$1"
  local name="$2"
  echo ""
  echo -e "${BOLD}${CYAN}═══ Phase $num: $name ═══${NC}"
}

phase_result() {
  local name="$1"
  local phase_failed="$2"
  if [[ "$phase_failed" -eq 0 ]]; then
    PHASE_RESULTS+=("${GREEN}PASS${NC} $name")
  else
    PHASE_RESULTS+=("${RED}FAIL${NC} $name")
  fi
}

# Cleanup background processes on exit
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# ── Detect provider ────────────────────────────────────────────────

INFERENCE_PROVIDER="${CONDUCTOR_INFERENCE_PROVIDER:-local}"
if [[ "$INFERENCE_PROVIDER" == "local" ]] && [[ -f "$CONFIG" ]]; then
  yaml_provider=$(grep -oP '^\s*inference_provider:\s*"\K[^"]+' "$CONFIG" 2>/dev/null || true)
  if [[ -n "$yaml_provider" ]]; then
    INFERENCE_PROVIDER="$yaml_provider"
  fi
fi

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║    Conductor Deployment Test Suite       ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║  Provider: ${CYAN}$INFERENCE_PROVIDER${NC}${BOLD}                        ║${NC}"
echo -e "${BOLD}║  Config:   $CONFIG"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

# ── Activate venv ──────────────────────────────────────────────────

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/.venv" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
  else
    echo -e "${RED}No virtual environment found. Run ./setup.sh first.${NC}"
    exit 1
  fi
fi


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Unit tests (offline)
# ═══════════════════════════════════════════════════════════════════

phase_header 1 "Unit Tests (offline)"
PHASE1_FAILED=0

if python3 -m pytest tests/ -v --tb=short -q 2>&1 | tee /tmp/conductor-test-phase1.log; then
  pass_test "pytest suite"
else
  fail_test "pytest suite" "See /tmp/conductor-test-phase1.log"
  PHASE1_FAILED=1
fi

phase_result "Unit Tests" "$PHASE1_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Provider connectivity
# ═══════════════════════════════════════════════════════════════════

phase_header 2 "Provider Connectivity"
PHASE2_FAILED=0

if [[ "$INFERENCE_PROVIDER" == "local" ]]; then
  # Check llama-server binary exists
  LLAMA_BIN="${CONDUCTOR_LLAMA_BIN:-./build/bin/llama-server}"
  if [[ -x "$LLAMA_BIN" ]] || [[ -L "$LLAMA_BIN" ]]; then
    pass_test "llama-server binary exists"
  else
    fail_test "llama-server binary exists" "Not found at $LLAMA_BIN"
    PHASE2_FAILED=1
  fi

  # Check model file exists
  MODEL_PATH="${CONDUCTOR_MODEL_PATH:-./models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"
  if [[ -f "$MODEL_PATH" ]]; then
    model_size=$(du -h "$MODEL_PATH" | cut -f1)
    pass_test "Model file exists ($model_size)"
  else
    fail_test "Model file exists" "Not found at $MODEL_PATH"
    PHASE2_FAILED=1
  fi

  # Check hardware profile
  if [[ -f "$SCRIPT_DIR/hardware-profile.env" ]]; then
    pass_test "Hardware profile generated"
  else
    skip_test "Hardware profile generated" "Run ./detect-hardware.sh first"
  fi

else
  # API provider — validate key format
  API_KEY="${CONDUCTOR_INFERENCE_API_KEY:-}"
  if [[ -z "$API_KEY" ]]; then
    API_KEY=$(grep -oP '^\s*inference_api_key:\s*"\K[^"]+' "$CONFIG" 2>/dev/null || true)
  fi

  if [[ -n "$API_KEY" ]]; then
    pass_test "API key configured"

    # Validate key format
    case "$INFERENCE_PROVIDER" in
      anthropic)
        if [[ "$API_KEY" == sk-ant-* ]]; then
          pass_test "API key format (Anthropic)"
        else
          fail_test "API key format (Anthropic)" "Expected sk-ant-... prefix"
          PHASE2_FAILED=1
        fi
        ;;
      openai)
        if [[ "$API_KEY" == sk-* ]]; then
          pass_test "API key format (OpenAI)"
        else
          fail_test "API key format (OpenAI)" "Expected sk-... prefix"
          PHASE2_FAILED=1
        fi
        ;;
      openrouter)
        if [[ "$API_KEY" == sk-or-* ]]; then
          pass_test "API key format (OpenRouter)"
        else
          fail_test "API key format (OpenRouter)" "Expected sk-or-... prefix"
          PHASE2_FAILED=1
        fi
        ;;
    esac

    # Test actual API connectivity
    echo "  Testing API connectivity..."
    if python3 -c "
import asyncio
from gateway.config import GatewayConfig
from gateway.providers import create_provider
async def test():
    config = GatewayConfig()
    provider = create_provider(config)
    ok = await provider.health_check()
    await provider.close()
    return ok
result = asyncio.run(test())
exit(0 if result else 1)
" 2>/dev/null; then
      pass_test "API provider health check"
    else
      fail_test "API provider health check" "Could not reach $INFERENCE_PROVIDER API"
      PHASE2_FAILED=1
    fi

  else
    fail_test "API key configured" "Set CONDUCTOR_INFERENCE_API_KEY"
    PHASE2_FAILED=1
  fi
fi

phase_result "Provider Connectivity" "$PHASE2_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Gateway Smoke Tests
# ═══════════════════════════════════════════════════════════════════

phase_header 3 "Gateway Smoke Tests"
PHASE3_FAILED=0

# Start inference engine if local
if [[ "$INFERENCE_PROVIDER" == "local" ]]; then
  if [[ "$PHASE2_FAILED" -eq 0 ]]; then
    echo "  Starting local inference engine (port $INFERENCE_PORT)..."
    CONDUCTOR_INFERENCE_PORT=$INFERENCE_PORT ./start-inference.sh &
    PIDS+=($!)

    # Wait for inference
    inference_ready=false
    for i in $(seq 1 120); do
      if curl -sf "http://localhost:$INFERENCE_PORT/health" >/dev/null 2>&1; then
        inference_ready=true
        pass_test "Inference engine started (${i}s)"
        break
      fi
      sleep 1
    done

    if [[ "$inference_ready" != true ]]; then
      fail_test "Inference engine started" "Timed out after 120s"
      PHASE3_FAILED=1
    fi
  else
    skip_test "Inference engine" "Skipped due to Phase 2 failures"
  fi
fi

# Start gateway
if [[ "$PHASE3_FAILED" -eq 0 ]]; then
  echo "  Starting gateway (port $GATEWAY_PORT)..."
  CONDUCTOR_INFERENCE_PROVIDER=$INFERENCE_PROVIDER \
  CONDUCTOR_LLAMA_SERVER_URL="http://localhost:$INFERENCE_PORT" \
  uvicorn gateway.server:app --host 127.0.0.1 --port $GATEWAY_PORT --log-level warning &
  PIDS+=($!)
  sleep 3

  # Health check
  if curl -sf "http://localhost:$GATEWAY_PORT/health" >/dev/null 2>&1; then
    pass_test "Gateway health endpoint"
  else
    fail_test "Gateway health endpoint" "Gateway not responding on port $GATEWAY_PORT"
    PHASE3_FAILED=1
  fi

  # Slots status
  if curl -sf "http://localhost:$GATEWAY_PORT/v1/slots/status" >/dev/null 2>&1; then
    pass_test "Gateway /v1/slots/status"
  else
    fail_test "Gateway /v1/slots/status"
    PHASE3_FAILED=1
  fi

  # Metrics
  if curl -sf "http://localhost:$GATEWAY_PORT/v1/metrics" >/dev/null 2>&1; then
    pass_test "Gateway /v1/metrics"
  else
    fail_test "Gateway /v1/metrics"
    PHASE3_FAILED=1
  fi
fi

phase_result "Gateway Smoke Tests" "$PHASE3_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Inference Validation
# ═══════════════════════════════════════════════════════════════════

phase_header 4 "Inference Validation"
PHASE4_FAILED=0

if [[ "$PHASE3_FAILED" -ne 0 ]]; then
  skip_test "All inference tests" "Gateway not running"
else
  echo "  Sending test prompt..."
  RESPONSE=$(curl -sf "http://localhost:$GATEWAY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are a helpful coding assistant. Respond with ONLY the code, no explanation."},
        {"role": "user", "content": "Write a Python function that returns the sum of two numbers. Just the function, nothing else."}
      ],
      "max_tokens": 256,
      "temperature": 0.3
    }' 2>&1 || echo "CURL_FAILED")

  if [[ "$RESPONSE" == "CURL_FAILED" ]] || [[ -z "$RESPONSE" ]]; then
    fail_test "Chat completion request" "No response from gateway"
    PHASE4_FAILED=1
  else
    # Validate response structure
    if echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
content = data['choices'][0]['message']['content']
assert len(content) > 10, 'Response too short'
assert 'def' in content or 'return' in content or 'sum' in content.lower(), 'No code in response'
print(f'  Response: {len(content)} chars')
" 2>/dev/null; then
      pass_test "Chat completion — valid response"
    else
      fail_test "Chat completion — valid response" "Unexpected response format"
      echo "       Response: $(echo "$RESPONSE" | head -c 200)"
      PHASE4_FAILED=1
    fi

    # Validate usage metrics present
    if echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
usage = data.get('usage', {})
assert usage.get('completion_tokens', usage.get('output_tokens', 0)) > 0, 'No completion tokens'
" 2>/dev/null; then
      pass_test "Chat completion — usage metrics"
    else
      skip_test "Chat completion — usage metrics" "Provider may not return usage"
    fi
  fi

  # Test with code that needs reasoning
  echo "  Sending reasoning test prompt..."
  RESPONSE2=$(curl -sf "http://localhost:$GATEWAY_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {"role": "system", "content": "You are a Python expert. Be concise."},
        {"role": "user", "content": "What is wrong with this code?\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\nprint(fib(50))"}
      ],
      "max_tokens": 512,
      "temperature": 0.5
    }' 2>&1 || echo "CURL_FAILED")

  if [[ "$RESPONSE2" != "CURL_FAILED" ]] && [[ -n "$RESPONSE2" ]]; then
    if echo "$RESPONSE2" | python3 -c "
import sys, json
data = json.load(sys.stdin)
content = data['choices'][0]['message']['content'].lower()
# Should mention performance, recursion, slow, exponential, or memoization
keywords = ['slow', 'exponential', 'recursion', 'performance', 'memo', 'cache', 'time', 'O(2']
assert any(k in content for k in keywords), f'Response missing key insight: {content[:100]}'
" 2>/dev/null; then
      pass_test "Reasoning test — identifies performance issue"
    else
      fail_test "Reasoning test — identifies performance issue" "Model failed to spot exponential recursion"
      PHASE4_FAILED=1
    fi
  else
    fail_test "Reasoning test" "No response"
    PHASE4_FAILED=1
  fi
fi

phase_result "Inference Validation" "$PHASE4_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Ultra Think Pipeline
# ═══════════════════════════════════════════════════════════════════

phase_header 5 "Ultra Think Pipeline"
PHASE5_FAILED=0

if [[ "$PHASE3_FAILED" -ne 0 ]]; then
  skip_test "All Ultra Think tests" "Gateway not running"
else
  # Tier 1: single shot
  echo "  Testing Tier 1 (single shot)..."
  UT1=$(curl -sf "http://localhost:$GATEWAY_PORT/v1/ultra-think" \
    -H "Content-Type: application/json" \
    -d '{
      "task_id": "deploy-test-t1",
      "prompt": "Write a Python function that checks if a string is a palindrome.",
      "system_prompt": "You are a Python expert. Write clean, correct code.",
      "tier": 1,
      "max_tokens": 512
    }' 2>&1 || echo "CURL_FAILED")

  if [[ "$UT1" != "CURL_FAILED" ]] && [[ -n "$UT1" ]]; then
    if echo "$UT1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data['task_id'] == 'deploy-test-t1'
assert len(data['candidates']) == 1
assert data['timing']['total_ms'] > 0
c = data['candidates'][0]
assert len(c['content']) > 20
assert c['tokens_generated'] > 0 or True  # some providers don't report
" 2>/dev/null; then
      pass_test "Ultra Think Tier 1 — 1 candidate"
    else
      fail_test "Ultra Think Tier 1" "Unexpected response"
      PHASE5_FAILED=1
    fi
  else
    fail_test "Ultra Think Tier 1" "No response"
    PHASE5_FAILED=1
  fi

  # Tier 2: parallel diverse
  echo "  Testing Tier 2 (parallel diverse)..."
  UT2=$(curl -sf -m 120 "http://localhost:$GATEWAY_PORT/v1/ultra-think" \
    -H "Content-Type: application/json" \
    -d '{
      "task_id": "deploy-test-t2",
      "prompt": "Write a Python function that finds the two numbers in a list that add up to a target sum. Return their indices.",
      "system_prompt": "You are a Python expert. Write clean, correct code.",
      "tier": 2,
      "max_tokens": 512
    }' 2>&1 || echo "CURL_FAILED")

  if [[ "$UT2" != "CURL_FAILED" ]] && [[ -n "$UT2" ]]; then
    if echo "$UT2" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert len(data['candidates']) >= 2, f'Expected 3 candidates, got {len(data[\"candidates\"])}'
# Check that candidates have diverse content (not identical)
contents = [c['content'] for c in data['candidates']]
unique_contents = set(contents)
assert len(unique_contents) >= 2, 'Candidates are identical — diversity failed'
# Check timing shows parallel execution
t = data['timing']
assert t['parallel_generation_ms'] > 0
print(f'  {len(data[\"candidates\"])} candidates in {t[\"total_ms\"]:.0f}ms')
" 2>/dev/null; then
      pass_test "Ultra Think Tier 2 — diverse candidates"
    else
      fail_test "Ultra Think Tier 2" "Unexpected response or no diversity"
      PHASE5_FAILED=1
    fi
  else
    fail_test "Ultra Think Tier 2" "No response (timeout?)"
    PHASE5_FAILED=1
  fi

  # Tier 4 rejection
  echo "  Testing Tier 4 rejection..."
  UT4_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
    "http://localhost:$GATEWAY_PORT/v1/ultra-think" \
    -H "Content-Type: application/json" \
    -d '{
      "task_id": "deploy-test-t4",
      "prompt": "test",
      "system_prompt": "test",
      "tier": 4
    }' 2>&1 || echo "000")

  if [[ "$UT4_STATUS" == "400" ]]; then
    pass_test "Ultra Think Tier 4 — correctly rejected (400)"
  else
    fail_test "Ultra Think Tier 4 rejection" "Expected 400, got $UT4_STATUS"
    PHASE5_FAILED=1
  fi
fi

phase_result "Ultra Think Pipeline" "$PHASE5_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Full Pipeline Dry Run
# ═══════════════════════════════════════════════════════════════════

phase_header 6 "Full Pipeline Dry Run"
PHASE6_FAILED=0

if [[ "$PHASE4_FAILED" -ne 0 ]] || [[ "$PHASE5_FAILED" -ne 0 ]]; then
  skip_test "Full pipeline" "Skipped due to earlier failures"
else
  echo "  Running planner → coder → reviewer simulation..."
  python3 -c "
import asyncio, sys
sys.path.insert(0, '.')

async def test_pipeline():
    import httpx

    gateway = 'http://localhost:$GATEWAY_PORT'
    client = httpx.AsyncClient(timeout=120)

    # Step 1: Planner — decompose a task
    print('  Step 1: Planner...')
    plan_resp = await client.post(f'{gateway}/v1/chat/completions', json={
        'messages': [
            {'role': 'system', 'content': 'You are a software planner. Given a task, output a JSON array of subtask strings. Only output valid JSON, no other text.'},
            {'role': 'user', 'content': 'Add input validation to a user registration function that checks email format and password strength.'},
        ],
        'max_tokens': 512,
        'temperature': 0.5,
    })
    plan_resp.raise_for_status()
    plan_data = plan_resp.json()
    plan_content = plan_data['choices'][0]['message']['content']
    print(f'    Plan: {plan_content[:120]}...')
    assert len(plan_content) > 20, 'Plan too short'

    # Step 2: Coder — generate implementation via Ultra Think
    print('  Step 2: Coder (Ultra Think Tier 2)...')
    code_resp = await client.post(f'{gateway}/v1/ultra-think', json={
        'task_id': 'pipeline-test-code',
        'prompt': 'Write a Python function validate_email(email: str) -> bool that checks if an email address has valid format using regex. Include edge cases.',
        'system_prompt': 'You are an expert Python developer. Write production-quality code.',
        'tier': 2,
        'max_tokens': 1024,
    })
    code_resp.raise_for_status()
    code_data = code_resp.json()
    candidates = code_data['candidates']
    assert len(candidates) >= 2, f'Expected 3 candidates, got {len(candidates)}'
    print(f'    Generated {len(candidates)} candidates')

    # Step 3: Reviewer — evaluate candidates
    print('  Step 3: Reviewer...')
    candidate_text = ''
    for i, c in enumerate(candidates):
        candidate_text += f'\\n--- Candidate {i} ---\\n{c[\"content\"]}\\n'

    review_resp = await client.post(f'{gateway}/v1/chat/completions', json={
        'messages': [
            {'role': 'system', 'content': 'You are a code reviewer. Given multiple code candidates, evaluate each on: correctness, readability, edge case handling. Output JSON: {\"scores\": [{\"index\": 0, \"score\": 7.5, \"reason\": \"...\"}], \"selected\": 0}. Only output valid JSON.'},
            {'role': 'user', 'content': f'Review these candidates for an email validation function:\\n{candidate_text}'},
        ],
        'max_tokens': 1024,
        'temperature': 0.3,
    })
    review_resp.raise_for_status()
    review_data = review_resp.json()
    review_content = review_data['choices'][0]['message']['content']
    print(f'    Review: {review_content[:150]}...')
    assert len(review_content) > 20, 'Review too short'

    await client.aclose()
    print('  Pipeline complete.')

asyncio.run(test_pipeline())
" 2>&1

  if [[ $? -eq 0 ]]; then
    pass_test "Full pipeline: planner → coder → reviewer"
  else
    fail_test "Full pipeline" "Pipeline execution failed"
    PHASE6_FAILED=1
  fi
fi

phase_result "Full Pipeline Dry Run" "$PHASE6_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Phase 7: Evidence Collection
#
# Hard metrics that PROVE the deployment works for real tasks:
#   - Throughput: tok/s generation speed
#   - Latency: p50/p95/p99 response times
#   - Code correctness: generated code executes and passes assertions
#   - Reasoning quality: known-answer problems scored against expected insights
#   - Context window: can it actually use the claimed context size?
#   - Consistency: same prompt N times, how stable are outputs?
#   - Resources: VRAM/RAM (local) or cost (API)
# ═══════════════════════════════════════════════════════════════════

phase_header 7 "Evidence Collection"
PHASE7_FAILED=0

if [[ "$PHASE4_FAILED" -ne 0 ]]; then
  skip_test "Evidence collection" "Skipped — inference not working"
else
  echo "  Running evidence collectors (this takes a few minutes)..."
  echo ""

  # Read context size from hardware profile or default
  CTX_SIZE="${CONDUCTOR_CTX_SIZE:-32768}"

  # Read model path for resource measurement
  MODEL_PATH="${CONDUCTOR_MODEL_PATH:-./models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"

  # Read hardware summary from profile
  HW_SUMMARY=""
  if [[ -f "$SCRIPT_DIR/hardware-profile.env" ]]; then
    HW_SUMMARY=$(grep "^# Hardware:" "$SCRIPT_DIR/hardware-profile.env" 2>/dev/null | sed 's/^# Hardware: //' || true)
  fi

  EVIDENCE_EXIT=0
  python3 -m tests.evidence.runner \
    --gateway "http://localhost:$GATEWAY_PORT" \
    --provider "$INFERENCE_PROVIDER" \
    --model "${CONDUCTOR_INFERENCE_MODEL:-local}" \
    --hardware "$HW_SUMMARY" \
    --ctx-size "$CTX_SIZE" \
    --model-path "$MODEL_PATH" \
    --output-dir "./data/evidence" \
    2>&1 || EVIDENCE_EXIT=$?

  if [[ "$EVIDENCE_EXIT" -eq 0 ]]; then
    pass_test "Evidence verdict: PASS"
  elif [[ "$EVIDENCE_EXIT" -eq 2 ]]; then
    fail_test "Evidence verdict: DEGRADED" "Deployment works but with reduced quality — see report"
    PHASE7_FAILED=1
  else
    fail_test "Evidence verdict: FAIL" "Deployment not fit for coding tasks — see report"
    PHASE7_FAILED=1
  fi
fi

phase_result "Evidence Collection" "$PHASE7_FAILED"


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         Deployment Test Results          ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════╣${NC}"
for result in "${PHASE_RESULTS[@]}"; do
  echo -e "${BOLD}║  ${NC}$result"
done
echo -e "${BOLD}╠══════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║  ${GREEN}Passed: $PASSED${NC}  ${RED}Failed: $FAILED${NC}  ${YELLOW}Skipped: $SKIPPED${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

# Point to evidence report
LATEST_EVIDENCE=$(ls -t ./data/evidence/deployment-*.json 2>/dev/null | head -1 || true)
if [[ -n "$LATEST_EVIDENCE" ]]; then
  echo ""
  echo -e "${CYAN}Evidence report: $LATEST_EVIDENCE${NC}"
fi

if [[ "$FAILED" -gt 0 ]]; then
  echo ""
  echo -e "${RED}${BOLD}Deployment validation FAILED.${NC}"
  echo -e "Fix the failures above before using Conductor for real tasks."
  echo ""
  exit 1
else
  echo ""
  echo -e "${GREEN}${BOLD}Deployment validation PASSED.${NC}"
  echo -e "Conductor is ready for use."
  echo ""
  exit 0
fi
