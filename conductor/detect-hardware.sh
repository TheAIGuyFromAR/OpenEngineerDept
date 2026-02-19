#!/usr/bin/env bash
# Conductor Phase 0 — Hardware Detection & Model Selection
#
# Scans your system (GPUs, RAM, CPU) and generates hardware-profile.env
# with optimal settings for the inference engine.
#
# Usage:
#   ./detect-hardware.sh              # Auto-detect and write profile
#   ./detect-hardware.sh --dry-run    # Print what would be written
#   ./detect-hardware.sh --json       # Output raw detection as JSON
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_PATH="$SCRIPT_DIR/hardware-profile.env"
DRY_RUN=false
JSON_OUT=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --json)    JSON_OUT=true ;;
    --help|-h)
      echo "Usage: ./detect-hardware.sh [--dry-run] [--json]"
      exit 0
      ;;
  esac
done

# ═══════════════════════════════════════════════════════════════════
# Detection functions
# ═══════════════════════════════════════════════════════════════════

detect_cpu() {
  local cores threads model_name

  if [[ -f /proc/cpuinfo ]]; then
    cores=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 0)
    model_name=$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo "unknown")
  elif command -v sysctl >/dev/null 2>&1; then
    # macOS
    cores=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 0)
    model_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
  else
    cores=0
    model_name="unknown"
  fi

  threads=$cores  # logical cores = threads for our purposes
  echo "$cores|$model_name"
}

detect_ram() {
  local total_mb

  if [[ -f /proc/meminfo ]]; then
    total_kb=$(grep 'MemTotal' /proc/meminfo | awk '{print $2}')
    total_mb=$((total_kb / 1024))
  elif command -v sysctl >/dev/null 2>&1; then
    total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    total_mb=$((total_bytes / 1024 / 1024))
  else
    total_mb=0
  fi

  echo "$total_mb"
}

detect_gpus() {
  # Returns: count|gpu0_name|gpu0_vram_mb|gpu0_compute|gpu1_name|gpu1_vram_mb|gpu1_compute|...
  #
  # Tries nvidia-smi first, then falls back to checking for Apple Silicon.

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total,compute_cap \
      --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [[ -z "$gpu_info" ]]; then
      echo "0"
      return
    fi

    local count=0
    local result=""

    while IFS=',' read -r name vram_mb compute; do
      name=$(echo "$name" | xargs)
      vram_mb=$(echo "$vram_mb" | xargs)
      compute=$(echo "$compute" | xargs)
      result="${result}|${name}|${vram_mb}|${compute}"
      count=$((count + 1))
    done <<< "$gpu_info"

    echo "${count}${result}"
  elif [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple"; then
    # Apple Silicon unified memory — treat as 1 "GPU"
    local chip
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    local total_bytes
    total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    local total_mb=$((total_bytes / 1024 / 1024))
    # Apple Silicon shares RAM with GPU. Usable GPU memory ~75% of total.
    local gpu_mb=$((total_mb * 75 / 100))
    echo "1|$chip (Unified)|${gpu_mb}|metal"
  else
    echo "0"
  fi
}

# ═══════════════════════════════════════════════════════════════════
# Model selection logic
# ═══════════════════════════════════════════════════════════════════

# Known quantization tiers for Qwen3-Coder-Next from unsloth/Qwen3-Coder-Next-GGUF:
#   Q2_K_XL  — ~10GB, ~65-75% quality (DO NOT USE for coding)
#   Q3_K_XL  — ~14GB, ~80-85% quality (DO NOT USE for coding)
#   Q4_K_XL  — ~19GB, ~90-92% quality ← MINIMUM for coding tasks
#   Q5_K_XL  — ~23GB, ~95% quality
#   Q6_K     — ~27GB, ~97% quality
#   Q8_0     — ~35GB, ~99% quality (near-lossless)
#   f16      — ~65GB, 100% quality (baseline)
#
# PRINCIPLE: For coding, a larger model at Q4_K beats a smaller model at Q8_0.
# But below Q4_K there's a quality cliff — quantization errors compound through
# long reasoning chains and corrupt MoE expert routing decisions.
# NEVER go below Q4_K for coding. Instead, trade context size, parallel slots,
# and GPU layers to keep quant quality at Q4_K or above.
#
# Memory budget: model weights + KV cache + overhead
# KV cache at q8_0, ctx=32768, 5 slots ≈ varies by model size
# Rule of thumb: model needs ~1.2x file size in VRAM at runtime

select_model_config() {
  local total_gpu_mb=$1
  local total_ram_mb=$2
  local gpu_count=$3
  local compute_cap=$4  # e.g. "6.1", "7.5", "8.9", "metal"

  # Compute available memory for model (leave headroom for KV cache + OS)
  # KV cache budget: ~4GB for 5 slots @ 32k ctx @ q8_0 (varies by model)
  local kv_budget_mb=4096
  local os_overhead_mb=2048
  local available_gpu_mb=$((total_gpu_mb - kv_budget_mb - os_overhead_mb))

  # Model selection thresholds (approximate loaded size with 1.2x overhead)
  local quant=""
  local model_file=""
  local ctx_size=32768
  local n_parallel=5
  local n_gpu_layers=99
  local gpu_offload=""
  local cache_k="q8_0"
  local cache_v="q8_0"
  local batch_size=4096
  local ubatch_size=4096

  if [[ "$available_gpu_mb" -ge 78000 ]]; then
    # 80GB+ (A100, H100) — luxury tier
    quant="f16"
    model_file="*f16*.gguf"
    ctx_size=65536
  elif [[ "$available_gpu_mb" -ge 38000 ]]; then
    # 40-78GB (A6000, 2x P40 fully loaded)
    quant="Q8_0"
    model_file="*Q8_0*.gguf"
    ctx_size=49152
  elif [[ "$available_gpu_mb" -ge 28000 ]]; then
    # 30-40GB (A10, large single GPU)
    quant="Q6_K"
    model_file="*Q6_K*.gguf"
  elif [[ "$available_gpu_mb" -ge 20000 ]]; then
    # 22-30GB (2x P40 with offload, RTX 3090/4090)
    quant="Q5_K_XL"
    model_file="*Q5_K_XL*.gguf"
  elif [[ "$available_gpu_mb" -ge 14000 ]]; then
    # 16-22GB (single P40, RTX 4080/3080)
    quant="Q4_K_XL"
    model_file="*UD-Q4_K_XL*.gguf"
  elif [[ "$available_gpu_mb" -ge 10000 ]]; then
    # 12-16GB (RTX 3060 12GB) — Q4_K_XL is the floor for coding
    # Trade context and parallelism to stay at Q4_K
    quant="Q4_K_XL"
    model_file="*UD-Q4_K_XL*.gguf"
    ctx_size=16384
    n_parallel=3
    cache_k="q4_0"
    cache_v="q4_0"
  elif [[ "$available_gpu_mb" -ge 6000 ]]; then
    # 8-12GB (RTX 3060 8GB, RTX 2080) — aggressive tradeoffs to hold Q4_K
    quant="Q4_K_XL"
    model_file="*UD-Q4_K_XL*.gguf"
    ctx_size=8192
    n_parallel=2
    n_gpu_layers=40      # partial offload to CPU
    cache_k="q4_0"
    cache_v="q4_0"
  else
    # <8GB VRAM — heavy CPU offload, still Q4_K minimum
    quant="Q4_K_XL"
    model_file="*UD-Q4_K_XL*.gguf"
    n_gpu_layers=20
    ctx_size=8192
    n_parallel=2
    batch_size=2048
    ubatch_size=2048
    cache_k="q4_0"
    cache_v="q4_0"
  fi

  # Compute capability adjustments
  case "$compute_cap" in
    6.*)
      # Pascal (P40, P100, GTX 1080) — no tensor cores, FP16 is slower
      # MoE experts to CPU to save VRAM
      gpu_offload=".ffn_.*_exps.=CPU"
      ;;
    7.*)
      # Volta/Turing (V100, T4, RTX 2080) — has tensor cores
      gpu_offload=""
      ;;
    8.*|9.*)
      # Ampere/Hopper/Ada — full tensor core support
      gpu_offload=""
      ;;
    metal)
      # Apple Silicon — unified memory, no offload needed
      gpu_offload=""
      n_gpu_layers=99
      ;;
  esac

  # Multi-GPU: if >1 GPU, adjust parallel slots
  if [[ "$gpu_count" -ge 2 ]]; then
    # More slots available with more GPUs
    if [[ "$n_parallel" -lt 5 ]]; then
      n_parallel=5
    fi
  fi

  # RAM-based upgrade: if low VRAM but lots of RAM, we can fit the model
  # in RAM via CPU layers — this is slower but preserves Q4_K quality
  if [[ "$available_gpu_mb" -lt 14000 ]] && [[ "$total_ram_mb" -ge 64000 ]]; then
    # Enough RAM to hold Q4_K weights in CPU memory
    # More GPU layers for speed, CPU handles the rest
    if [[ "$total_ram_mb" -ge 96000 ]]; then
      # 96GB+ RAM: can afford larger context
      ctx_size=16384
    fi
  fi

  echo "${quant}|${model_file}|${ctx_size}|${n_parallel}|${n_gpu_layers}|${gpu_offload}|${cache_k}|${cache_v}|${batch_size}|${ubatch_size}"
}

# ═══════════════════════════════════════════════════════════════════
# Main detection
# ═══════════════════════════════════════════════════════════════════

echo "Scanning hardware..."
echo ""

# CPU
cpu_info=$(detect_cpu)
CPU_CORES=$(echo "$cpu_info" | cut -d'|' -f1)
CPU_MODEL=$(echo "$cpu_info" | cut -d'|' -f2)
echo "CPU:  $CPU_MODEL ($CPU_CORES threads)"

# RAM
RAM_MB=$(detect_ram)
RAM_GB=$((RAM_MB / 1024))
echo "RAM:  ${RAM_GB}GB"

# GPUs
gpu_info=$(detect_gpus)
GPU_COUNT=$(echo "$gpu_info" | cut -d'|' -f1)

TOTAL_GPU_MB=0
GPU_COMPUTE_CAP=""

if [[ "$GPU_COUNT" -gt 0 ]]; then
  # Parse GPU details
  remaining="$gpu_info"
  # Strip the count prefix
  remaining="${remaining#*|}"

  for i in $(seq 1 "$GPU_COUNT"); do
    gpu_name=$(echo "$remaining" | cut -d'|' -f1)
    gpu_vram=$(echo "$remaining" | cut -d'|' -f2)
    gpu_compute=$(echo "$remaining" | cut -d'|' -f3)
    remaining=$(echo "$remaining" | cut -d'|' -f4-)

    echo "GPU $i: $gpu_name — ${gpu_vram}MB VRAM (compute $gpu_compute)"
    TOTAL_GPU_MB=$((TOTAL_GPU_MB + gpu_vram))

    # Use the lowest compute capability (bottleneck)
    if [[ -z "$GPU_COMPUTE_CAP" ]] || [[ "$gpu_compute" < "$GPU_COMPUTE_CAP" ]]; then
      GPU_COMPUTE_CAP="$gpu_compute"
    fi
  done
  echo "Total GPU VRAM: $((TOTAL_GPU_MB / 1024))GB across $GPU_COUNT GPU(s)"
else
  echo "GPU:  None detected (CPU-only mode)"
  GPU_COMPUTE_CAP="none"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# Select optimal configuration
# ═══════════════════════════════════════════════════════════════════

config=$(select_model_config "$TOTAL_GPU_MB" "$RAM_MB" "$GPU_COUNT" "$GPU_COMPUTE_CAP")

QUANT=$(echo "$config" | cut -d'|' -f1)
MODEL_GLOB=$(echo "$config" | cut -d'|' -f2)
CTX_SIZE=$(echo "$config" | cut -d'|' -f3)
N_PARALLEL=$(echo "$config" | cut -d'|' -f4)
N_GPU_LAYERS=$(echo "$config" | cut -d'|' -f5)
GPU_OFFLOAD=$(echo "$config" | cut -d'|' -f6)
CACHE_K=$(echo "$config" | cut -d'|' -f7)
CACHE_V=$(echo "$config" | cut -d'|' -f8)
BATCH_SIZE=$(echo "$config" | cut -d'|' -f9)
UBATCH_SIZE=$(echo "$config" | cut -d'|' -f10)

echo "═══════════════════════════════════════════"
echo "Recommended configuration:"
echo "═══════════════════════════════════════════"
echo "  Quantization:   $QUANT"
echo "  Model glob:     $MODEL_GLOB"
echo "  Context size:   $CTX_SIZE"
echo "  Parallel slots: $N_PARALLEL"
echo "  GPU layers:     $N_GPU_LAYERS"
echo "  GPU offload:    ${GPU_OFFLOAD:-none}"
echo "  KV cache type:  k=$CACHE_K v=$CACHE_V"
echo "  Batch size:     $BATCH_SIZE / $UBATCH_SIZE"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Assess local inference quality and recommend API if needed
# ═══════════════════════════════════════════════════════════════════

API_RECOMMENDED=false
API_REASON=""

# Q4_K_XL with reduced context/layers = degraded local experience
if [[ "$CTX_SIZE" -le 8192 ]] && [[ "$N_GPU_LAYERS" -lt 99 ]]; then
  API_RECOMMENDED=true
  API_REASON="Low VRAM forces reduced context (${CTX_SIZE} tokens) and partial GPU offload (${N_GPU_LAYERS} layers). Local inference will be slow."
elif [[ "$CTX_SIZE" -le 16384 ]] && [[ "$N_PARALLEL" -le 2 ]]; then
  API_RECOMMENDED=true
  API_REASON="Limited memory restricts context to ${CTX_SIZE} tokens and ${N_PARALLEL} parallel slots. Multi-file tasks may be unreliable."
fi

# No GPU at all — everything runs on CPU
if [[ "$GPU_COUNT" -eq 0 ]] && [[ "$GPU_COMPUTE_CAP" == "none" ]]; then
  API_RECOMMENDED=true
  API_REASON="No GPU detected. CPU-only inference will be very slow (1-5 tok/s)."
fi

if [[ "$API_RECOMMENDED" == true ]]; then
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  API PROVIDER RECOMMENDED                                   ║"
  echo "╠══════════════════════════════════════════════════════════════╣"
  echo "║  $API_REASON"
  echo "║"
  echo "║  Consider using a cloud API instead for better performance:"
  echo "║"
  echo "║  Option 1 — Anthropic Claude (recommended for coding):"
  echo "║    inference_provider: \"anthropic\""
  echo "║    inference_api_key: \"sk-ant-...\""
  echo "║    inference_model: \"claude-sonnet-4-5-20250929\""
  echo "║"
  echo "║  Option 2 — OpenAI:"
  echo "║    inference_provider: \"openai\""
  echo "║    inference_api_key: \"sk-...\""
  echo "║    inference_model: \"gpt-4o\""
  echo "║"
  echo "║  Option 3 — OpenRouter (access any model):"
  echo "║    inference_provider: \"openrouter\""
  echo "║    inference_api_key: \"sk-or-...\""
  echo "║    inference_model: \"qwen/qwen3-coder\""
  echo "║"
  echo "║  Add these to your conductor.yaml or set as env vars:"
  echo "║    export CONDUCTOR_INFERENCE_PROVIDER=anthropic"
  echo "║    export CONDUCTOR_INFERENCE_API_KEY=sk-ant-..."
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Local inference will still work, but expect degraded performance."
  echo "You can always switch later by editing conductor.yaml."
  echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# Check if model file exists for the recommended quant
# ═══════════════════════════════════════════════════════════════════

MODEL_DIR="$SCRIPT_DIR/models/qwen3-coder-next"
MODEL_PATH=""
if [[ -d "$MODEL_DIR" ]]; then
  # Find model matching the glob
  MODEL_PATH=$(compgen -G "$MODEL_DIR/$MODEL_GLOB" 2>/dev/null | head -1 || echo "")
fi

if [[ -n "$MODEL_PATH" ]]; then
  echo "Model found: $MODEL_PATH"
else
  echo "Model NOT found for $QUANT quantization."
  echo "Download with:"
  echo "  huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \\"
  echo "    --include \"$MODEL_GLOB\" \\"
  echo "    --local-dir $MODEL_DIR"
  # Use placeholder path for the profile
  MODEL_PATH="./models/qwen3-coder-next/Qwen3-Coder-Next-${QUANT}.gguf"
fi

# ═══════════════════════════════════════════════════════════════════
# JSON output mode
# ═══════════════════════════════════════════════════════════════════

if [[ "$JSON_OUT" == true ]]; then
  cat <<ENDJSON
{
  "cpu": {
    "model": "$CPU_MODEL",
    "threads": $CPU_CORES
  },
  "ram_mb": $RAM_MB,
  "gpus": {
    "count": $GPU_COUNT,
    "total_vram_mb": $TOTAL_GPU_MB,
    "min_compute_cap": "$GPU_COMPUTE_CAP"
  },
  "recommendation": {
    "quantization": "$QUANT",
    "model_glob": "$MODEL_GLOB",
    "ctx_size": $CTX_SIZE,
    "n_parallel": $N_PARALLEL,
    "n_gpu_layers": $N_GPU_LAYERS,
    "gpu_offload": "$GPU_OFFLOAD",
    "cache_type_k": "$CACHE_K",
    "cache_type_v": "$CACHE_V",
    "batch_size": $BATCH_SIZE,
    "ubatch_size": $UBATCH_SIZE
  },
  "api_recommended": $API_RECOMMENDED,
  "api_reason": "$API_REASON"
}
ENDJSON
  exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# Write profile
# ═══════════════════════════════════════════════════════════════════

API_NOTE=""
if [[ "$API_RECOMMENDED" == true ]]; then
  API_NOTE="
# WARNING: Local inference is degraded on this hardware.
# $API_REASON
# Consider switching to an API provider in conductor.yaml:
#   inference_provider: \"anthropic\"
#   inference_api_key: \"sk-ant-...\"
# Or set env vars:
#   CONDUCTOR_INFERENCE_PROVIDER=anthropic
#   CONDUCTOR_INFERENCE_API_KEY=sk-ant-...
"
fi

PROFILE_CONTENT="# Conductor hardware profile — generated by detect-hardware.sh
# $(date -u '+%Y-%m-%dT%H:%M:%SZ')
#
# Hardware: $CPU_MODEL | ${RAM_GB}GB RAM | ${GPU_COUNT}x GPU (${TOTAL_GPU_MB}MB total VRAM)
# Selected: $QUANT quantization
${API_NOTE}
CONDUCTOR_MODEL_PATH=\"$MODEL_PATH\"
CONDUCTOR_N_GPU_LAYERS=$N_GPU_LAYERS
CONDUCTOR_GPU_OFFLOAD=\"$GPU_OFFLOAD\"
CONDUCTOR_N_PARALLEL=$N_PARALLEL
CONDUCTOR_CTX_SIZE=$CTX_SIZE
CONDUCTOR_BATCH_SIZE=$BATCH_SIZE
CONDUCTOR_UBATCH_SIZE=$UBATCH_SIZE
CONDUCTOR_CACHE_TYPE_K=$CACHE_K
CONDUCTOR_CACHE_TYPE_V=$CACHE_V

# HuggingFace download command for this quantization:
# huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF --include \"$MODEL_GLOB\" --local-dir ./models/qwen3-coder-next
"

if [[ "$DRY_RUN" == true ]]; then
  echo "═══════════════════════════════════════════"
  echo "Would write to: $PROFILE_PATH"
  echo "═══════════════════════════════════════════"
  echo "$PROFILE_CONTENT"
else
  echo "$PROFILE_CONTENT" > "$PROFILE_PATH"
  echo "Profile written to: $PROFILE_PATH"
  echo ""
  echo "Next steps:"
  echo "  1. Review the profile:  cat hardware-profile.env"
  echo "  2. Download the model:  ./setup.sh"
  echo "  3. Run deployment tests: ./test-deployment.sh"
  echo "  4. Launch:              ./launch-conductor.sh"
  if [[ "$API_RECOMMENDED" == true ]]; then
    echo ""
    echo "  Or, to use an API provider instead of local inference:"
    echo "  1. Edit conductor.yaml (set inference_provider + inference_api_key)"
    echo "  2. Run deployment tests: ./test-deployment.sh"
    echo "  3. Launch:              ./launch-conductor.sh"
  fi
fi
