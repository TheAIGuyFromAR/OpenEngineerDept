#!/usr/bin/env bash
# Conductor Phase 0 — Bootstrap Script
# Run once after cloning to set up the full local stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
ask()   { echo -en "${YELLOW}[?]${NC}     $* [y/N] "; read -r ans; [[ "$ans" =~ ^[Yy] ]]; }

# ── Parse flags ─────────────────────────────────────────────────────
SKIP_ENGINE=false
SKIP_MODEL=false
SKIP_COUCHDB=false
WITH_COUCHDB=false

for arg in "$@"; do
  case "$arg" in
    --skip-engine)  SKIP_ENGINE=true  ;;
    --skip-model)   SKIP_MODEL=true   ;;
    --skip-couchdb) SKIP_COUCHDB=true ;;
    --with-couchdb) WITH_COUCHDB=true ;;
    --help|-h)
      echo "Usage: ./setup.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-engine   Skip building ik_llama.cpp (if you already have it)"
      echo "  --skip-model    Skip downloading the GGUF model"
      echo "  --skip-couchdb  Skip CouchDB setup even if docker-compose exists"
      echo "  --with-couchdb  Start CouchDB via docker compose"
      echo "  --help          Show this message"
      exit 0
      ;;
    *) warn "Unknown flag: $arg" ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    Conductor Phase 0 — Bootstrap Setup   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Step 1: System prerequisites ────────────────────────────────────
info "Checking system prerequisites..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python 3.11+."

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
  fail "Python 3.11+ required, found $PY_VERSION"
fi
ok "Python $PY_VERSION"

command -v git >/dev/null 2>&1 || fail "git not found."
ok "git"

# ── Step 2: Python environment ──────────────────────────────────────
info "Setting up Python environment..."

if command -v uv >/dev/null 2>&1; then
  ok "uv found — using uv for package management"
  PIP_CMD="uv pip"
  if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    info "Creating virtual environment with uv..."
    uv venv "$SCRIPT_DIR/.venv"
  fi
elif command -v pip >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1; then
  PIP_CMD="${PIP3_CMD:-pip3}"
  command -v pip3 >/dev/null 2>&1 && PIP_CMD="pip3"
  command -v pip >/dev/null 2>&1 && PIP_CMD="pip"
  ok "pip found — using $PIP_CMD"
  if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    info "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
  fi
else
  fail "Neither uv nor pip found. Install one:\n  curl -LsSf https://astral.sh/uv/install.sh | sh\n  OR: python3 -m ensurepip"
fi

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"
ok "Virtual environment activated: $SCRIPT_DIR/.venv"

info "Installing conductor package + dependencies..."
if [[ "$PIP_CMD" == "uv pip" ]]; then
  uv pip install -e "$SCRIPT_DIR[dev]"
else
  $PIP_CMD install -e "$SCRIPT_DIR[dev]"
fi
ok "Python dependencies installed"

# ── Step 3: Build ik_llama.cpp ──────────────────────────────────────
if [[ "$SKIP_ENGINE" == true ]]; then
  warn "Skipping inference engine build (--skip-engine)"
elif [[ -x "$SCRIPT_DIR/build/bin/llama-server" ]]; then
  ok "llama-server binary already exists at build/bin/llama-server"
else
  info "Building ik_llama.cpp inference engine..."

  # Check for CUDA
  if ! command -v nvcc >/dev/null 2>&1; then
    warn "nvcc not found — CUDA build may fail. Make sure CUDA toolkit is in PATH."
  fi

  if ! command -v cmake >/dev/null 2>&1; then
    fail "cmake not found. Install cmake 3.20+."
  fi

  IK_LLAMA_DIR="$SCRIPT_DIR/ik_llama.cpp"

  if [[ ! -d "$IK_LLAMA_DIR" ]]; then
    info "Cloning ik_llama.cpp..."
    git clone https://github.com/ikawrakow/ik_llama.cpp.git "$IK_LLAMA_DIR"
  else
    ok "ik_llama.cpp source already present"
  fi

  info "Configuring cmake (sm_61 for Tesla P40)..."
  cmake -S "$IK_LLAMA_DIR" -B "$IK_LLAMA_DIR/build" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61" \
    -DCMAKE_BUILD_TYPE=Release

  info "Building (this may take a few minutes)..."
  cmake --build "$IK_LLAMA_DIR/build" --config Release -j"$(nproc)" \
    --target llama-server llama-cli llama-bench llama-gguf-split

  # Symlink into conductor/build/bin/
  mkdir -p "$SCRIPT_DIR/build/bin"
  for bin in llama-server llama-cli llama-bench llama-gguf-split; do
    src="$IK_LLAMA_DIR/build/bin/$bin"
    if [[ -f "$src" ]]; then
      ln -sf "$src" "$SCRIPT_DIR/build/bin/$bin"
    fi
  done

  ok "ik_llama.cpp built and linked into build/bin/"
fi

# ── Step 4: Detect hardware & select model ──────────────────────────
info "Detecting hardware and selecting optimal model..."
bash "$SCRIPT_DIR/detect-hardware.sh"
ok "Hardware profile written to hardware-profile.env"

# Source the profile to get CONDUCTOR_MODEL_PATH and the download glob
PROFILE="$SCRIPT_DIR/hardware-profile.env"
if [[ -f "$PROFILE" ]]; then
  source "$PROFILE"
fi

# ── Step 5: Download model ──────────────────────────────────────────
MODEL_DIR="$SCRIPT_DIR/models/qwen3-coder-next"
# Use the path from hardware profile, fall back to default Q4_K_XL
MODEL_PATH="${CONDUCTOR_MODEL_PATH:-./models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"

if [[ "$SKIP_MODEL" == true ]]; then
  warn "Skipping model download (--skip-model)"
elif [[ -f "$MODEL_PATH" ]]; then
  ok "Model GGUF already present: $MODEL_PATH"
else
  info "Downloading Qwen3-Coder-Next GGUF (this is a large download)..."

  # Ensure huggingface_hub is available
  if [[ "$PIP_CMD" == "uv pip" ]]; then
    uv pip install huggingface_hub hf_transfer
  else
    $PIP_CMD install huggingface_hub hf_transfer
  fi

  mkdir -p "$MODEL_DIR"

  # Extract the glob pattern from hardware profile comment
  DOWNLOAD_GLOB=$(grep 'huggingface-cli download' "$PROFILE" 2>/dev/null | grep -oP '(?<=--include ")[^"]+' || echo "*UD-Q4_K_XL*")

  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    unsloth/Qwen3-Coder-Next-GGUF \
    --include "$DOWNLOAD_GLOB" \
    --local-dir "$MODEL_DIR"

  ok "Model downloaded to $MODEL_DIR"
fi

# ── Step 6: Create data directories ────────────────────────────────
info "Ensuring data directories exist..."
mkdir -p "$SCRIPT_DIR/data/kv-cache"
mkdir -p "$SCRIPT_DIR/data/metrics"
mkdir -p "$SCRIPT_DIR/data/training"
mkdir -p "$SCRIPT_DIR/data/exemplars"
ok "Data directories ready"

# ── Step 7: CouchDB (optional) ─────────────────────────────────────
if [[ "$WITH_COUCHDB" == true ]] && [[ "$SKIP_COUCHDB" != true ]]; then
  if command -v docker >/dev/null 2>&1; then
    info "Starting CouchDB via docker compose..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d couchdb
    ok "CouchDB starting on port 5984"

    # Wait for CouchDB to be ready
    info "Waiting for CouchDB to accept connections..."
    for i in $(seq 1 30); do
      if curl -sf http://localhost:5984/ >/dev/null 2>&1; then
        ok "CouchDB is ready"
        break
      fi
      sleep 1
      if [[ $i -eq 30 ]]; then
        warn "CouchDB didn't respond within 30s — check 'docker compose logs couchdb'"
      fi
    done

    # Create the obsidian database if it doesn't exist
    info "Ensuring 'obsidian' database exists..."
    COUCH_USER="${COUCHDB_USER:-admin}"
    COUCH_PASS="${COUCHDB_PASSWORD:-conductor}"
    curl -sf -X PUT "http://${COUCH_USER}:${COUCH_PASS}@localhost:5984/obsidian" >/dev/null 2>&1 \
      && ok "Created 'obsidian' database" \
      || ok "'obsidian' database already exists"
  else
    warn "Docker not found — skipping CouchDB setup. Install Docker or run CouchDB manually."
  fi
elif [[ "$WITH_COUCHDB" != true ]]; then
  info "CouchDB setup skipped (use --with-couchdb to enable)"
fi

# ── Step 8: Validate project config ────────────────────────────────
EXAMPLE_CONFIG="$SCRIPT_DIR/projects/example/conductor.yaml"
if [[ -f "$EXAMPLE_CONFIG" ]]; then
  if grep -q '/path/to/repo' "$EXAMPLE_CONFIG"; then
    warn "projects/example/conductor.yaml still has placeholder paths!"
    warn "Edit it before launching:"
    warn "  project_dir: <your actual repo path>"
    warn "  obsidian_vault: <your Obsidian vault path>"
  else
    ok "Project config looks configured"
  fi
fi

# ── Step 9: Summary ────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           Setup Complete                 ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check what's ready vs what still needs attention
READY=true

if [[ -x "$SCRIPT_DIR/build/bin/llama-server" ]] || [[ -L "$SCRIPT_DIR/build/bin/llama-server" ]]; then
  ok "Inference engine:  build/bin/llama-server"
else
  warn "Inference engine:  MISSING — run setup.sh without --skip-engine"
  READY=false
fi

if [[ -f "$MODEL_PATH" ]]; then
  ok "Model:             $MODEL_PATH"
else
  warn "Model:             MISSING — run setup.sh without --skip-model"
  READY=false
fi

if [[ -f "$PROFILE" ]]; then
  ok "Hardware profile:  hardware-profile.env"
fi

ok "Python venv:       .venv/ (activate with: source .venv/bin/activate)"
ok "Data directories:  data/{kv-cache,metrics,training,exemplars}/"

if grep -q '/path/to/repo' "$EXAMPLE_CONFIG" 2>/dev/null; then
  warn "Config:            projects/example/conductor.yaml needs editing"
  READY=false
else
  ok "Config:            projects/example/conductor.yaml"
fi

echo ""
if [[ "$READY" == true ]]; then
  ok "Ready to launch! Run:"
  echo "    source .venv/bin/activate"
  echo "    ./launch-conductor.sh"
else
  warn "Some components still need setup — see warnings above."
fi
echo ""
