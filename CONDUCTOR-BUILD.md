# Conductor Phase 0: Claude Code Build Instructions

## Goal

Build a local-first autonomous coding orchestration stack ("Conductor") in Phase 0 with these components:

1. Inference Engine (`ik_llama.cpp` + local Qwen3-Coder-Next GGUF)
2. Inference Gateway (Python FastAPI proxy for slots, KV cache, Ultra Think)
3. Conductor Orchestrator (planner/coder/reviewer loop + Obsidian watcher)

All components communicate over HTTP. Build in that order.

---

## Environment

- CPU: i9-13900K
- RAM: 128GB
- GPU: 2x NVIDIA Tesla P40 (24GB each)
- Python: 3.11+

> Notes:
> - P40 is `sm_61` and lacks tensor cores.
> - Some fused MoE CUDA paths may not be available; CPU fallback for expert-heavy paths is acceptable in Phase 0.

---

## Component 1: Inference Engine (`ik_llama.cpp`)

### 1.1 Clone and Build

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="61" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) \
  --target llama-server llama-cli llama-bench llama-gguf-split
```

### 1.2 Download Model

```bash
python3 -m pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  unsloth/Qwen3-Coder-Next-GGUF \
  --include "*UD-Q4_K_XL*" \
  --local-dir ./models/qwen3-coder-next
```

### 1.3 Launch Script

Create `start-inference.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="./models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

./build/bin/llama-server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  \
  --n-gpu-layers 99 \
  -ot ".ffn_.*_exps.=CPU" \
  \
  -np 5 \
  --slot-save-path ./kv-cache \
  \
  --ctx-size 32768 \
  -b 4096 \
  -ub 4096 \
  \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  \
  --cache-reuse 256 \
  -sps 0.3 \
  \
  --jinja \
  \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 40 \
  --min-p 0.01
```

### 1.4 Validate Engine

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
curl -s http://localhost:8080/health
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"Write a Python hello world"}],"max_tokens":100}' \
  | python3 -m json.tool
```

Record baseline: prompt processing tok/s, generation tok/s, first-token latency.

---

## Component 2: Inference Gateway (Python Proxy)

### 2.1 Structure

```
conductor/
├── gateway/
│   ├── __init__.py
│   ├── server.py
│   ├── slot_manager.py
│   ├── ultra_think.py
│   ├── prefix_cache.py
│   └── config.py
├── tests/
│   ├── test_slot_manager.py
│   ├── test_ultra_think.py
│   └── test_prefix_cache.py
├── pyproject.toml
└── README.md
```

### 2.2 API Surface

- `POST /v1/chat/completions`
- `POST /v1/ultra-think`
- `POST /v1/project/load`
- `POST /v1/project/save`
- `POST /v1/project/restore`
- `GET /v1/slots/status`
- `GET /v1/metrics`

### 2.3 Slot Manager Contract

- Slot `0`: template only (no generation)
- Slots `1-4`: worker generation slots
- Save/restore via llama-server `/slots` API
- Always restore template cache into worker before generation

Track per request: `slot_restore_time_ms`, `prefix_tokens_cached`, `suffix_tokens_processed`, `generation_time_ms`, `tokens_per_second`.

### 2.4 Ultra Think Contract

Tier defaults: Tier 1 N=1, Tier 2 N=3, Tier 3 N=5, Tier 4 decompose/escalate.

Diversity profile:
- candidate1: `temp=0.7, top_p=0.9, top_k=30`
- candidate2: `temp=1.0, top_p=0.95, top_k=40`
- candidate3: `temp=1.2, top_p=0.98, top_k=50`

Use `asyncio.gather()` and pin requests with `id_slot`.

### 2.5 Prefix Cache Manager

```
kv-cache/projects/{project_id}/
├── template.bin
├── template.meta.json
└── history/
```

Invalidation: hash of Layer0 + knowledge context. Match = restore. Mismatch = recompute.

### 2.6 Gateway Config

```python
from pydantic_settings import BaseSettings

class GatewayConfig(BaseSettings):
    llama_server_url: str = "http://localhost:8080"
    template_slot_id: int = 0
    worker_slot_ids: list[int] = [1, 2, 3, 4]
    kv_cache_dir: str = "./kv-cache"
    tier2_candidates: int = 3
    tier3_candidates: int = 5
    default_max_tokens: int = 4096
    generation_timeout_seconds: int = 300
    slot_restore_timeout_seconds: int = 30
    metrics_log_path: str = "./metrics/gateway.jsonl"
```

### 2.7 Required Tests

1. Mock llama-server endpoints
2. Validate slot lifecycle and slot 0 protection
3. Validate Tier 2 concurrent dispatch
4. Validate cache reuse and invalidation by content hash

---

## Component 3: Conductor Orchestrator

### 3.1 Structure

```
conductor/
├── orchestrator/
│   ├── conductor.py
│   ├── planner.py
│   ├── coder.py
│   ├── reviewer.py
│   ├── memory/
│   │   ├── layer0.py
│   │   ├── layer1.py
│   │   ├── layer2.py
│   │   ├── changelog.py
│   │   └── knowledge_graph.py
│   ├── interfaces/
│   │   ├── obsidian_watcher.py
│   │   └── openwebui.py
│   ├── tools/
│   │   ├── file_ops.py
│   │   ├── shell.py
│   │   ├── git.py
│   │   └── test_runner.py
│   ├── training/
│   │   ├── data_collector.py
│   │   └── exemplar_library.py
│   └── config.py
└── projects/example/
    ├── conductor.yaml
    └── constraints.md
```

### 3.2 Core Loop

1. Receive task from Obsidian inbox
2. Load project context via gateway
3. Planner decomposes task
4. For each subtask: estimate tier → Ultra Think → review → apply → test → retry/escalate
5. Write changelog entry
6. Record training data
7. Write result to Obsidian completed file

### 3.3 Memory Stack

- Layer 0: pinned constraints from markdown (always included)
- Layer 1: working memory for active task
- Layer 2: stub (compressed history placeholder)
- Layer 3: JSONL changelog (append-only)
- Layer 4: knowledge graph stub

### 3.4 Obsidian Interface

- `{vault}/conductor/inbox/` — new tasks
- `{vault}/conductor/completed/` — results
- `{vault}/conductor/failed/` — errors
- constraints.md change → invalidate prefix cache

### 3.5 Training Data

Per Ultra Think cycle: prompt, all candidates, reviewer scores, test outcomes, accepted index, human status. Append-only JSONL.

---

## Component 4: Integration and Launch

### 4.1 Config (`projects/example/conductor.yaml`)

```yaml
project_id: "example"
project_dir: "/path/to/repo"
obsidian_vault: "/path/to/obsidian-vault"
gateway_url: "http://localhost:9090"
inference_url: "http://localhost:8080"
max_retries: 3
accept_threshold: 7.0
max_working_memory_tokens: 8000
layer0_path: "./constraints.md"
training_data_dir: "./training-data"
exemplar_library_dir: "./exemplars"
```

### 4.2 Launch Script (`launch-conductor.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
./start-inference.sh &
sleep 10
uvicorn gateway.server:app --host 0.0.0.0 --port 9090 &
sleep 2
python -m orchestrator.conductor --project example --config projects/example/conductor.yaml
```

---

## Phase 0 Success Criteria

1. Inference engine loads model and responds consistently
2. Prefix caching shows measurable speedup
3. Tier 2 Ultra Think yields diverse candidates
4. Reviewer scores correlate with manual quality checks
5. Obsidian drop-in → completed output loop works end-to-end
6. Training JSONL accumulates valid traces
7. Metrics emitted: tok/s, cache hit rate, latency, acceptance, tier usage

Target: first-attempt acceptance rate >= 30%.

## Phase 0 Exclusions

- Full Layer 4 knowledge graph
- OpenWebUI adapter (stub only)
- Scout daemon
- Fine-tuning pipeline execution (collect data only)
- Multi-project scheduling
- Learned difficulty model (heuristics only)
- Candidate synthesis engine (pick best by reviewer)

## Python Dependencies

```toml
[project]
name = "conductor"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.34",
  "httpx>=0.28",
  "pydantic>=2.10",
  "pydantic-settings>=2.7",
  "watchdog>=6.0",
  "pyyaml>=6.0",
  "rich>=13.9",
]
```

## Architecture Invariants

1. Conductor does not directly execute code/commands.
2. All model inference calls pass through Gateway.
3. Memory layers are explicit and assembled per request.
4. Slot 0 is template-only.
5. All runs emit structured logs/events.
6. Inference backend is swappable behind OpenAI-compatible Gateway API.
7. Training data is generated from normal operation, not a separate workflow.
