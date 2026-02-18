#!/usr/bin/env bash
# Conductor — Backup configs, data, and state
#
# Creates timestamped backups of:
#   - Project configs (conductor.yaml, constraints.md)
#   - Hardware profile
#   - Training data (JSONL traces)
#   - Exemplar library
#   - Metrics logs
#   - KV cache metadata (not the binary cache files)
#   - Langfuse database (if running via docker compose)
#   - CouchDB (if running via docker compose)
#
# Usage:
#   ./backup.sh                    # Full backup
#   ./backup.sh --configs-only     # Just configs and profiles
#   ./backup.sh --data-only        # Just training data and metrics
#   ./backup.sh --restore <file>   # Restore from backup archive
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BACKUP_DIR="$SCRIPT_DIR/backups"
TIMESTAMP=$(date -u '+%Y%m%dT%H%M%SZ')
BACKUP_NAME="conductor-backup-${TIMESTAMP}"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

CONFIGS_ONLY=false
DATA_ONLY=false
RESTORE_FILE=""

for arg in "$@"; do
  case "$arg" in
    --configs-only)  CONFIGS_ONLY=true ;;
    --data-only)     DATA_ONLY=true ;;
    --restore)       shift; RESTORE_FILE="${2:-}"; shift ;;
    --help|-h)
      echo "Usage: ./backup.sh [--configs-only|--data-only|--restore <file>]"
      exit 0
      ;;
  esac
done

# ── Restore mode ────────────────────────────────────────────────────
if [[ -n "$RESTORE_FILE" ]]; then
  if [[ ! -f "$RESTORE_FILE" ]]; then
    echo "[ERROR] Backup file not found: $RESTORE_FILE"
    exit 1
  fi

  echo "[restore] Extracting: $RESTORE_FILE"
  RESTORE_DIR=$(mktemp -d)
  tar -xzf "$RESTORE_FILE" -C "$RESTORE_DIR"

  # Find the backup root (handle nested directory)
  RESTORE_ROOT=$(find "$RESTORE_DIR" -maxdepth 1 -type d | tail -1)

  # Restore configs
  if [[ -d "$RESTORE_ROOT/configs" ]]; then
    echo "[restore] Restoring project configs..."
    cp -r "$RESTORE_ROOT/configs/"* "$SCRIPT_DIR/projects/" 2>/dev/null || true
  fi

  # Restore hardware profile
  if [[ -f "$RESTORE_ROOT/hardware-profile.env" ]]; then
    echo "[restore] Restoring hardware profile..."
    cp "$RESTORE_ROOT/hardware-profile.env" "$SCRIPT_DIR/"
  fi

  # Restore training data
  if [[ -d "$RESTORE_ROOT/training" ]]; then
    echo "[restore] Restoring training data..."
    mkdir -p "$SCRIPT_DIR/data/training"
    cp -r "$RESTORE_ROOT/training/"* "$SCRIPT_DIR/data/training/" 2>/dev/null || true
  fi

  # Restore exemplars
  if [[ -d "$RESTORE_ROOT/exemplars" ]]; then
    echo "[restore] Restoring exemplar library..."
    mkdir -p "$SCRIPT_DIR/data/exemplars"
    cp -r "$RESTORE_ROOT/exemplars/"* "$SCRIPT_DIR/data/exemplars/" 2>/dev/null || true
  fi

  # Restore metrics
  if [[ -d "$RESTORE_ROOT/metrics" ]]; then
    echo "[restore] Restoring metrics..."
    mkdir -p "$SCRIPT_DIR/data/metrics"
    cp -r "$RESTORE_ROOT/metrics/"* "$SCRIPT_DIR/data/metrics/" 2>/dev/null || true
  fi

  # Restore prompt templates
  if [[ -d "$RESTORE_ROOT/prompts" ]]; then
    echo "[restore] Restoring prompt templates..."
    mkdir -p "$SCRIPT_DIR/orchestrator/prompts/templates"
    cp -r "$RESTORE_ROOT/prompts/"* "$SCRIPT_DIR/orchestrator/prompts/templates/" 2>/dev/null || true
  fi

  rm -rf "$RESTORE_DIR"
  echo "[restore] Done."
  exit 0
fi

# ── Backup mode ─────────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_PATH"

echo "╔══════════════════════════════════════════╗"
echo "║       Conductor — Creating Backup        ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Backup: $BACKUP_NAME"
echo ""

# ── Configs ─────────────────────────────────────────────────────────
if [[ "$DATA_ONLY" != true ]]; then
  echo "[backup] Project configs..."
  mkdir -p "$BACKUP_PATH/configs"
  # Copy all project directories (conductor.yaml + constraints.md)
  for project_dir in "$SCRIPT_DIR/projects"/*/; do
    if [[ -d "$project_dir" ]]; then
      project_name=$(basename "$project_dir")
      mkdir -p "$BACKUP_PATH/configs/$project_name"
      cp "$project_dir"*.yaml "$BACKUP_PATH/configs/$project_name/" 2>/dev/null || true
      cp "$project_dir"*.md "$BACKUP_PATH/configs/$project_name/" 2>/dev/null || true
    fi
  done

  # Hardware profile
  if [[ -f "$SCRIPT_DIR/hardware-profile.env" ]]; then
    echo "[backup] Hardware profile..."
    cp "$SCRIPT_DIR/hardware-profile.env" "$BACKUP_PATH/"
  fi

  # .env (if exists — careful, contains secrets)
  if [[ -f "$SCRIPT_DIR/.env" ]]; then
    echo "[backup] Environment config (.env)..."
    cp "$SCRIPT_DIR/.env" "$BACKUP_PATH/"
  fi

  # Prompt templates
  if [[ -d "$SCRIPT_DIR/orchestrator/prompts/templates" ]]; then
    echo "[backup] Prompt templates..."
    mkdir -p "$BACKUP_PATH/prompts"
    cp "$SCRIPT_DIR/orchestrator/prompts/templates/"* "$BACKUP_PATH/prompts/" 2>/dev/null || true
  fi
fi

# ── Data ────────────────────────────────────────────────────────────
if [[ "$CONFIGS_ONLY" != true ]]; then
  # Training data (JSONL files — can be large)
  if [[ -d "$SCRIPT_DIR/data/training" ]] && [[ -n "$(ls -A "$SCRIPT_DIR/data/training" 2>/dev/null)" ]]; then
    echo "[backup] Training data..."
    mkdir -p "$BACKUP_PATH/training"
    cp -r "$SCRIPT_DIR/data/training/"* "$BACKUP_PATH/training/"
  fi

  # Exemplar library
  if [[ -d "$SCRIPT_DIR/data/exemplars" ]] && [[ -n "$(ls -A "$SCRIPT_DIR/data/exemplars" 2>/dev/null)" ]]; then
    echo "[backup] Exemplar library..."
    mkdir -p "$BACKUP_PATH/exemplars"
    cp -r "$SCRIPT_DIR/data/exemplars/"* "$BACKUP_PATH/exemplars/"
  fi

  # Metrics logs
  if [[ -d "$SCRIPT_DIR/data/metrics" ]] && [[ -n "$(ls -A "$SCRIPT_DIR/data/metrics" 2>/dev/null)" ]]; then
    echo "[backup] Metrics..."
    mkdir -p "$BACKUP_PATH/metrics"
    cp -r "$SCRIPT_DIR/data/metrics/"* "$BACKUP_PATH/metrics/"
  fi

  # KV cache metadata (not binaries — those are huge and regenerable)
  if [[ -d "$SCRIPT_DIR/data/kv-cache" ]]; then
    echo "[backup] KV cache metadata..."
    mkdir -p "$BACKUP_PATH/kv-cache-meta"
    find "$SCRIPT_DIR/data/kv-cache" -name "*.meta.json" -exec cp {} "$BACKUP_PATH/kv-cache-meta/" \; 2>/dev/null || true
  fi

  # Langfuse PostgreSQL dump (if running in docker)
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'conductor-langfuse-db'; then
    echo "[backup] Langfuse database (PostgreSQL dump)..."
    docker exec conductor-langfuse-db pg_dump -U langfuse langfuse \
      > "$BACKUP_PATH/langfuse-db.sql" 2>/dev/null || echo "  (Langfuse DB dump failed — skipping)"
  fi

  # CouchDB backup (if running in docker)
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q 'conductor-couchdb'; then
    echo "[backup] CouchDB databases..."
    mkdir -p "$BACKUP_PATH/couchdb"
    # List all databases and dump each
    COUCH_USER="${COUCHDB_USER:-admin}"
    COUCH_PASS="${COUCHDB_PASSWORD:-conductor}"
    COUCH_URL="http://${COUCH_USER}:${COUCH_PASS}@localhost:5984"

    DBS=$(curl -sf "$COUCH_URL/_all_dbs" 2>/dev/null || echo "[]")
    if [[ "$DBS" != "[]" ]]; then
      for db in $(echo "$DBS" | python3 -c "import sys,json; [print(d) for d in json.load(sys.stdin) if not d.startswith('_')]" 2>/dev/null); do
        echo "  Dumping: $db"
        curl -sf "$COUCH_URL/$db/_all_docs?include_docs=true" \
          > "$BACKUP_PATH/couchdb/${db}.json" 2>/dev/null || true
      done
    fi
  fi
fi

# ── Compress ────────────────────────────────────────────────────────
echo ""
echo "[backup] Compressing..."
ARCHIVE="${BACKUP_PATH}.tar.gz"
tar -czf "$ARCHIVE" -C "$BACKUP_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_PATH"

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           Backup Complete                ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  File: $ARCHIVE"
echo "  Size: $SIZE"
echo ""
echo "  Restore with: ./backup.sh --restore $ARCHIVE"
echo ""

# Prune old backups (keep last 10)
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/conductor-backup-*.tar.gz 2>/dev/null | wc -l)
if [[ "$BACKUP_COUNT" -gt 10 ]]; then
  echo "[cleanup] Pruning old backups (keeping last 10)..."
  ls -1t "$BACKUP_DIR"/conductor-backup-*.tar.gz | tail -n +11 | xargs rm -f
fi
