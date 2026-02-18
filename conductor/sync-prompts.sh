#!/usr/bin/env bash
# Conductor — Sync prompts between local files and Langfuse
#
# Usage:
#   ./sync-prompts.sh pull    # Langfuse → local files (update from Langfuse)
#   ./sync-prompts.sh push    # Local files → Langfuse (publish edits)
#   ./sync-prompts.sh list    # Show all local prompt templates
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if needed
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "$SCRIPT_DIR/.venv" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

ACTION="${1:-list}"

case "$ACTION" in
  pull)
    echo "Pulling prompts from Langfuse → local files..."
    python3 -c "
from orchestrator.prompts.prompt_manager import PromptManager
pm = PromptManager()
synced = pm.sync_from_langfuse()
print(f'Synced {len(synced)} prompts:')
for name in synced:
    print(f'  - {name}')
if not synced:
    print('  (none — check Langfuse connection)')
"
    echo ""
    echo "Local prompt files are in: orchestrator/prompts/templates/"
    echo "Commit changes with: git add orchestrator/prompts/templates/ && git commit -m 'sync prompts from Langfuse'"
    ;;

  push)
    echo "Pushing local prompts → Langfuse..."
    python3 -c "
from orchestrator.prompts.prompt_manager import PromptManager
pm = PromptManager()
pushed = pm.sync_to_langfuse()
print(f'Pushed {len(pushed)} prompts:')
for name in pushed:
    print(f'  - {name}')
if not pushed:
    print('  (none — check Langfuse connection and local files)')
"
    ;;

  list)
    echo "Local prompt templates:"
    echo ""
    python3 -c "
from orchestrator.prompts.prompt_manager import PromptManager
pm = PromptManager(langfuse_enabled=False)
for name in pm.list_local_prompts():
    print(f'  {name}')
"
    ;;

  *)
    echo "Usage: ./sync-prompts.sh [pull|push|list]"
    exit 1
    ;;
esac
