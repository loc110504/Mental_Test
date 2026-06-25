#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO_ROOT/src"
DATA_DIR="${1:-$REPO_ROOT/data/processed_dev_daic_woz}"
CSV_FILE="${2:-$REPO_ROOT/evaluation/agentmental_dev_full.csv}"

export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:30b}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither python nor python3 is available on PATH." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 1
fi

cd "$SRC_DIR"

shopt -s nullglob
files=("$DATA_DIR"/*.json)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No JSON files found in: $DATA_DIR" >&2
  exit 1
fi

echo "Running original AgentMental pipeline on ${#files[@]} dev samples"
echo "Data dir: $DATA_DIR"
echo "Result CSV: $CSV_FILE"
echo "LLM provider: $LLM_PROVIDER"
echo "Ollama model: $OLLAMA_MODEL"

for file_path in "${files[@]}"; do
  echo
  echo "=== Processing $(basename "$file_path") ==="
  "$PYTHON_BIN" main.py \
    --sample-path "$file_path" \
    --csv-file "$CSV_FILE" \
    --skip-if-evaluated
done

echo
echo "Completed full dev run."
