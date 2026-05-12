#!/bin/bash
# Build RiddleBench train/val parquet files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Azure OpenAI config (OpenAI-compatible endpoint)
export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
export API_BASE="${API_BASE:-https://aja7154-december15-pleasedonotre.services.ai.azure.com/api/projects/aja7154-december15-pleasedonotremovemeplease/openai/v1}"

MODEL="${MODEL:-gpt-4o-mini}"
TRAIN_SIZE="${TRAIN_SIZE:-250}"
VAL_SIZE="${VAL_SIZE:-150}"
K="${K:-2}"
MAX_WORKERS="${MAX_WORKERS:-16}"
OUTPUT_DIR="data/riddlebench"

echo "[construct] Building RiddleBench dataset (train=$TRAIN_SIZE, val=$VAL_SIZE, k=$K, model=$MODEL)"

.venv/bin/python advisor_models/riddlebench/construct_dataset.py \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --train_size "$TRAIN_SIZE" \
    --val_size "$VAL_SIZE" \
    --k "$K" \
    --max_workers "$MAX_WORKERS"

echo "[construct] Done. Files in $OUTPUT_DIR"
