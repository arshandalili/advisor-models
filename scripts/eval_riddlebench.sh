#!/bin/bash
# Evaluate a trained (or untrained) RiddleBench advisor checkpoint.
#
# Usage:
#   bash scripts/eval_riddlebench.sh \
#       --ckpt $HOME/ckpts/riddlebench_improved \
#       --student gpt-4o-mini \
#       --output results/eval_improved.json
#
#   # Untrained baseline
#   bash scripts/eval_riddlebench.sh \
#       --ckpt Qwen/Qwen2.5-1.5B-Instruct \
#       --student gpt-4o-mini \
#       --output results/eval_untrained.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set; export your Azure OpenAI key}"
export API_BASE="${API_BASE:-https://aja7154-december15-pleasedonotre.services.ai.azure.com/api/projects/aja7154-december15-pleasedonotremovemeplease/openai/v1}"

CKPT=""
STUDENT="gpt-4o-mini"
OUTPUT="results/eval_riddlebench.json"
NUM_RUNS=3
MAX_EXAMPLES=""
MAX_WORKERS=16
TP_SIZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)       CKPT="$2";        shift 2 ;;
    --student)    STUDENT="$2";     shift 2 ;;
    --output)     OUTPUT="$2";      shift 2 ;;
    --num_runs)   NUM_RUNS="$2";    shift 2 ;;
    --max_examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --tp)         TP_SIZE="$2";     shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "Usage: $0 --ckpt <path_or_hf_id> [--student MODEL] [--output FILE]"
  exit 1
fi

DATA_FILE="data/riddlebench/validation_gpt-4o-mini.parquet"
mkdir -p "$(dirname "$OUTPUT")"

echo "[eval] Advisor=$CKPT, Student=$STUDENT, Runs=$NUM_RUNS"

EXTRA_ARGS=""
if [[ -n "$MAX_EXAMPLES" ]]; then
  EXTRA_ARGS="--max_examples $MAX_EXAMPLES"
fi

SkyRL/skyrl-train/.venv/bin/python -m advisor_models.riddlebench.eval_riddlebench \
  --model_name "$CKPT" \
  --dataset_path "$DATA_FILE" \
  --student_model "$STUDENT" \
  --num_runs "$NUM_RUNS" \
  --max_workers "$MAX_WORKERS" \
  --tensor_parallel_size "$TP_SIZE" \
  $EXTRA_ARGS

echo "[eval] Done"
