#!/bin/bash
# Master orchestrator: runs everything within a 12h budget.
#
# Timeline (conservative):
#   T+0h:    Dataset construction + student-only baseline eval (parallel)
#   T+1.5h:  Train run 1 (standard advisor)
#   T+3h:    Train run 2 (diag + k=2)
#   T+4.5h:  Train run 3 (full method)
#   T+6h:    All evals complete (run in background after each training)
#
# Set OPENAI_API_KEY before running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set; export your Azure OpenAI key}"
export API_BASE="${API_BASE:-https://aja7154-december15-pleasedonotre.services.ai.azure.com/api/projects/aja7154-december15-pleasedonotremovemeplease/openai/v1}"

mkdir -p logs results

T0=$(date +%s)
elapsed() { echo "T+$(( ($(date +%s) - T0) / 60 ))min"; }

echo "$(elapsed) === Starting full pipeline ==="

# ── Dataset construction (background) ──────────────────────────────────
echo "$(elapsed) Building dataset..."
bash scripts/construct_riddlebench_dataset.sh > logs/dataset.log 2>&1 &
DATASET_PID=$!

# ── Student-only baseline eval (no GPU needed, parallel) ───────────────
echo "$(elapsed) Running student-only baseline eval..."
.venv/bin/python -m baselines.no_advisor.eval_riddlebench \
    --model gpt-4o-mini \
    --data_file data/riddlebench/validation_gpt-4o-mini.parquet \
    --output_dir results \
    --num_runs 3 \
    --max_workers 20 > logs/eval_student_only.log 2>&1 &
BASELINE_PID=$!

# Wait for dataset before training
wait $DATASET_PID
echo "$(elapsed) Dataset ready."

# ── Train run 1: standard advisor ─────────────────────────────────────
echo "$(elapsed) Starting training: standard advisor..."
bash scripts/train_riddlebench_standard.sh > logs/train_standard.log 2>&1
echo "$(elapsed) Standard training done."

# Eval run 1 in background
bash scripts/eval_riddlebench.sh \
    --ckpt "$HOME/ckpts/riddlebench_standard" \
    --student gpt-4o-mini \
    --output results/eval_standard.json > logs/eval_standard.log 2>&1 &

# ── Train run 2: diag + k=2 ───────────────────────────────────────────
echo "$(elapsed) Starting training: diag advisor..."
bash scripts/train_riddlebench_diag.sh > logs/train_diag.log 2>&1
echo "$(elapsed) Diag training done."

bash scripts/eval_riddlebench.sh \
    --ckpt "$HOME/ckpts/riddlebench_diag" \
    --student gpt-4o-mini \
    --output results/eval_diag.json > logs/eval_diag.log 2>&1 &

# ── Train run 3: full method ───────────────────────────────────────────
echo "$(elapsed) Starting training: full advisor..."
bash scripts/train_riddlebench_improved.sh > logs/train_improved.log 2>&1
echo "$(elapsed) Full training done."

bash scripts/eval_riddlebench.sh \
    --ckpt "$HOME/ckpts/riddlebench_improved" \
    --student gpt-4o-mini \
    --output results/eval_improved.json > logs/eval_improved.log 2>&1 &

# ── Transferability eval (stronger student) ────────────────────────────
bash scripts/eval_riddlebench.sh \
    --ckpt "$HOME/ckpts/riddlebench_improved" \
    --student gpt-4.1-mini \
    --output results/eval_transfer.json > logs/eval_transfer.log 2>&1 &

# ── Untrained advisor baseline ─────────────────────────────────────────
bash scripts/eval_riddlebench.sh \
    --ckpt "Qwen/Qwen2.5-1.5B-Instruct" \
    --student gpt-4o-mini \
    --output results/eval_untrained.json > logs/eval_untrained.log 2>&1 &

# Wait for all background jobs
wait
echo "$(elapsed) === All done. Results in results/ ==="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in results/eval_*.json; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    python -c "
import json, sys
d = json.load(open('$f'))
agg = d.get('aggregate_stats', {})
mean = agg.get('mean', agg.get('accuracy', '?'))
print(f'  Model: {d.get(\"model\", \"?\")}')
print(f'  Student: {d.get(\"student\", d.get(\"model\", \"?\"))}')
print(f'  Accuracy: {mean}')
" 2>/dev/null || echo "  (parse error)"
done
