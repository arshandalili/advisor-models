#!/bin/bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-32768}"

# --- serve ---
echo "[$(date)] Starting vLLM..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name student \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.9 \
    --reasoning-parser qwen3 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# wait for readiness
for i in $(seq 1 120); do
    curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; exit 1; }
    sleep 5
done
echo "[$(date)] vLLM ready"

# --- eval ---
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -m baselines.no_advisor.eval_zebralogic \
    --model "hosted_vllm/student" \
    --api_base "http://localhost:$PORT/v1" \
    --sizes "5*6,6*5,6*6" --per_size 40 \
    --output_suffix hard \
    --output_dir baselines/no_advisor/results

echo "[$(date)] Done"
ls -la baselines/no_advisor/results/ | grep zebralogic
