#!/bin/bash
#SBATCH --job-name=qwen3-zl-hard
#SBATCH --account=icdsai_crch_llmspring26
#SBATCH --partition=standard
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/work/sbd5760/diva/logs/qwen3-zl-hard-%j.out
#SBATCH --error=/storage/work/sbd5760/diva/logs/qwen3-zl-hard-%j.err

set -euo pipefail

DIVA=/storage/work/sbd5760/diva
SCRATCH=/scratch/sbd5760
REPO=$DIVA/advisor-models
SIF=$DIVA/vllm-openai.sif
HF_CACHE=$SCRATCH/diva-hf-cache

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_NAME=student
PORT=8000
TP=2
MAX_LEN=32768

mkdir -p $DIVA/logs $HF_CACHE
[ -f "$SIF" ] || { echo "Missing SIF" >&2; exit 1; }

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/$USER-apptainer}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$DIVA/cache/vllm" "$DIVA/cache/torch-inductor"

echo "[$(date)] Launching vLLM (thinking enabled)"
apptainer run --nv \
    --bind "$DIVA":"$DIVA" --bind "$SCRATCH":"$SCRATCH" --bind /etc/pki:/etc/pki:ro \
    --env HF_HOME="$HF_CACHE" --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env VLLM_CACHE_ROOT="$DIVA/cache/vllm" \
    --env TORCHINDUCTOR_CACHE_DIR="$DIVA/cache/torch-inductor" \
    --env VLLM_LOGGING_LEVEL=INFO \
    "$SIF" \
    --model "$MODEL" --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.9 \
    --reasoning-parser qwen3 \
    > $DIVA/logs/vllm-zl-hard-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 240); do
    if curl -fsS http://localhost:$PORT/v1/models >/dev/null 2>&1; then
        echo "[$(date)] vLLM ready after ~${i}*5s"; break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died"; tail -80 $DIVA/logs/vllm-zl-hard-$SLURM_JOB_ID.log; exit 1
    fi
    sleep 5
done

cd $REPO
source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "[$(date)] ==== ZebraLogic 5*6, 6*5, 6*6 (40 puzzles each) ===="
python -m baselines.no_advisor.eval_zebralogic \
    --model "hosted_vllm/$SERVED_NAME" \
    --api_base "http://localhost:$PORT/v1" \
    --sizes "5*6,6*5,6*6" --per_size 40 \
    --output_suffix hard \
    --output_dir baselines/no_advisor/results

echo "[$(date)] Done"
ls -la baselines/no_advisor/results/ | grep zebralogic
