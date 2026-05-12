#!/bin/bash
set -e
export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
export API_BASE="${API_BASE:-https://aja7154-december15-pleasedonotre.services.ai.azure.com/api/projects/aja7154-december15-pleasedonotremovemeplease/openai/v1}"
VENV="/data/arshan/cse587/project/advisor-models/SkyRL/skyrl-train/.venv/bin/python"
LOG="logs/extraction.log"
mkdir -p logs

run_with_vllm() {
    local model_path="$1"
    local output="$2"
    echo "=== Starting vLLM for $model_path ===" | tee -a "$LOG"
    $VENV -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name advisor_model \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.3 \
        --max-model-len 4096 \
        > logs/vllm_extraction.log 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for server
    echo "Waiting for vLLM to be ready..." | tee -a "$LOG"
    until curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do sleep 3; done
    echo "vLLM ready." | tee -a "$LOG"

    $VENV scripts/extract_case_studies.py \
        --advisor_model "$model_path" \
        --output "$output" \
        --max_per_type 40 \
        2>&1 | tee -a "$LOG"

    kill $VLLM_PID 2>/dev/null || true
    until ! ss -tlnp | grep -q :8000; do sleep 2; done
    echo "vLLM stopped." | tee -a "$LOG"
}

run_with_vllm "Qwen/Qwen2.5-1.5B-Instruct" "data/riddlebench/case_studies_untrained.json"
run_with_vllm "ckpts/riddlebench_improved_v2/hf_model" "data/riddlebench/case_studies_improved.json"

echo "=== All extractions done ===" | tee -a "$LOG"
