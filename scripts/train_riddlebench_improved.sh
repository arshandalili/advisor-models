#!/bin/bash
# Full method v2: specificity reward + full advisor output to student. Resume from step 37, run 4 epochs total.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKYRL_PYTHON="$REPO_ROOT/SkyRL/skyrl-train/.venv/bin/python"

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="$REPO_ROOT/SkyRL/skyrl-train:$REPO_ROOT:${PYTHONPATH:-}"
export ADVISOR_MODELS_MODE=advisor
export STUDENT_MODEL="${STUDENT_MODEL:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set; export your Azure OpenAI key}"
export API_BASE="${API_BASE:-https://aja7154-december15-pleasedonotre.services.ai.azure.com/api/projects/aja7154-december15-pleasedonotremovemeplease/openai/v1}"

DATA_DIR="$REPO_ROOT/data/riddlebench"
MODEL="${ADVISOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NUM_GPUS="${NUM_GPUS:-8}"
LOGGER="${LOGGER:-tensorboard}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/riddlebench_improved_v2}"
RUN_NAME="${RUN_NAME:-riddlebench_improved_v2}"

echo "[train] Full advisor — model=$MODEL, gpus=$NUM_GPUS"

"$SKYRL_PYTHON" -m advisor_models.riddlebench.main_riddlebench \
  data.train_data="['$DATA_DIR/train_gpt-4o-mini.parquet']" \
  data.val_data="['$DATA_DIR/validation_gpt-4o-mini.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=4 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=18 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=18 \
  trainer.max_prompt_length=8192 \
  generator.sampling_params.max_generate_length=16384 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=riddlebench \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="cse587_advisor_models" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH"
