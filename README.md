# How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models

> **This repository extends the Advisor Models framework** from the paper:
>
> *How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models*
> Parth Asawa\*, Alan Zhu\*, Abby O'Neill, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez (\*equal contribution)
> 📜 [arXiv:2510.02453](https://arxiv.org/pdf/2510.02453)
>
> All credit for the original framework, training infrastructure (SkyRL), and baseline environments goes to the original authors. This repo adds a new **RiddleBench** environment as a course project extension (CSE 587).

---

## Setup

Run `uv sync` to install local development dependencies, then activate the environment:

```bash
source .venv/bin/activate
```

To set up the separate training virtual environment (required for all training scripts):

```bash
cd SkyRL/skyrl-train
uv sync --extra vllm
source .venv/bin/activate
```

You will also need to export the following environment variables before running any training or evaluation:

```bash
export OPENAI_API_KEY=<your key>
export WANDB_API_KEY=<your key>      # only needed if using wandb logger
```

---

## Advisor Models Overview

![image](assets/advisor_models.png)

Customizing powerful, black-box models is a major challenge, with most practitioners typically limited to static prompting. The Advisor Models framework trains a small open-source "advisor" model with RL to guide a black-box model via feedback, optimizing for a specific task or environment.

![image](assets/example.png)

---

## RiddleBench Extension

This repo adds full support for training and evaluating advisor models on [RiddleBench](https://huggingface.co/datasets/RiddleBench/RiddleBench), a benchmark of logic, spatial, and constraint-based puzzles.

### Dataset

Pre-built train/validation parquet files (generated from `gpt-4o-mini` initial responses) are in `data/riddlebench/`. To regenerate from scratch:

```bash
bash scripts/construct_riddlebench_dataset.sh
```

This calls `advisor_models/riddlebench/construct_dataset.py`, which pulls the RiddleBench dataset, collects initial student responses, and writes the parquet files.

### Training

Three training variants are provided, each corresponding to a different ablation:

| Script | Environment class | Description |
|---|---|---|
| `scripts/train_riddlebench_standard.sh` | `riddlebench_standard` | Standard GRPO advisor, outcome reward only |
| `scripts/train_riddlebench_diag.sh` | `riddlebench_diag` | Adds a diagnosis format reward |
| `scripts/train_riddlebench_improved.sh` | `riddlebench_improved` | Diagnosis format + process reward |

All three train a `Qwen2.5-1.5B-Instruct` advisor model by default. Override with:

```bash
ADVISOR_MODEL=Qwen/Qwen2.5-3B-Instruct NUM_GPUS=4 bash scripts/train_riddlebench_standard.sh
```

Key config is in `advisor_models/riddlebench/config.py` (prompts, error categories, reward weights) and `advisor_models/riddlebench/env.py` (environment logic).

### Evaluation

To evaluate a trained checkpoint against the no-advisor baseline:

```bash
bash scripts/eval_riddlebench.sh
```

For the no-advisor baseline only:

```bash
python baselines/no_advisor/eval_riddlebench.py
```

Results are written to `outputs/` and `results/`.

---

## 📜 License

**Advisor Models** is Apache 2.0 licensed, making it suitable for both academic and commercial use.

## 📋 Citation

```text
@article{asawa2026trainadvisorsteeringblackbox,
  title={How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models},
  author={Parth Asawa and Alan Zhu and Abby O'Neill and Matei Zaharia and Alexandros G. Dimakis and Joseph E. Gonzalez},
  year={2026},
  journal={arXiv preprint arXiv:2510.02453},
}
```
