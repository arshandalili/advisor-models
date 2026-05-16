"""GEPA baseline for **ZebraLogic** (grid puzzles).

Prompt formatting, JSON extraction, and grid scoring are imported from
`baselines/no_advisor/eval_zebralogic.py` so this baseline stays aligned with the hosted
no-advisor eval.

Training / data layout (things you need to adopt this baseline properly):

1. **Parquet rows** — each row should include at least:
   - ``id``: stable identifier (string/int)
   - ``size``: e.g. ``"5*6"`` (must match `--sizes` filtering)
   - ``puzzle``: natural-language clues + setup (same as eval)
   - ``solution``: dict with ``{"header": [...], "rows": [...]}`` where ``header[0] == "House"``
     and each row aligns with ``eval_zebralogic.score``.

2. **Train vs eval (important)** — For real experiments, point ``--optimization-train-parquet``
   at a train split and ``--final-eval-parquet`` at a held-out split (the default eval path matches
   ``eval_zebralogic.py``). For local smoke tests only, ``--reuse-data-for-gepa-split`` shuffles one
   parquet and uses **disjoint** contiguous slices for (train+val) vs final eval so rows are not
   double-counted in the same phase.

3. **LLMs** — Set ``OPENAI_API_KEY``. For LiteLLM / local OpenAI-compatible servers (vLLM, etc.)
   export ``OPENAI_BASE_URL`` and point ``--student-lm`` / ``--reflection-lm`` at the served model id.

4. **Reward signal** — Default metric is **per-cell accuracy** normalized to `[0, 1]` (fraction of grid
   cells matching gold). Alternate: `--reward puzzle` uses `1.0` if the full puzzle matches else `0.0`.

5. **W&B** — Same pattern as other GEPA baselines; requires ``WANDB_API_KEY`` when W&B is enabled.

Example (smoke: tiny split, reuse one file):

    python -m baselines.gepa.gepa_zebralogic \\
        --reuse-data-for-gepa-split \\
        --train-size 8 --val-size 4 --final-eval-size 8 \\
        --sizes "5*6" --per-size 20 \\
        --max-calls 128 --minibatch-size 3 --num-threads 4 \\
        --num-runs 1 --no-wandb \\
        --log-dir baselines/gepa/logs/zebralogic \\
        --output-dir baselines/gepa/results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import dspy
import pandas as pd
from dspy import GEPA
from dspy.evaluate import Evaluate

from baselines.no_advisor.eval_zebralogic import (
    build_prompt,
    extract_last_complete_json,
    score,
)
from utils.eval_utils import compute_multi_run_statistics

random.seed(42)

RewardMode = Literal["cells", "puzzle"]


def dataframe_to_records(
    df: pd.DataFrame,
    *,
    sizes: Sequence[str],
    per_size_limit: int | None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for s in sizes:
        sub = df[df["size"] == s]
        if per_size_limit is not None:
            sub = sub.head(per_size_limit)
        items.extend(sub.to_dict("records"))
    if max_records is not None:
        items = items[:max_records]
    return items


def records_to_examples(records: Iterable[dict[str, Any]]) -> list[dspy.Example]:
    out: list[dspy.Example] = []
    for it in records:
        out.append(
            dspy.Example(
                puzzle_prompt=build_prompt(it),
                gold_solution=it["solution"],
                puzzle_id=str(it["id"]),
                size=str(it["size"]),
            ).with_inputs("puzzle_prompt")
        )
    return out


class ZebraGridSolve(dspy.Signature):
    """Solve a ZebraLogic grid puzzle. Think step by step, then output the complete JSON solution object matching the template exactly."""

    puzzle_prompt = dspy.InputField(
        desc="Full puzzle specification including JSON output template."
    )
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning followed by the complete JSON solution object matching the template."
    )


class ZebraModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ZebraGridSolve)

    def forward(self, puzzle_prompt, gold_solution=None):
        return self.generate(puzzle_prompt=puzzle_prompt)


def _reward_from_score(
    mode: RewardMode,
    *,
    cells_correct: int,
    cells_total: int,
    solved: bool,
) -> float:
    if cells_total <= 0:
        return 0.0
    if mode == "puzzle":
        return 1.0 if solved else 0.0
    return float(cells_correct) / float(cells_total)


def _parse_pred(pred) -> dict:
    """Extract the JSON puzzle dict from a prediction regardless of field layout."""
    for attr in ("solution", "answer", "reasoning_and_answer", "reasoning"):
        raw = getattr(pred, attr, None)
        if isinstance(raw, dict):
            return {"solution": raw}
        if isinstance(raw, str) and raw.strip():
            parsed = extract_last_complete_json(raw)
            if parsed:
                return parsed if "solution" in parsed else {"solution": parsed}
    return {}


def compute_score_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    *,
    reward_mode: RewardMode,
):
    parsed = _parse_pred(pred)
    cc, ct, solved = score(parsed, example.gold_solution)
    return _reward_from_score(reward_mode, cells_correct=cc, cells_total=ct, solved=solved)


def feedback_metric(example, pred, trace=None, *args, reward_mode="cells", **kwargs):
    rm: RewardMode = reward_mode if reward_mode in ("cells", "puzzle") else "cells"
    parsed = _parse_pred(pred)
    cc, ct, solved = score(parsed, example.gold_solution)
    r = _reward_from_score(rm, cells_correct=cc, cells_total=ct, solved=solved)
    if rm == "puzzle":
        detail = "puzzle_exact_match=1" if solved else "puzzle_exact_match=0"
    else:
        detail = f"{cc}/{ct} grid cells matched"
    feedback = (
        f"Puzzle id {example.puzzle_id} ({example.size}): reward={r:.4f} ({detail}). "
        f"Produce valid closing JSON aligned with every House and attribute slot."
    )
    return dspy.Prediction(score=r, feedback=feedback)


class _MetricCallable:
    def __init__(self, reward_mode: RewardMode):
        self.reward_mode = reward_mode

    def __call__(self, example, pred, trace=None):
        return compute_score_metric(example, pred, trace, reward_mode=self.reward_mode)


class _FeedbackCallable:
    def __init__(self, reward_mode: RewardMode):
        self.reward_mode = reward_mode

    def __call__(self, example, pred, trace=None, *args, **kwargs):
        return feedback_metric(example, pred, trace, reward_mode=self.reward_mode, *args, **kwargs)


def evaluate_model(
    model,
    dataset: list[dspy.Example],
    model_name: str,
    *,
    reward_mode: RewardMode,
    num_threads: int,
):
    print(f"\n=== Evaluating {model_name} ===")
    evaluator = Evaluate(
        devset=dataset,
        metric=_MetricCallable(reward_mode),
        num_threads=num_threads,
        display_progress=True,
    )
    eval_result = evaluator(model)
    results = [entry[2] for entry in eval_result.results]
    reward_se = statistics.stdev(results) / (len(results) ** 0.5) if len(results) > 1 else 0
    print(f"Average reward ({reward_mode}): {eval_result.score:.4f}±{reward_se:.4f}")
    return results


def run_multi_evaluation(
    model,
    dataset,
    *,
    reward_mode: RewardMode,
    num_runs,
    num_threads,
):
    all_run_scores = []
    for run_idx in range(num_runs):
        print(f"\n=== Run {run_idx + 1}/{num_runs} ===")
        scores = evaluate_model(model, dataset, f"Run {run_idx + 1}", reward_mode=reward_mode, num_threads=num_threads)
        all_run_scores.append(scores)
    return all_run_scores


def save_optimized_prompt(model, output_dir: str, domain_name: str):
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"{domain_name}_optimized_model.json"
    model.save(str(model_path))
    print(f"Saved optimized model to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="GEPA baseline for ZebraLogic grid puzzles.")

    parser.add_argument("--data-parquet", type=str, default="data/zebralogic/grid_mode_test.parquet")
    parser.add_argument(
        "--optimization-train-parquet",
        type=str,
        default=None,
        help="Optional parquet for GEPA train/val slicing (recommended once available). "
        "If omitted, reuse `--data-parquet` when `--reuse-data-for-gepa-split` is set.",
    )
    parser.add_argument(
        "--final-eval-parquet",
        type=str,
        default=None,
        help="Optional held-out parquet for post-GEPA metrics. Defaults to `--data-parquet`.",
    )
    parser.add_argument(
        "--reuse-data-for-gepa-split",
        action="store_true",
        help="Allow train/validation windows to be carved from `--data-parquet` (potential leakage—debug only).",
    )
    parser.add_argument("--sizes", type=str, default="5*6,6*5,6*6", help="Comma-separated size tags.")
    parser.add_argument(
        "--per-size",
        type=int,
        default=None,
        help="Cap puzzles per size before splitting (omit for unlimited).",
    )
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--final-eval-size", type=int, default=120)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=4096)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--log-dir", type=str, default="baselines/gepa/logs/zebralogic")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="baselines/gepa/results")
    parser.add_argument("--wandb-name", type=str, default="zebralogic_gepa")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--student-lm", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--reflection-lm", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--student-max-tokens", type=int, default=16384)
    parser.add_argument("--reflection-max-tokens", type=int, default=4000)
    parser.add_argument("--reward", choices=("cells", "puzzle"), default="cells")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible server (overrides OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the server (overrides OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking mode (passes enable_thinking=False in extra_body).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)

    reward_mode: RewardMode = args.reward if args.reward in ("cells", "puzzle") else "cells"

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    feedback_fn = _FeedbackCallable(reward_mode)

    optimization_path = args.optimization_train_parquet or args.data_parquet
    eval_path = args.final_eval_parquet or args.data_parquet
    single_file_reuse = (
        args.optimization_train_parquet is None
        and eval_path == optimization_path
        and args.reuse_data_for_gepa_split
    )

    if args.optimization_train_parquet is None and eval_path == optimization_path:
        if not args.reuse_data_for_gepa_split:
            raise SystemExit(
                "Refusing to optimize on the evaluation split without acknowledgement.\n"
                "Provide `--optimization-train-parquet PATH` pointing at train data, "
                "or pass `--reuse-data-for-gepa-split` for debugging-only runs."
            )

    need = args.train_size + args.val_size

    if single_file_reuse:
        # One parquet: disjoint train / val / final-eval slices after one shuffle (debug only).
        combined = dataframe_to_records(
            pd.read_parquet(optimization_path),
            sizes=sizes,
            per_size_limit=args.per_size,
            max_records=None,
        )
        random.shuffle(combined)
        min_total = need + args.final_eval_size
        if len(combined) < min_total:
            raise SystemExit(
                f"With `--reuse-data-for-gepa-split`, need at least train+val+final_eval = {min_total} "
                f"puzzles after filters (got {len(combined)}). Lower sizes or add more rows."
            )
        optim_slice = combined[:need]
        eval_slice = combined[need : need + args.final_eval_size]
        optim_examples = records_to_examples(optim_slice)
        train_subset = optim_examples[: args.train_size]
        val_subset = optim_examples[args.train_size : need]
        eval_examples = records_to_examples(eval_slice)
        print(
            "\nWARNING: `--reuse-data-for-gepa-split` uses a single parquet split into disjoint "
            f"slices from {optimization_path}. This is for smoke tests only; publish separate "
            "train/eval files for paper-quality numbers.\n"
        )
    else:
        train_pool = dataframe_to_records(
            pd.read_parquet(optimization_path),
            sizes=sizes,
            per_size_limit=args.per_size,
            max_records=None,
        )
        random.shuffle(train_pool)

        if len(train_pool) < need:
            raise SystemExit(
                f"Need at least {need} optimization puzzles after `--sizes`/`--per-size` filters "
                f"(got {len(train_pool)})"
            )

        optim_examples = records_to_examples(train_pool)
        train_subset = optim_examples[: args.train_size]
        val_subset = optim_examples[args.train_size : need]

        eval_pool = dataframe_to_records(
            pd.read_parquet(eval_path),
            sizes=sizes,
            per_size_limit=args.per_size,
            max_records=None,
        )
        if not eval_pool:
            raise SystemExit("Final evaluation pool is empty after filtering (check parquets and `--sizes`).")

        random.shuffle(eval_pool)
        eval_examples = records_to_examples(eval_pool[: args.final_eval_size])

    print("Configuration:")
    print(f"  Optimization parquet: {optimization_path}")
    print(f"  Final eval parquet:   {eval_path}")
    print(f"  Train / val puzzles: {len(train_subset)} / {len(val_subset)}")
    print(f"  Final evaluation puzzles: {len(eval_examples)}")
    print(f"  Reward mode: {reward_mode}")
    print(f"  Student LM: {args.student_lm}")
    print(f"  Reflection LM: {args.reflection_lm}")
    print(f"  Temperature: {args.temperature}")

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    llm_kwargs = dict(cache=False, temperature=args.temperature)
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key
    if args.no_thinking:
        llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    llm_kwargs["max_tokens"] = args.student_max_tokens
    student_lm = dspy.LM(args.student_lm, **llm_kwargs)
    dspy.settings.configure(lm=student_lm)

    refl_kwargs = dict(
        temperature=args.temperature,
        max_tokens=args.reflection_max_tokens,
        **({"api_base": api_base} if api_base else {}),
        **({"api_key": api_key} if api_key else {}),
        **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if args.no_thinking else {}),
    )
    reflection_lm = dspy.LM(args.reflection_lm, **refl_kwargs)

    model = ZebraModule()
    gepa_kwargs = {
        "metric": feedback_fn,
        "max_metric_calls": args.max_calls,
        "num_threads": args.num_threads,
        "track_stats": True,
        "reflection_minibatch_size": args.minibatch_size,
        "reflection_lm": reflection_lm,
        "log_dir": args.log_dir,
    }

    if not args.no_wandb:
        gepa_kwargs.update(
            {
                "use_wandb": True,
                "wandb_init_kwargs": {
                    "entity": "bare-sky",
                    "project": "advisor-models-baselines",
                    "name": args.wandb_name,
                },
                "wandb_api_key": os.getenv("WANDB_API_KEY"),
            }
        )

    gepa = GEPA(**gepa_kwargs)

    optimized = gepa.compile(model, trainset=train_subset, valset=val_subset)

    print("\nOptimized signature instructions:")
    for name, pred in optimized.named_predictors():
        print("================================")
        print(f"Predictor: {name}")
        print(pred.signature.instructions)
        print("*********************************")

    save_optimized_prompt(optimized, args.output_dir, "zebralogic")

    print(f"\nRunning {args.num_runs} post-optimization evaluation runs...")
    all_run_scores = run_multi_evaluation(
        optimized,
        eval_examples,
        reward_mode=reward_mode,
        num_runs=args.num_runs,
        num_threads=args.num_threads,
    )

    stats = compute_multi_run_statistics(all_run_scores)
    print("\n=== Final Evaluation Statistics ===")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"SEM: {stats['sem']:.4f}")
    print(f"95% Bootstrap CI: [{stats['bootstrap_ci_lower']:.4f}, {stats['bootstrap_ci_upper']:.4f}]")

    results_file = Path(args.output_dir) / f"zebralogic_gepa_{args.num_runs}runs.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with results_file.open("w") as f:
        json.dump(
            {
                "domain": "zebralogic",
                "reward_mode": reward_mode,
                "num_runs": args.num_runs,
                "num_samples": len(eval_examples),
                "optimization_parquet": optimization_path,
                "final_eval_parquet": eval_path,
                "statistics": stats,
            },
            f,
            indent=2,
        )
    print(f"Saved aggregate results to {results_file}")


if __name__ == "__main__":
    main()
