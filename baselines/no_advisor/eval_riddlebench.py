"""Student-only baseline evaluation for RiddleBench.

Evaluates a single model on RiddleBench without any advisor.

Example usage:
    python -m baselines.no_advisor.eval_riddlebench \
        --model gpt-4o-mini \
        --data_file data/riddlebench/validation_gpt-4o-mini.parquet \
        --num_runs 3 \
        --max_workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import litellm
import numpy as np
import pandas as pd
from tqdm import tqdm

_API_BASE = os.environ.get("API_BASE", None)
_API_KEY = os.environ.get("OPENAI_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.riddlebench.config import compute_riddle_score
from utils.eval_utils import compute_multi_run_statistics, format_ci_string

litellm.drop_params = True


@dataclass
class EvalResult:
    question: str
    response: str
    reward: float
    ground_truth: str


def evaluate_single(args_tuple) -> EvalResult:
    task, model = args_tuple
    question = task["original_question"]
    ground_truth = task["reward_spec"]["ground_truth"]

    try:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.0,
        }
        if _API_BASE:
            kwargs["api_base"] = _API_BASE
        if _API_KEY:
            kwargs["api_key"] = _API_KEY
        resp = litellm.completion(**kwargs)
        response = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"API error: {e}")
        response = ""

    reward, _ = compute_riddle_score(response, ground_truth)
    return EvalResult(question=question[:200], response=response[:200],
                      reward=reward, ground_truth=ground_truth)


def evaluate_dataset(tasks: List[Dict[str, Any]], model: str, max_workers: int) -> List[EvalResult]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(evaluate_single, [(t, model) for t in tasks]),
            total=len(tasks),
            desc="Evaluating",
        ))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", default="results/no_advisor")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data_file)
    tasks = df.to_dict("records")
    if args.num_samples:
        tasks = tasks[: args.num_samples]
    print(f"Evaluating {len(tasks)} examples with {args.model} (no advisor) ...")

    all_run_rewards: List[List[float]] = []
    for run_idx in range(args.num_runs):
        print(f"\nRun {run_idx + 1}/{args.num_runs}")
        results = evaluate_dataset(tasks, args.model, args.max_workers)
        rewards = [r.reward for r in results]
        all_run_rewards.append(rewards)
        print(f"  Accuracy: {np.mean(rewards):.4f}")

    if args.num_runs > 1:
        agg = compute_multi_run_statistics(all_run_rewards)
        print(f"\n{format_ci_string(agg, 'Accuracy')}")
    else:
        agg = {"mean": float(np.mean(all_run_rewards[0]))}

    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir) / f"riddlebench_{args.model.replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "num_runs": args.num_runs,
                   "num_samples": len(tasks), "aggregate_stats": agg}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
