"""Dataset construction for RiddleBench domain.

Downloads ai4bharat/RiddleBench, queries GPT-4o-mini for k=2 initial responses,
computes null rewards, categorizes errors, and saves train/val parquet files.

Example usage:
    python advisor_models/riddlebench/construct_dataset.py \
        --output_dir data/riddlebench \
        --train_size 250 \
        --val_size 150 \
        --model gpt-4o-mini \
        --k 2
"""

from __future__ import annotations

import argparse
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import os

import litellm
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm

from advisor_models.riddlebench.config import (
    ADVISOR_INSTRUCTIONS,
    ADVISOR_SYSTEM_PROMPT,
    ERROR_CATEGORIES,
    compute_riddle_score,
)

litellm.drop_params = True

_API_BASE = os.environ.get("API_BASE", None)
_API_KEY = os.environ.get("OPENAI_API_KEY", None)


def _call_model(messages: List[Dict[str, str]], model: str, temperature: float = 0.7) -> str:
    try:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if _API_BASE:
            kwargs["api_base"] = _API_BASE
        if _API_KEY:
            kwargs["api_key"] = _API_KEY
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[API error] {e}")
        return ""


def get_initial_response(question: str, model: str) -> str:
    return _call_model([{"role": "user", "content": question}], model, temperature=0.7)


def categorize_error(riddle_type: str) -> str:
    """Map riddle type to error category without extra API calls."""
    return ERROR_CATEGORIES.get(riddle_type, "LOGICAL_ERROR")


def build_advisor_prompt(question: str, k_responses: List[str]) -> List[Dict[str, str]]:
    attempts = "\n\n".join(f"Attempt {i + 1}:\n{r}" for i, r in enumerate(k_responses))
    user_content = (
        f"{question}\n\n"
        f"The student made the following attempt(s):\n\n{attempts}\n\n"
        f"{ADVISOR_INSTRUCTIONS}"
    )
    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def process_problem(problem: Dict[str, Any], model: str, k: int = 2) -> Dict[str, Any] | None:
    """Process one RiddleBench example into a training row."""
    question = problem["question"]
    ground_truth = problem["answer"]
    riddle_type = problem.get("type", "")

    # k initial responses (in parallel within the problem)
    with ThreadPoolExecutor(max_workers=k + 1) as ex:
        futures_k = [ex.submit(get_initial_response, question, model) for _ in range(k)]
        future_null = ex.submit(get_initial_response, question, model)
        k_responses = [f.result() for f in futures_k]
        null_response = future_null.result()

    original_response = k_responses[0]
    initial_reward, _ = compute_riddle_score(original_response, ground_truth)
    null_reward, _ = compute_riddle_score(null_response, ground_truth)

    error_type = categorize_error(riddle_type) if initial_reward == 0.0 else ""

    advisor_prompt = build_advisor_prompt(question, k_responses)

    return {
        "prompt": advisor_prompt,
        "env_class": "riddlebench",
        "reward_spec": {
            "ground_truth": ground_truth,
            "error_type": error_type,
            "null_reward": null_reward,
            "is_null_advice": False,
        },
        "model": model,
        "original_question": question,
        "original_response": original_response,
        "k_responses": k_responses,
        "initial_reward": initial_reward,
    }


def inject_null_advice_rows(rows: List[Dict[str, Any]], fraction: float = 0.20) -> List[Dict[str, Any]]:
    """Deep-copy ~fraction of rows and mark them as null-advice (counterfactual baseline)."""
    n_null = max(1, int(len(rows) * fraction))
    null_rows = []
    for row in random.sample(rows, n_null):
        null_row = copy.deepcopy(row)
        null_row["reward_spec"] = dict(null_row["reward_spec"])
        null_row["reward_spec"]["is_null_advice"] = True
        null_row["reward_spec"]["null_reward"] = 0.0  # no subtraction on null rows
        null_rows.append(null_row)
    return rows + null_rows


def process_problems_parallel(
    problems: List[Dict[str, Any]],
    model: str,
    k: int,
    max_workers: int = 12,
) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_problem, p, model, k) for p in problems]
        for future in tqdm(as_completed(futures), total=len(problems), desc="Processing"):
            try:
                row = future.result()
                if row is not None:
                    results.append(row)
            except Exception as e:
                print(f"Error: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--train_size", type=int, default=250)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--k", type=int, default=2, help="Number of initial student responses")
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ai4bharat/RiddleBench ...")
    ds = load_dataset("ai4bharat/RiddleBench", trust_remote_code=True)["train"]
    all_problems = list(ds)
    random.shuffle(all_problems)

    total = args.train_size + args.val_size
    if len(all_problems) < total:
        raise ValueError(f"Not enough examples: {len(all_problems)} < {total}")

    train_problems = all_problems[: args.train_size]
    val_problems = all_problems[args.train_size : total]

    print(f"Processing {args.train_size} train examples ...")
    train_rows = process_problems_parallel(train_problems, args.model, args.k, args.max_workers)
    train_rows = inject_null_advice_rows(train_rows, fraction=0.20)
    random.shuffle(train_rows)

    print(f"Processing {args.val_size} val examples ...")
    val_rows = process_problems_parallel(val_problems, args.model, args.k, args.max_workers)
    # No null-advice in val set
    random.shuffle(val_rows)

    if train_rows:
        Dataset.from_list(train_rows).to_parquet(output_dir / f"train_{args.model}.parquet")
        avg = sum(r["initial_reward"] for r in train_rows) / len(train_rows)
        print(f"Saved {len(train_rows)} train rows (avg initial reward: {avg:.3f})")

    if val_rows:
        Dataset.from_list(val_rows).to_parquet(output_dir / f"validation_{args.model}.parquet")
        avg = sum(r["initial_reward"] for r in val_rows) / len(val_rows)
        print(f"Saved {len(val_rows)} val rows (avg initial reward: {avg:.3f})")


if __name__ == "__main__":
    main()
