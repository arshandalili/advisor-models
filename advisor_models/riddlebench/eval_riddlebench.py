"""Evaluation script for RiddleBench advisor model.

Example usage:
    # Untrained baseline
    python -m advisor_models.riddlebench.eval_riddlebench \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --dataset_path data/riddlebench/validation_gpt-4o-mini.parquet \
        --student_model gpt-4o-mini \
        --num_runs 3

    # Trained checkpoint
    python -m advisor_models.riddlebench.eval_riddlebench \
        --model_name $HOME/ckpts/riddlebench_improved \
        --dataset_path data/riddlebench/validation_gpt-4o-mini.parquet \
        --student_model gpt-4o-mini \
        --num_runs 3

    # Transferability: different student
    python -m advisor_models.riddlebench.eval_riddlebench \
        --model_name $HOME/ckpts/riddlebench_improved \
        --dataset_path data/riddlebench/validation_gpt-4o-mini.parquet \
        --student_model gpt-4.1-mini \
        --num_runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import os

import litellm
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

_API_BASE = os.environ.get("API_BASE", None)
_API_KEY = os.environ.get("OPENAI_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor_models.riddlebench.config import (
    STUDENT_SYSTEM_PROMPT,
    compute_riddle_score,
    extract_advice_section,
)
from utils.eval_utils import (
    add_common_eval_args,
    cleanup_vllm_server,
    compute_multi_run_statistics,
    format_ci_string,
    setup_vllm_server,
)

litellm.drop_params = True


class RiddleBenchEvaluator:
    def __init__(
        self,
        advisor_model: str,
        advisor_api_base: str = "http://127.0.0.1:8000/v1",
        student_model: str = "gpt-4o-mini",
    ):
        self.advisor_model = advisor_model
        self.advisor_api_base = advisor_api_base
        self.student_model = student_model
        self.openai_client = OpenAI()

    def _generate_advisor_feedback(self, prompt: List[Dict[str, str]]) -> str:
        try:
            resp = litellm.completion(
                model=self.advisor_model,
                messages=prompt,
                temperature=0.0,
                api_base=self.advisor_api_base,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Advisor error: {e}")
            return ""

    def _get_student_response(
        self,
        advisor_feedback: str,
        original_question: str,
        original_response: str,
    ) -> str:
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>", 1)[1]
        # Send full advisor output (diagnosis + advice) — matches DiagEnv training behavior
        user_content = (
            f"{advisor_feedback.strip()}\n\n"
            "Your previous answer was wrong. Do NOT adjust it — start completely "
            "from scratch, reason step by step, then give your final answer."
        )

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": original_question},
            {"role": "assistant", "content": original_response},
            {"role": "user", "content": user_content},
        ]
        try:
            kwargs: dict = {"model": self.student_model, "messages": messages, "temperature": 0.0}
            if _API_BASE:
                kwargs["api_base"] = _API_BASE
            if _API_KEY:
                kwargs["api_key"] = _API_KEY
            resp = litellm.completion(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"Student error: {e}")
            return ""

    def _process_example(self, idx_row: tuple) -> Dict[str, Any]:
        idx, row = idx_row
        try:
            prompt = row["prompt"]
            if isinstance(prompt, str):
                try:
                    import ast; prompt = ast.literal_eval(prompt)
                except Exception:
                    prompt = [{"role": "user", "content": prompt}]
            elif hasattr(prompt, "tolist"):
                prompt = prompt.tolist()

            original_question = row["original_question"]
            original_response = row.get("original_response", "")
            ground_truth = row["reward_spec"]["ground_truth"]

            advisor_feedback = self._generate_advisor_feedback(prompt)
            student_response = self._get_student_response(
                advisor_feedback, original_question, original_response
            )
            score, info = compute_riddle_score(student_response, ground_truth)

            return {
                "index": idx,
                "ground_truth": ground_truth,
                "advisor_feedback": advisor_feedback[:500],
                "student_response": student_response[:300],
                "score": score,
                "info": info,
            }
        except Exception as e:
            print(f"Error on example {idx}: {e}")
            return {"index": idx, "score": 0.0, "info": str(e)}

    def evaluate_dataset(
        self,
        dataset_path: str,
        max_workers: int = 12,
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        df = pd.read_parquet(dataset_path)
        if max_examples:
            df = df.sample(n=min(max_examples, len(df)), random_state=42)

        examples = list(df.iterrows())
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_example, ex): ex[0] for ex in examples}
            for future in tqdm(as_completed(futures), total=len(examples)):
                results.append(future.result())

        scores = [r["score"] for r in results]
        return {
            "metrics": {
                "total": len(scores),
                "accuracy": float(np.mean(scores)),
                "correct": int(sum(1 for s in scores if s > 0)),
            },
            "all_scores": scores,
            "detailed_results": results,
        }


def run_multi_evaluation(
    evaluator: RiddleBenchEvaluator,
    dataset_path: str,
    num_runs: int,
    max_examples: Optional[int],
    max_workers: int,
) -> Dict[str, Any]:
    all_run_scores = []
    for run_idx in range(num_runs):
        print(f"\n{'='*60}\nRUN {run_idx + 1}/{num_runs}\n{'='*60}")
        res = evaluator.evaluate_dataset(dataset_path, max_workers, max_examples)
        all_run_scores.append(res["all_scores"])
        print(f"Run {run_idx + 1} accuracy: {res['metrics']['accuracy']:.4f}")
    agg = compute_multi_run_statistics(all_run_scores)
    return {"run_scores": all_run_scores, "aggregate_stats": agg}


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    args = parser.parse_args()

    served_model_name = "advisor_model"
    vllm_process = None
    try:
        vllm_process = setup_vllm_server(
            model_path=args.model_name,
            served_model_name=served_model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        evaluator = RiddleBenchEvaluator(
            advisor_model="hosted_vllm/" + served_model_name,
            advisor_api_base="http://127.0.0.1:8000/v1",
            student_model=args.student_model,
        )

        multi = run_multi_evaluation(
            evaluator, args.dataset_path, args.num_runs, args.max_examples, args.max_workers
        )

        print(f"\n{'='*60}\nFINAL RESULTS — RiddleBench\n{'='*60}")
        print(f"Runs: {args.num_runs}, Dataset: {args.dataset_path}")
        print(f"Student: {args.student_model}")
        print(format_ci_string(multi["aggregate_stats"], "Accuracy"))

        out_dir = os.path.dirname(args.dataset_path)
        out_file = os.path.join(
            out_dir,
            f"eval_riddlebench_{args.model_name.replace('/', '_')}_{args.student_model}.json",
        )
        with open(out_file, "w") as f:
            json.dump(
                {
                    "model": args.model_name,
                    "student": args.student_model,
                    "num_runs": args.num_runs,
                    "aggregate_stats": multi["aggregate_stats"],
                },
                f,
                indent=2,
            )
        print(f"Saved to {out_file}")

    finally:
        cleanup_vllm_server(vllm_process)


if __name__ == "__main__":
    main()
