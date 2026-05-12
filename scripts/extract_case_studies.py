"""
Extract case study examples: find examples where Improved gets correct
but No Advisor (initial_reward=0) and Untrained advisor get wrong.
Saves detailed text for each case study to JSON.
"""
import argparse
import ast
import json
import os
import sys
import time

import litellm
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from advisor_models.riddlebench.config import STUDENT_SYSTEM_PROMPT, compute_riddle_score

_API_BASE = os.environ.get("API_BASE", None)
_API_KEY = os.environ.get("OPENAI_API_KEY", None)

ADVISOR_API_BASE = "http://127.0.0.1:8000/v1"


def get_advisor_feedback(prompt, advisor_model_name):
    if isinstance(prompt, str):
        try:
            prompt = ast.literal_eval(prompt)
        except Exception:
            prompt = [{"role": "user", "content": prompt}]
    elif hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    resp = litellm.completion(
        model="hosted_vllm/advisor_model",
        messages=prompt,
        temperature=0.0,
        api_base=ADVISOR_API_BASE,
    )
    return resp.choices[0].message.content.strip()


def get_student_response(advisor_feedback, original_question, original_response, student_model):
    if "</think>" in advisor_feedback:
        advisor_feedback = advisor_feedback.split("</think>", 1)[1]
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
    kwargs = {"model": student_model, "messages": messages, "temperature": 0.0}
    if _API_BASE:
        kwargs["api_base"] = _API_BASE
    if _API_KEY:
        kwargs["api_key"] = _API_KEY
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content or ""


def no_advisor_student_response(original_question, original_response, student_model):
    """Re-run student with no advice (just repeat the question)."""
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": original_question},
    ]
    kwargs = {"model": student_model, "messages": messages, "temperature": 0.0}
    if _API_BASE:
        kwargs["api_base"] = _API_BASE
    if _API_KEY:
        kwargs["api_key"] = _API_KEY
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advisor_model", required=True, help="advisor model name tag")
    parser.add_argument("--dataset_path", default="data/riddlebench/validation_150.parquet")
    parser.add_argument("--student_model", default="gpt-4o-mini")
    parser.add_argument("--max_per_type", type=int, default=30)
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset_path)

    # Only consider examples where student initially failed
    failed = df[df["initial_reward"] == 0.0].copy()
    print(f"Total failed examples: {len(failed)}")

    # Collect error types
    failed["error_type"] = failed["reward_spec"].apply(
        lambda x: (x if isinstance(x, dict) else json.loads(x)).get("error_type", "UNKNOWN")
    )
    failed["ground_truth"] = failed["reward_spec"].apply(
        lambda x: (x if isinstance(x, dict) else json.loads(x))["ground_truth"]
    )

    results = []
    for error_type in ["LOGICAL_ERROR", "MISSED_CONSTRAINT", "FALSE_ASSUMPTION"]:
        subset = failed[failed["error_type"] == error_type].head(args.max_per_type)
        print(f"\n--- {error_type}: testing {len(subset)} examples ---")
        for idx, (_, row) in enumerate(subset.iterrows()):
            gt = row["ground_truth"]
            q = row["original_question"]
            orig_resp = row["original_response"]

            print(f"  [{idx+1}/{len(subset)}] GT={gt!r} ...", end=" ", flush=True)
            try:
                adv_fb = get_advisor_feedback(row["prompt"], args.advisor_model)
                adv_resp = get_student_response(adv_fb, q, orig_resp, args.student_model)
                score, _ = compute_riddle_score(adv_resp, gt)
                print(f"score={score:.0f}")
                results.append({
                    "error_type": error_type,
                    "original_question": q,
                    "original_response": orig_resp,
                    "ground_truth": gt,
                    "advisor_feedback": adv_fb,
                    "student_response_with_advice": adv_resp,
                    "score_with_advisor": score,
                })
            except Exception as e:
                print(f"ERROR: {e}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
