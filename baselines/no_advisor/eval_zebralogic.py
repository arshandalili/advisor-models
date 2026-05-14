"""ZebraLogic eval, aligned with WildEval/ZeroEval (ZEBRA_GRID prompt + brace-matched JSON extraction).

Reports cell-level accuracy and puzzle-level exact-match per (NxM) size.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import litellm
import pandas as pd
from tqdm import tqdm

# Official template from ZeroEval/src/templates/ZEBRA_GRID.py
ZEBRA_GRID = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def build_prompt(item):
    """Render the official template, with json_template filled from the gold header."""
    solution = item["solution"]
    columns = solution["header"]
    assert columns[0] == "House"
    num_houses = len(solution["rows"])
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i + 1}"] = {columns[j]: "___" for j in range(1, len(columns))}
    return ZEBRA_GRID.replace("{puzzle}", item["puzzle"]).replace("{json_template}", json.dumps(json_template, indent=4))


def extract_last_complete_json(s):
    """Brace-matched extraction of the last top-level JSON object. Mirrors ZeroEval."""
    if not s:
        return None
    stack = []
    last_json_start = None
    last_json_str = None
    for i, c in enumerate(s):
        if c == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif c == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_json_str = s[last_json_start:i + 1]
                    last_json_start = None
    if not last_json_str:
        return None
    try:
        return json.loads(last_json_str.replace("\n", ""))
    except json.JSONDecodeError:
        return None


def score(pred_obj, solution):
    """Strict cell match (lowercased + stripped). Returns (cell_correct, cell_total, solved)."""
    columns = solution["header"]
    rows = solution["rows"]
    attrs = columns[1:]
    total = len(rows) * len(attrs)
    if not isinstance(pred_obj, dict) or "solution" not in pred_obj or not isinstance(pred_obj["solution"], dict):
        return 0, total, False
    table = pred_obj["solution"]
    correct = 0
    for r in rows:
        house = table.get(f"House {r[0]}")
        if not isinstance(house, dict):
            continue
        for i, attr in enumerate(attrs):
            pv = house.get(attr)
            if pv is None:
                continue
            if str(pv).lower().strip() == str(r[i + 1]).lower().strip():
                correct += 1
    return correct, total, correct == total


def run_one(args):
    item, model, api_base, timeout = args
    prompt = build_prompt(item)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.7, "timeout": timeout}
    if api_base:
        kwargs["api_base"] = api_base
    try:
        r = litellm.completion(**kwargs)
        text = r.choices[0].message.content or ""
    except Exception as e:
        print(f"err id={item['id']}: {e}")
        text = ""
    pred = extract_last_complete_json(text)
    cc, ct, solved = score(pred, item["solution"])
    return {"id": item["id"], "size": item["size"], "cell_correct": cc, "cell_total": ct,
            "solved": int(solved), "parsed": int(pred is not None), "raw": text[-2500:]}


def evaluate(items, model, api_base, max_workers, timeout):
    args_list = [(it, model, api_base, timeout) for it in items]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(ex.map(run_one, args_list), total=len(items), desc="ZebraLogic"):
            results.append(r)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--api_base", default=None)
    p.add_argument("--data_file", default="data/zebralogic/grid_mode_test.parquet")
    p.add_argument("--output_dir", default="baselines/no_advisor/results")
    p.add_argument("--output_suffix", default="")
    p.add_argument("--sizes", default="5*6,6*5,6*6")
    p.add_argument("--per_size", type=int, default=40)
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=1800.0)
    args = p.parse_args()

    df = pd.read_parquet(args.data_file)
    sizes = [s.strip() for s in args.sizes.split(",")]
    items = []
    for s in sizes:
        sub = df[df["size"] == s].head(args.per_size).to_dict("records")
        items.extend(sub)
        print(f"  {s}: {len(sub)} puzzles")
    print(f"Total: {len(items)}")

    results = evaluate(items, args.model, args.api_base, args.max_workers, args.timeout)

    by_size = {}
    for r in results:
        d = by_size.setdefault(r["size"], {"solved": 0, "n": 0, "cells": 0, "ctot": 0, "parsed": 0})
        d["solved"] += r["solved"]; d["n"] += 1
        d["cells"] += r["cell_correct"]; d["ctot"] += r["cell_total"]
        d["parsed"] += r["parsed"]
    print("\nResults by size:")
    for s in sizes:
        if s in by_size:
            d = by_size[s]
            print(f"  {s}: puzzle_acc={d['solved']}/{d['n']}={d['solved']/d['n']:.3f}  "
                  f"cell_acc={d['cells']}/{d['ctot']}={d['cells']/max(1,d['ctot']):.3f}  "
                  f"parsed={d['parsed']}/{d['n']}")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out = os.path.join(args.output_dir, f"zebralogic{suffix}_{args.model.replace('/', '_')}.json")
    with open(out, "w") as f:
        json.dump({"model": args.model, "by_size": by_size, "results": results}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
