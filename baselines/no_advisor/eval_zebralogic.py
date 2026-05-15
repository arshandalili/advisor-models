"""ZebraLogic eval with vLLM batched inference (ZEBRA_GRID prompt + brace-matched JSON extraction)."""

import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    solution = item["solution"]
    columns = solution["header"]
    assert columns[0] == "House"
    num_houses = len(solution["rows"])
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i + 1}"] = {columns[j]: "___" for j in range(1, len(columns))}
    return ZEBRA_GRID.replace("{puzzle}", item["puzzle"]).replace("{json_template}", json.dumps(json_template, indent=4))


def extract_last_complete_json(s):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_file", default="data/zebralogic/grid_mode_test.parquet")
    p.add_argument("--output_dir", default="baselines/no_advisor/results")
    p.add_argument("--output_suffix", default="")
    p.add_argument("--sizes", default="5*6,6*5,6*6")
    p.add_argument("--per_size", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.data_file)
    sizes = [s.strip() for s in args.sizes.split(",")]
    items = []
    for s in sizes:
        sub = df[df["size"] == s].head(args.per_size).to_dict("records")
        items.extend(sub)
        print(f"  {s}: {len(sub)} puzzles")
    print(f"Total: {len(items)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(it)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for it in items
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling)

    results = []
    for it, out in zip(items, outputs):
        text = out.outputs[0].text
        pred = extract_last_complete_json(text)
        cc, ct, solved = score(pred, it["solution"])
        results.append({"id": it["id"], "size": it["size"], "cell_correct": cc, "cell_total": ct,
                        "solved": int(solved), "parsed": int(pred is not None), "raw": text[-2500:]})

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
