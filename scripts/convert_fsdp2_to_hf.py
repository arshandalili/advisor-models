"""Convert FSDP2 sharded checkpoint to HuggingFace safetensors format.

Usage:
    python scripts/convert_fsdp2_to_hf.py \
        --ckpt_dir ckpts/riddlebench_improved_v2/global_step_36/policy \
        --output_dir ckpts/riddlebench_improved_v2/hf_model

Merges the 8 FSDP2 DTensor shards (model_world_size_8_rank_N.pt) into a single
HuggingFace-compatible model directory with safetensors weights.
"""

import argparse
import os
import shutil
import sys
import json
from pathlib import Path

import torch


def load_shard(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_local_tensor(t):
    """Extract local tensor from DTensor or ShardedTensor, or return as-is."""
    # DTensor (FSDP2)
    if hasattr(t, "_local_tensor"):
        return t._local_tensor.contiguous()
    # torch.distributed._shard.sharded_tensor.ShardedTensor (FSDP1)
    if hasattr(t, "local_shards"):
        shards = t.local_shards()
        if shards:
            return shards[0].tensor.contiguous()
    # Regular tensor
    if isinstance(t, torch.Tensor):
        return t.contiguous()
    raise TypeError(f"Unknown tensor type: {type(t)}")


def get_shard_dim(t) -> int:
    """Return the sharding dimension, or -1 if replicated."""
    if hasattr(t, "placements"):
        from torch.distributed.tensor.placement_types import Shard, Replicate
        for p in t.placements:
            if isinstance(p, Shard):
                return p.dim
    if hasattr(t, "local_shards") and t.local_shards():
        shards = t.local_shards()
        try:
            return shards[0].metadata.shard_offsets.index(
                next(o for o in shards[0].metadata.shard_offsets if o != 0)
            )
        except StopIteration:
            return 0
    return -1  # replicated


def merge_shards(rank_states: list[dict]) -> dict:
    world_size = len(rank_states)
    keys = list(rank_states[0].keys())
    merged = {}
    for key in keys:
        tensors = [rank_states[r][key] for r in range(world_size)]
        dim = get_shard_dim(tensors[0])
        locals_ = [get_local_tensor(t) for t in tensors]
        if dim >= 0:
            full = torch.cat(locals_, dim=dim)
        else:
            full = locals_[0]  # replicated — rank 0 is authoritative
        merged[key] = full.to(torch.bfloat16)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Path to the policy shard directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for HF model")
    parser.add_argument("--world_size", type=int, default=8)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Loading {args.world_size} shards from {ckpt_dir}")
    rank_states = []
    for rank in range(args.world_size):
        path = ckpt_dir / f"model_world_size_{args.world_size}_rank_{rank}.pt"
        if not path.exists():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"  rank {rank} ...", end=" ", flush=True)
        rank_states.append(load_shard(str(path)))
        print("ok")

    print("[convert] Merging shards ...")
    merged = merge_shards(rank_states)
    del rank_states

    print(f"[convert] Saving {len(merged)} tensors to {output_dir}")
    try:
        from safetensors.torch import save_file
        save_file(merged, str(output_dir / "model.safetensors"))
    except ImportError:
        print("safetensors not available, saving as model.bin instead")
        torch.save(merged, str(output_dir / "pytorch_model.bin"))

    # Copy tokenizer + config files from the huggingface/ subdir
    hf_src = ckpt_dir / "huggingface"
    if hf_src.is_dir():
        for f in hf_src.iterdir():
            shutil.copy2(f, output_dir / f.name)
        print(f"[convert] Copied HF metadata from {hf_src}")
    else:
        print(f"WARNING: {hf_src} not found; copying from base model")

    # Update config: set torch_dtype to bfloat16
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = "bfloat16"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"[convert] Done. Model saved to {output_dir}")
    print(f"  Tensors: {len(merged)}")
    sample_key = next(iter(merged))
    print(f"  Sample ({sample_key}): shape={merged[sample_key].shape}, dtype={merged[sample_key].dtype}")


if __name__ == "__main__":
    main()
