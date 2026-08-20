"""
Merge multiple independently-trained LoRA adapters by weight averaging
("LoRA soup" / FedAvg-style averaging).

WHY THIS IS VALID HERE, SPECIFICALLY: naively averaging two models' weights
only works when they live in compatible regions of weight space — usually
because they share the same initialization. Every adapter this script merges
was created by kaggle_kernels/day11_train{,_teammate}/script.py, all using
seed=42 and identical code up to get_peft_model(), so LoRA's A matrix (the
only randomly-initialized part; B starts at zero per the LoRA paper) has the
IDENTICAL random init in every adapter before local training diverges it.
Each adapter was then trained on a disjoint, roughly-IID shard of the same
underlying distribution (random contiguous slices of one shuffle). That is
close to the textbook FedAvg setting, where weight averaging has real
theoretical and empirical grounding — this is not "average two unrelated
models and hope."

WHAT THIS DOES NOT DO: decide whether the merge is actually better than any
individual adapter. It never is guaranteed to be. Use --eval (or
score_adapters.py) to check on real held-out data before trusting the
output — never submit a merge you have not scored.

Usage:
    python src/merge_adapters.py \
        --adapters experiments/shard0.../adapter experiments/shard2.../adapter \
                   experiments/shard3.../adapter experiments/shard4.../adapter \
        --out experiments/merged_soup/adapter
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def load_configs(adapter_dirs):
    configs = []
    for d in adapter_dirs:
        with open(os.path.join(d, "adapter_config.json")) as f:
            configs.append(json.load(f))
    return configs


def assert_compatible(adapter_dirs, configs):
    """Refuse to silently average incompatible adapters. A shape mismatch or
    a differing r/alpha/target_modules means these were not trained under
    the shared-init assumption this technique depends on."""
    keys_to_check = ["r", "lora_alpha", "target_modules", "peft_type", "task_type"]

    def normalize(cfg):
        d = {k: cfg.get(k) for k in keys_to_check}
        # target_modules is a SET of module names -- LoRA applies to each
        # independently, so list order has no effect on training or
        # compatibility. It just comes back differently ordered across
        # separate training processes (set serialization isn't stable).
        # Comparing it as an ordered list produces false positives.
        if d.get("target_modules") is not None:
            d["target_modules"] = sorted(d["target_modules"])
        return d

    base = normalize(configs[0])
    for d, cfg in zip(adapter_dirs, configs):
        this = normalize(cfg)
        if this != base:
            raise SystemExit(
                f"REFUSING TO MERGE: {d} has incompatible LoRA config.\n"
                f"  expected: {base}\n  got:      {this}\n"
                f"Averaging adapters with different rank/alpha/target_modules "
                f"produces a nonsense result, not an error you'd notice."
            )


def merge(adapter_dirs, out_dir, weights=None):
    if len(adapter_dirs) < 2:
        raise SystemExit("need >=2 adapters to merge; that's the whole point")

    configs = load_configs(adapter_dirs)
    assert_compatible(adapter_dirs, configs)

    weights = weights or [1.0] * len(adapter_dirs)
    if len(weights) != len(adapter_dirs):
        raise SystemExit("--weights must have one value per adapter")
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    tensors = [load_file(os.path.join(d, "adapter_model.safetensors")) for d in adapter_dirs]
    base_keys = set(tensors[0].keys())
    for d, t in zip(adapter_dirs, tensors):
        if set(t.keys()) != base_keys:
            raise SystemExit(
                f"REFUSING TO MERGE: {d}'s adapter has different tensor keys "
                f"than {adapter_dirs[0]} — not the same architecture/target_modules."
            )

    merged = {}
    for key in base_keys:
        shape = tensors[0][key].shape
        for d, t in zip(adapter_dirs, tensors):
            if t[key].shape != shape:
                raise SystemExit(
                    f"REFUSING TO MERGE: {d}'s '{key}' has shape {t[key].shape}, "
                    f"expected {shape}. Adapters are not compatible."
                )
        acc = torch.zeros_like(tensors[0][key], dtype=torch.float32)
        for w, t in zip(weights, tensors):
            acc += w * t[key].float()
        merged[key] = acc.to(tensors[0][key].dtype)

    os.makedirs(out_dir, exist_ok=True)
    save_file(merged, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(adapter_dirs[0], "adapter_config.json"), out_dir)
    # Tokenizer is identical across all adapters (same base model, none of
    # them add tokens) — copy from any one of them so downstream loading
    # (src/infer.py) works without also needing --base_model wired through.
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(adapter_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy(src, out_dir)

    print(f"merged {len(adapter_dirs)} adapters (weights={[round(w,3) for w in weights]}) "
          f"-> {out_dir}")
    print(f"tensor keys: {len(merged)}  |  source adapters:")
    for d in adapter_dirs:
        print(f"  {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="paths to adapter directories (each with adapter_model.safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="optional per-adapter weight, e.g. to weight a "
                        "2-shard chained adapter higher than a 1-shard one. "
                        "Defaults to equal weighting.")
    args = ap.parse_args()
    merge(args.adapters, args.out, args.weights)


if __name__ == "__main__":
    main()
