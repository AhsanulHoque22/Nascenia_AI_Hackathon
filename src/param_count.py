"""
Rulebook compliance check: total parameter count at inference must be
<= 3,000,000,000 (3B), including any LoRA/adapter weights.

Usage:
    python src/param_count.py <base_model_id_or_path> [--adapter <peft_adapter_path>]

Examples:
    python src/param_count.py Qwen/Qwen2.5-1.5B-Instruct
    python src/param_count.py Qwen/Qwen2.5-1.5B-Instruct --adapter experiments/run3/adapter
"""

import argparse

CAP = 3_000_000_000


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model id or local path (base model)")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    base_params = count_params(model)
    print(f"  base params: {base_params:,} ({base_params/1e9:.3f}B)")

    total_params = base_params
    if args.adapter:
        from peft import PeftModel

        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        total_params = count_params(model)
        adapter_params = total_params - base_params
        print(f"  adapter params: {adapter_params:,} ({adapter_params/1e6:.2f}M)")

    print(f"\nTOTAL inference-time params: {total_params:,} ({total_params/1e9:.4f}B)")
    print(f"3B cap: {CAP:,}")
    if total_params <= CAP:
        print(f"PASS — {(CAP - total_params)/1e6:.1f}M params of headroom remaining")
    else:
        print(f"FAIL — {(total_params - CAP)/1e6:.1f}M params OVER the cap")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
