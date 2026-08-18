"""
Batched inference / submission generation.

Two things here are worth more than any training hyperparameter:

1. OUTPUT LENGTH. Token-F1 and ROUGE-L are F1 measures, so under-generating
   caps recall outright. References average ~654 tokens (~620 chars); the old
   max_new_tokens=200 produced ~192 chars — a length ratio of 0.31, which
   alone caps Token-F1 near 0.49 no matter how good the content is. The
   measured Token-F1-vs-length curve is sharply asymmetric: going from the
   ~750-char optimum down to 192 costs ~38%, while going up to 1200 costs
   ~0.7%. So aim ABOVE the reference median, and put a floor under it with
   min_new_tokens. BERTScore F1 is length-normalized and barely moves, so
   length is tuned purely for Token-F1 and ROUGE-L, which peak together.

2. BATCHING. 1,000 test rows x ~900 tokens unbatched on a T4 is ~8 hours.
   Batched with left-padding and length-sorted batches it is ~30-45 minutes.
   Decoder-only models REQUIRE left padding for batched generation — with
   right padding the model continues from pad tokens and emits garbage.

Beam search is deliberately not used: for long-form decoder-only generation
it exhibits the beam-search curse (systematically prefers shorter
hypotheses), which is the exact direction that hurts us here, at 2.5-4x the
runtime.

Usage:
    # score a val sample end-to-end
    python src/infer.py --adapter experiments/<run>/adapter --split val --n 200

    # generate the Kaggle submission
    python src/infer.py --adapter experiments/<run>/adapter --split test \
        --out submissions/sub.csv
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_template import build_inference_prompt

DEFAULT_BASE = "Qwen/Qwen3-1.7B"


def load_model(base_model, adapter, dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only + batched generation REQUIRES left padding.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        # Merging folds the adapter into the base weights — meaningfully
        # faster than routing through LoRA layers on every decode step.
        model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate(model, tokenizer, inputs, gen_kwargs, batch_size=16, log_every=5):
    """Length-sorted batched generation; returns predictions in input order."""
    # Sorting by length keeps each batch's padding tight. We restore the
    # original order afterwards so rows still line up with their ids.
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
    preds = [None] * len(inputs)

    t0 = time.time()
    n_batches = (len(order) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(order), batch_size), 1):
        idx = order[start:start + batch_size]
        prompts = [build_inference_prompt(tokenizer, inputs[i]) for i in idx]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, pad_token_id=tokenizer.pad_token_id, **gen_kwargs
            )
        # With left padding every sequence starts generating at the same
        # offset, so this slice is valid for the whole batch.
        gen = out[:, enc["input_ids"].shape[1]:]
        for i, g in zip(idx, gen):
            preds[i] = tokenizer.decode(g, skip_special_tokens=True).strip()
        if bi % log_every == 0 or bi == n_batches:
            el = time.time() - t0
            print(f"  batch {bi}/{n_batches}  {el:.0f}s elapsed  "
                  f"~{el / bi * (n_batches - bi):.0f}s left", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir; omit for zero-shot")
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--n", type=int, default=None, help="subsample size (val only)")
    ap.add_argument("--out", default=None, help="CSV path to write")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--min_new_tokens", type=int, default=400)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; ~0.3 escapes repetition loops")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    df = pd.read_json(os.path.join(args.data_dir, f"{args.split}.jsonl"), lines=True)
    if args.n and args.split == "val":
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"{args.split}: {len(df)} rows")

    model, tokenizer = load_model(args.base_model, args.adapter)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=1,
    )
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.9, top_k=20)
    else:
        gen_kwargs.update(do_sample=False)
    print(f"generation: {gen_kwargs}")

    t0 = time.time()
    preds = generate(model, tokenizer, df["input"].tolist(), gen_kwargs, args.batch_size)
    print(f"generated {len(preds)} in {(time.time()-t0)/60:.1f} min")

    n_think = sum("<think" in p for p in preds)
    print(f"mean pred chars: {sum(len(p) for p in preds)/len(preds):.0f}"
          + (f"   ref chars: {df['output'].str.len().mean():.0f}" if "output" in df else "")
          + f"   leftover <think>: {n_think}")

    if args.split == "test":
        out = args.out or "submissions/submission.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame({"id": df["id"], "output": preds}).to_csv(out, index=False)
        print(f"wrote {out}")
        return

    from eval_metrics import evaluate

    res = evaluate(preds, df["output"].tolist())
    print("\n=== scores ===")
    for k in ["mean_bertscore_f1", "mean_token_f1", "mean_rouge_l_f1", "mean_composite",
              "mean_token_f1_whitespace", "mean_rouge_l_f1_whitespace",
              "mean_composite_whitespace", "length_ratio"]:
        print(f"  {k}: {res[k]:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame({"input": df["input"], "reference": df["output"], "prediction": preds}
                     ).to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_scores.json", "w") as f:
            json.dump({k: v for k, v in res.items() if k.startswith(("mean_", "length_"))},
                      f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
