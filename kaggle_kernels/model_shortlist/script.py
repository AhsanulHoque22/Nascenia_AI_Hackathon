"""
Day 6 - Model shortlist comparison. Runs on Kaggle GPU (T4/P100).

For each candidate model:
  1. Load tokenizer + model, verify param count <= 3B.
  2. Measure tokenizer fragmentation on Bengali text (tokens per character --
     lower is better, means less sequence-length waste).
  3. Zero-shot-generate on the same 15 val examples used in the Day 5 local
     baseline (same seed=42 sample), score with the same composite metric
     used in eval_metrics.py, for an apples-to-apples comparison against the
     Qwen2.5-0.5B floor (composite 0.4409).

Candidates:
  - Qwen/Qwen2.5-1.5B-Instruct
  - Qwen/Qwen2.5-3B-Instruct
  - google/gemma-2-2b-it        (gated on HF -- may fail to load; caught and reported)
  - sarvamai/sarvam-2b-v0.5     (Indic/Bengali-focused candidate)

Data source: /kaggle/input/nascenia-processed-data/val.jsonl
"""

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "bert-score", "rouge-score"],
    check=True,
)

import gc
import json
import re
import time
import unicodedata
from collections import Counter

import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

CAP = 3_000_000_000
N = 15
SEED = 42
VAL_PATH = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/val.jsonl"

CANDIDATES = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-2-2b-it",
    "sarvamai/sarvam-2b-v0.5",
]

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)

# ---- eval_metrics logic, inlined (kernel has no local src/ import) ----
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text):
    return _WORD_RE.findall(unicodedata.normalize("NFC", text))


class _SharedTokenizer:
    def tokenize(self, text):
        return simple_tokenize(text)


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_SharedTokenizer())


def token_f1(pred, ref):
    p, r = simple_tokenize(pred), simple_tokenize(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l_f1(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def bengali_ratio(s):
    return len(re.findall(r"[ঀ-৿]", s)) / max(len(s), 1)


# ---- main ----
val = pd.read_json(VAL_PATH, lines=True).sample(n=N, random_state=SEED).reset_index(drop=True)
refs = val["output"].tolist()

# BERTScore loaded once, reused across candidates
from bert_score import score as bertscore_score

results = {}

for model_id in CANDIDATES:
    print(f"\n{'='*70}\n{model_id}\n{'='*70}")
    row = {"model_id": model_id, "status": "ok"}
    try:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="cuda"
        )
        model.eval()
        load_time = time.time() - t0

        n_params = sum(p.numel() for p in model.parameters())
        under_cap = n_params < CAP
        print(f"  params: {n_params:,} ({n_params/1e9:.3f}B) -- under 3B cap: {under_cap}")
        row["params"] = n_params
        row["under_3b_cap"] = under_cap
        row["load_time_s"] = round(load_time, 1)

        # Tokenizer fragmentation: tokens per Bengali character, over the val sample inputs
        char_counts = val["input"].str.len().sum()
        token_counts = sum(len(tokenizer.encode(t)) for t in val["input"])
        frag = token_counts / char_counts
        print(f"  tokenizer fragmentation: {frag:.3f} tokens/char (over {N} val inputs)")
        row["tokens_per_char"] = round(frag, 4)

        preds = []
        t0 = time.time()
        for _, r in val.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["input"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(gen)
        gen_time = time.time() - t0
        print(f"  generated {N} responses in {gen_time:.1f}s ({gen_time/N:.2f}s/example)")
        row["gen_time_s"] = round(gen_time, 1)

        _, _, bert_f1 = bertscore_score(
            preds, refs, model_type="bert-base-multilingual-cased", lang="bn",
            batch_size=16, verbose=False, device="cuda",
        )
        bert_f1 = bert_f1.tolist()
        tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
        rg_f1 = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
        composite = [0.5 * b + 0.3 * t + 0.2 * rg for b, t, rg in zip(bert_f1, tok_f1, rg_f1)]

        row["mean_bertscore_f1"] = round(sum(bert_f1) / len(bert_f1), 4)
        row["mean_token_f1"] = round(sum(tok_f1) / len(tok_f1), 4)
        row["mean_rouge_l_f1"] = round(sum(rg_f1) / len(rg_f1), 4)
        row["mean_composite"] = round(sum(composite) / len(composite), 4)

        print(f"  mean BERTScore F1: {row['mean_bertscore_f1']}")
        print(f"  mean Token F1:     {row['mean_token_f1']}")
        print(f"  mean ROUGE-L F1:   {row['mean_rouge_l_f1']}")
        print(f"  mean composite:    {row['mean_composite']}")

        pd.DataFrame({"id": val["id"], "input": val["input"], "output": val["output"], "pred": preds}).to_csv(
            f"/kaggle/working/day6_{model_id.replace('/', '_')}.csv", index=False
        )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        row["status"] = f"failed: {type(e).__name__}: {e}"

    results[model_id] = row

print("\n\n=== SUMMARY ===")
summary_df = pd.DataFrame(results.values())
print(summary_df.to_string(index=False))
summary_df.to_csv("/kaggle/working/day6_model_shortlist_summary.csv", index=False)

with open("/kaggle/working/day6_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nDone. Outputs saved to /kaggle/working/.")
