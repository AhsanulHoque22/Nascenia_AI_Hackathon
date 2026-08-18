"""
Day 9 - Score the FINE-TUNED model with the competition composite metric.

Why this exists: every fine-tuning result so far (Day 8's eval_loss
1.21->0.88, Day 9's 1.13->0.79) is TRAINING LOSS, not the competition
metric. Loss falling proves the model is learning the data; it does not
prove BERTScore/TokenF1/ROUGE-L went up. Before committing money to rented
GPU time, we need the one number that decides whether scaling up is worth
it: how far does fine-tuning actually move the composite from the 0.5037
zero-shot baseline?

Loads the adapter produced by the day9-qwen3-validation kernel (500 train
examples, 1 epoch -- deliberately weak, so treat the result as a DIRECTION
and a lower bound, not the ceiling of a full run).

Scored two ways:
  - Same 15 val examples (seed=42) used for every zero-shot number in
    MODEL_SELECTION.md, for a directly comparable apples-to-apples delta.
  - A larger 100-example sample for a more stable estimate.

Data:   /kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/
Adapter: output of kernel ahsanulhoque48cu/nascenia-day9-qwen3-validation
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sessions may hand out 2 GPUs; see day9 validation kernel

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1",
     "bert-score", "rouge-score"],
    check=True,
)
# Kaggle preinstalls torchao 0.10.0; peft's LoRA dispatcher calls
# is_torchao_available(), which RAISES (not returns False) on a version it
# considers too old, killing PeftModel.from_pretrained. We never use torchao,
# and if it is absent the check returns False and the dispatcher is skipped.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import glob
import json
import re
import time
import unicodedata
from collections import Counter

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42
N_SMALL = 15    # matches every zero-shot number in MODEL_SELECTION.md
N_LARGE = 100   # more stable estimate
MAX_NEW_TOKENS = 200

# The adapter lives in this kernel's mounted input; find it wherever Kaggle put it.
candidates = glob.glob("/kaggle/input/**/adapter/adapter_config.json", recursive=True)
if not candidates:
    raise SystemExit(f"adapter not found. /kaggle/input contains: {glob.glob('/kaggle/input/*')}")
ADAPTER_DIR = os.path.dirname(candidates[0])
print(f"Using adapter: {ADAPTER_DIR}")

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def build_inference_prompt(tokenizer, patient_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# ---- eval_metrics logic, inlined (same as Day 6 methodology) ----
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


# ---- load fine-tuned model (fp16, adapter merged in for fast inference) ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()
model.eval()
print("Fine-tuned model loaded and adapter merged.")

# The large sample is a superset of the small one, so we generate once.
val_all = pd.read_json(f"{DATA_DIR}/val.jsonl", lines=True)
val_small = val_all.sample(n=N_SMALL, random_state=SEED).reset_index(drop=True)
val_large = val_all.sample(n=N_LARGE, random_state=SEED).reset_index(drop=True)


def generate_all(df):
    preds = []
    t0 = time.time()
    for i, (_, r) in enumerate(df.iterrows(), 1):
        prompt = build_inference_prompt(tokenizer, r["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        preds.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        if i % 20 == 0:
            print(f"  {i}/{len(df)} generated ({time.time()-t0:.0f}s)")
    return preds


from bert_score import score as bertscore_score


def score_set(df, preds, label):
    refs = df["output"].tolist()
    _, _, bert_f1 = bertscore_score(
        preds, refs, model_type="bert-base-multilingual-cased", lang="bn",
        batch_size=16, verbose=False, device="cuda",
    )
    bert_f1 = bert_f1.tolist()
    tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg_f1 = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
    composite = [0.5 * b + 0.3 * t + 0.2 * g for b, t, g in zip(bert_f1, tok_f1, rg_f1)]
    res = {
        "label": label,
        "n": len(df),
        "mean_bertscore_f1": round(sum(bert_f1) / len(bert_f1), 4),
        "mean_token_f1": round(sum(tok_f1) / len(tok_f1), 4),
        "mean_rouge_l_f1": round(sum(rg_f1) / len(rg_f1), 4),
        "mean_composite": round(sum(composite) / len(composite), 4),
        "leftover_think_tags": sum("<think>" in p for p in preds),
        "mean_pred_chars": round(sum(len(p) for p in preds) / len(preds), 1),
        "mean_ref_chars": round(sum(len(r) for r in refs) / len(refs), 1),
    }
    print(f"\n=== {label} (n={len(df)}) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


print(f"\nGenerating {N_LARGE} responses with the fine-tuned model...")
preds_large = generate_all(val_large)
res_large = score_set(val_large, preds_large, "fine-tuned (500ex/1ep), 100-example sample")

preds_small = generate_all(val_small)
res_small = score_set(val_small, preds_small, "fine-tuned (500ex/1ep), same 15 as zero-shot")

print("\n" + "=" * 70)
print("COMPARISON vs zero-shot baselines (same 15 examples, seed=42)")
print("=" * 70)
print(f"  Qwen2.5-0.5B zero-shot (Day 5 floor)   : 0.4409")
print(f"  Qwen2.5-1.5B zero-shot (Day 6)         : 0.4979")
print(f"  Qwen3-1.7B   zero-shot (Day 9)         : 0.5037")
print(f"  Qwen3-1.7B   FINE-TUNED (500ex, 1 ep)  : {res_small['mean_composite']}")
delta = res_small["mean_composite"] - 0.5037
print(f"  --> delta from fine-tuning: {delta:+.4f}")
print(f"\n  Leaderboard leaders (checked Aug 15)   : 0.87-0.90")

pd.DataFrame({
    "input": val_large["input"], "reference": val_large["output"], "prediction": preds_large,
}).to_csv("/kaggle/working/day9_finetuned_predictions.csv", index=False)
with open("/kaggle/working/day9_finetuned_eval.json", "w", encoding="utf-8") as f:
    json.dump({"large": res_large, "small": res_small,
               "zeroshot_qwen3_15ex": 0.5037, "delta_15ex": round(delta, 4)},
              f, ensure_ascii=False, indent=2)
print("\nDone.")
