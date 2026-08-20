"""
Day 11b - Checkpoint selection + submission generation.

Second half of the split run (day11_train is the first). Separate because the
two together exceed Kaggle's hard 12h session kill.

Two jobs:

  1. SELECT. Score each saved checkpoint on held-out val with the actual
     competition composite, and keep the winner. Deliberately NOT val loss:
     they are different objectives, and the gap between the best and the last
     checkpoint is routinely larger than any hyperparameter effect.

  2. GENERATE. Produce the 1,000-row test submission with the winner.

The generation config is where most of the score lives. References average
~654 tokens; the previous max_new_tokens=200 produced ~192 chars, a length
ratio of 0.31, which caps Token-F1 near 0.47 no matter how good the content
is — Token-F1 and ROUGE-L are F1 measures, so under-generation destroys
recall. The measured curve is asymmetric: 750 -> 192 chars costs ~38%, while
750 -> 1200 costs ~0.7%. Hence a high ceiling AND a hard floor.

Inputs:
  dataset: ahsanulhoque48cu/nascenia-processed-data
  kernel : ahsanulhoque48cu/nascenia-day11-train   (checkpoints)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bert-score", "rouge-score"],
    check=True,
)
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
N_SELECT = 150          # val rows used to rank checkpoints
BATCH_SIZE = 16
MAX_NEW_TOKENS = 900    # ~850 chars; plateau starts ~630 tokens
MIN_NEW_TOKENS = 650    # ~613 chars — under-generation is the catastrophic failure
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM = 6     # NOT 3: the summarisation default blocks legitimate
                        # trigrams the reference itself contains

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def regex_tok(t):
    # NOTE: Python's \w excludes Unicode Mn/Mc, where Bengali vowel signs
    # live, so this FRAGMENTS Bengali words ("হেলো" -> ['হ','ল']). Kept only
    # for continuity with earlier numbers; whitespace is reported alongside.
    return _WORD_RE.findall(unicodedata.normalize("NFC", t))


def ws_tok(t):
    return unicodedata.normalize("NFC", t).split()


class _Adapter:
    def __init__(self, fn):
        self.fn = fn

    def tokenize(self, text):
        return self.fn(text)


_rouge_re = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_Adapter(regex_tok))
_rouge_ws = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_Adapter(ws_tok))


def token_f1(pred, ref, tok=regex_tok):
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    ov = sum((Counter(p) & Counter(r)).values())
    if not ov:
        return 0.0
    prec, rec = ov / len(p), ov / len(r)
    return 2 * prec * rec / (prec + rec)


def build_prompt(tokenizer, patient_input):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": patient_input}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(model, tokenizer, inputs, batch_size=BATCH_SIZE, log_every=10):
    """Length-sorted batched generation; returns predictions in input order."""
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
    preds = [None] * len(inputs)
    n_batches = (len(order) + batch_size - 1) // batch_size
    t0 = time.time()
    for bi, start in enumerate(range(0, len(order), batch_size), 1):
        idx = order[start:start + batch_size]
        enc = tokenizer([build_prompt(tokenizer, inputs[i]) for i in idx],
                        return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS,
                do_sample=False, num_beams=1,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Left padding means every sequence starts generating at the same
        # offset, so one slice is valid for the whole batch.
        for i, g in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            preds[i] = tokenizer.decode(g, skip_special_tokens=True).strip()
        if bi % log_every == 0 or bi == n_batches:
            el = time.time() - t0
            print(f"    batch {bi}/{n_batches}  {el/60:.1f}min  "
                  f"~{el/bi*(n_batches-bi)/60:.1f}min left", flush=True)
    return preds


def score(preds, refs, bertscore=True):
    from bert_score import score as bs
    out = {}
    if bertscore:
        _, _, f = bs(preds, refs, model_type="bert-base-multilingual-cased",
                     lang="bn", batch_size=16, verbose=False, device="cuda")
        out["bertscore"] = float(f.mean())
    else:
        out["bertscore"] = 0.0
    tf = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg = [_rouge_re.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    tf_w = [token_f1(p, r, ws_tok) for p, r in zip(preds, refs)]
    rg_w = [_rouge_ws.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    m = lambda x: sum(x) / len(x)
    out.update(
        token_f1=m(tf), rouge_l=m(rg),
        token_f1_ws=m(tf_w), rouge_l_ws=m(rg_w),
        composite=0.5 * out["bertscore"] + 0.3 * m(tf) + 0.2 * m(rg),
        composite_ws=0.5 * out["bertscore"] + 0.3 * m(tf_w) + 0.2 * m(rg_w),
        mean_pred_chars=m([len(p) for p in preds]),
        mean_ref_chars=m([len(r) for r in refs]),
    )
    out["length_ratio"] = out["mean_pred_chars"] / max(out["mean_ref_chars"], 1)
    return out


def load(adapter_dir):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # REQUIRED for decoder-only batched generation
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if adapter_dir:
        m = PeftModel.from_pretrained(m, adapter_dir)
        m = m.merge_and_unload()   # folds LoRA into base; faster per decode step
    m.eval()
    m.config.use_cache = True
    return m, tok


# ---- find checkpoints from the training kernel's output ----
# kernel_sources mounts the training kernel's FULL output, which includes
# every intermediate Trainer checkpoint (checkpoint-250, -500, ...), not
# just adapter_final -- scoring those too roughly doubled runtime for no
# benefit (they're mid-training snapshots of a single trajectory, not
# independent submission candidates). Only adapter_final is a real
# candidate; published teammate/merge datasets only ever contain that
# folder anyway, so this filter is a no-op for them.
roots = [p for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
         if "adapter_final" in p]
ckpts = sorted({os.path.dirname(p) for p in roots})
if not ckpts:
    raise SystemExit(f"no adapters found. /kaggle/input: {glob.glob('/kaggle/input/*')}")
print(f"found {len(ckpts)} checkpoint(s):")
for c in ckpts:
    print("  ", c)

# Kaggle's free-tier hard-kills sessions with no warning (~9-12h); this
# script previously had no time budget at all, so a kill mid-ranking-loop
# would lose EVERYTHING -- no ranking, no submission. Two mitigations:
#   1. save the ranking after every single checkpoint, not just at the end
#   2. before starting each new checkpoint, check whether there's still
#      enough budget left to ALSO finish the final test-set generation --
#      that's the one artifact that actually matters (submission.csv), so
#      it's worth skipping remaining checkpoints to protect it.
MAX_HOURS = 8.5
t_start = time.time()

val = pd.read_json(f"{DATA_DIR}/val.jsonl", lines=True)
sel = val.sample(n=N_SELECT, random_state=SEED).reset_index(drop=True)
sel_inputs, sel_refs = sel["input"].tolist(), sel["output"].tolist()

test = pd.read_json(f"{DATA_DIR}/test.jsonl", lines=True)
test_inputs = test["input"].tolist()


def save_ranking(results):
    ranked = sorted(results, key=lambda r: -r["composite"])
    with open("/kaggle/working/checkpoint_selection.json", "w") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    return ranked


# ---- 1. rank checkpoints by the real composite ----
results = []
sec_per_val_row = None
for c in ckpts:
    elapsed = time.time() - t_start
    if sec_per_val_row is not None:
        projected_next_candidate = sec_per_val_row * N_SELECT
        projected_test_gen = sec_per_val_row * len(test_inputs)
        if elapsed + projected_next_candidate + projected_test_gen > MAX_HOURS * 3600:
            print(f"\nSkipping remaining {len(ckpts) - len(results)} checkpoint(s) -- "
                  f"not enough time budget left to also finish test generation. "
                  f"Scored {len(results)} so far.", flush=True)
            break
    print(f"\n=== scoring {os.path.basename(c)} ===", flush=True)
    c_t0 = time.time()
    model, tokenizer = load(c)
    preds = generate(model, tokenizer, sel_inputs)
    s = score(preds, sel_refs)
    s["checkpoint"] = c
    results.append(s)
    print(f"  composite {s['composite']:.4f} (ws {s['composite_ws']:.4f})  "
          f"tokenF1 {s['token_f1']:.4f}  len_ratio {s['length_ratio']:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()
    if sec_per_val_row is None:
        sec_per_val_row = (time.time() - c_t0) / N_SELECT
    save_ranking(results)  # incremental -- survives a kill mid-loop

if not results:
    raise SystemExit("no checkpoint finished scoring within the time budget")

results = save_ranking(results)
best = results[0]
print("\n" + "=" * 70)
print("RANKING")
for r in results:
    print(f"  {r['composite']:.4f}  {os.path.basename(r['checkpoint'])}  "
          f"(len_ratio {r['length_ratio']:.2f})")
print(f"\nBEST: {best['checkpoint']}  composite {best['composite']:.4f}")
print("Reference points — constant-string baseline ~0.60, "
      "zero-shot (200-tok cap) 0.5037")

# ---- 2. generate the submission with the winner ----
print(f"\n=== generating test submission with {os.path.basename(best['checkpoint'])} ===",
      flush=True)
model, tokenizer = load(best["checkpoint"])
t0 = time.time()
preds = generate(model, tokenizer, test_inputs)
print(f"generated {len(preds)} rows in {(time.time()-t0)/60:.1f} min")

mean_chars = sum(len(p) for p in preds) / len(preds)
n_think = sum("<think" in p for p in preds)
pct_helo = 100 * sum(p.startswith("হেলো") for p in preds) / len(preds)
print(f"mean chars {mean_chars:.0f} (references ~619)   leftover <think>: {n_think}")
# 76% of reference responses open with this; a fine-tuned model should have
# learned it. A collapse here means the run did not take.
print(f"opening with 'হেলো': {pct_helo:.1f}%  (references 76.3%)")

pd.DataFrame({"id": test["id"], "output": preds}).to_csv(
    "/kaggle/working/submission.csv", index=False
)
pd.DataFrame({"id": test["id"], "input": test["input"], "output": preds}).to_csv(
    "/kaggle/working/submission_with_inputs.csv", index=False
)
print("\nwrote /kaggle/working/submission.csv")
