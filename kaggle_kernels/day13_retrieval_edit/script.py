"""
Day 13 -- retrieval + minimal-edit generation, tested on val before deciding
whether to spend a submission on it.

Two prior findings motivate this:
  1. Pure retrieval (copy the nearest real answer verbatim, zero model)
     beats the fine-tuned model: 0.5830 local / 0.55756 real vs 0.5597
     local / 0.52418 real. Real answer text scores higher than anything the
     model generates from scratch.
  2. Day 12's RAG approach (condition generation on a retrieved example,
     framed as a "style reference", then generate FREELY) made things
     WORSE in its own ablation (0.5562 vs 0.5584 without retrieval) --
     free generation conditioned on an example is not the same as reusing
     the example's actual text.

This tries a third framing: explicitly instruct the model to EDIT/ADAPT the
retrieved real answer for the new patient, preserving most of its wording
and structure, changing only what doesn't fit. The goal is to keep
retrieval's high lexical/BERTScore overlap while actually addressing the
new patient's specific question, unlike pure copy-paste which just returns
someone else's case verbatim regardless of whether it fits.

Scores THREE things on the same 150 val rows for a clean comparison:
  a) pure retrieval (copy verbatim, no model at all)
  b) retrieval + edit (this kernel's new idea)
  c) plain generation, no retrieval (the original fine-tuned model)
so the decision of what (if anything) to submit is based on real numbers,
not assumption.

Inputs:
  dataset: ahsanulhoque48cu/nascenia-processed-data (train_curated.jsonl,
           val.jsonl, test.jsonl)
           sanzidislam/nascenia-shard-2-adapter (the currently-submitted
           best LoRA adapter, real leaderboard 0.52418 on its own)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bert-score", "rouge-score", "scikit-learn"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import glob
import json
import re
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42
N_SELECT = 150
BATCH_SIZE = 16
MAX_NEW_TOKENS = 900
MIN_NEW_TOKENS = 650
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM = 6
MAX_HOURS = 8.5

RETRIEVED_INPUT_CHARS = 300
RETRIEVED_CONTEXT_CHARS = 900  # full answer this time, not a truncated style snippet -- it needs to be editable

EDIT_PROMPT = (
    "নিচে একটি পূর্ববর্তী রোগীর প্রশ্ন এবং তার জন্য ডাক্তারের প্রকৃত উত্তর দেওয়া হলো।\n"
    "পূর্ববর্তী প্রশ্ন: {ex_input}\n"
    "পূর্ববর্তী উত্তর: {ex_output}\n\n"
    "এখন একজন নতুন রোগী নিচের প্রশ্নটি করেছেন। উপরের উত্তরটিকে ভিত্তি হিসেবে ব্যবহার করে, "
    "এর গঠন, শৈলী ও অধিকাংশ শব্দ অপরিবর্তিত রেখে শুধুমাত্র প্রয়োজনীয় অংশটুকু নতুন রোগীর "
    "নির্দিষ্ট প্রশ্নের সাথে মানানসই করে পরিবর্তন করুন। সম্পূর্ণ নতুন উত্তর লিখবেন না, বরং "
    "উপরের উত্তরটি সম্পাদনা/অভিযোজন করুন।\n"
    "নতুন রোগীর প্রশ্ন: {real_input}\n"
    "নতুন রোগীর জন্য অভিযোজিত উত্তর:"
)

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def regex_tok(t):
    return _WORD_RE.findall(unicodedata.normalize("NFC", t))


class _Adapter:
    def tokenize(self, text):
        return regex_tok(text)


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_Adapter())


def token_f1(pred, ref):
    p, r = regex_tok(pred), regex_tok(ref)
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


def generate(model, tokenizer, inputs, batch_size=BATCH_SIZE, log_every=5):
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
        for i, g in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            preds[i] = tokenizer.decode(g, skip_special_tokens=True).strip()
        if bi % log_every == 0 or bi == n_batches:
            el = time.time() - t0
            print(f"    batch {bi}/{n_batches}  {el/60:.1f}min  "
                  f"~{el/bi*(n_batches-bi)/60:.1f}min left", flush=True)
    return preds


def score(preds, refs, label):
    print(f"  [{label}] scoring {len(preds)} pairs...", flush=True)
    from bert_score import score as bs
    print(f"  [{label}] bert_score imported, calling scorer...", flush=True)
    _, _, f = bs(preds, refs, model_type="bert-base-multilingual-cased",
                 lang="bn", batch_size=16, verbose=True, device="cuda")
    print(f"  [{label}] bert_score done", flush=True)
    bertscore = float(f.mean())
    tf = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg = [_rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    mean_tf, mean_rg = sum(tf) / len(tf), sum(rg) / len(rg)
    composite = 0.5 * bertscore + 0.3 * mean_tf + 0.2 * mean_rg
    print(f"  [{label}] bertscore {bertscore:.4f}  token_f1 {mean_tf:.4f}  "
          f"rouge_l {mean_rg:.4f}  composite {composite:.4f}", flush=True)
    return {
        "bertscore": bertscore, "token_f1": mean_tf, "rouge_l": mean_rg,
        "composite": composite, "mean_pred_chars": sum(len(p) for p in preds) / len(preds),
    }


def load(adapter_dir):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
    )
    m = PeftModel.from_pretrained(m, adapter_dir)
    m = m.merge_and_unload()
    m.eval()
    m.config.use_cache = True
    return m, tok


def normalize_rows(m):
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


def build_retrieval(train_inputs):
    word_vec = TfidfVectorizer(max_features=50000)
    word_matrix = word_vec.fit_transform(train_inputs)
    char_vec = TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(3, 5))
    char_matrix = char_vec.fit_transform(train_inputs)
    return word_vec, word_matrix, char_vec, char_matrix


def retrieve(word_vec, word_matrix, char_vec, char_matrix, query_inputs):
    word_q = word_vec.transform(query_inputs)
    char_q = char_vec.transform(query_inputs)
    word_sims = cosine_similarity(word_q, word_matrix)
    char_sims = cosine_similarity(char_q, char_matrix)
    combined = normalize_rows(word_sims) + normalize_rows(char_sims)
    return combined.argmax(axis=1)


t_start = time.time()

roots = [p for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
         if "/checkpoint-" not in p]
if not roots:
    raise SystemExit(f"no adapter found. /kaggle/input: {glob.glob('/kaggle/input/*')}")
adapter_dir = os.path.dirname(roots[0])
print(f"adapter: {adapter_dir}", flush=True)

train_pool = pd.read_json(f"{DATA_DIR}/train_curated.jsonl", lines=True)
val = pd.read_json(f"{DATA_DIR}/val.jsonl", lines=True)
sel = val.sample(n=N_SELECT, random_state=SEED).reset_index(drop=True)
test = pd.read_json(f"{DATA_DIR}/test.jsonl", lines=True)

print("building hybrid TF-IDF retrieval index...", flush=True)
word_vec, word_matrix, char_vec, char_matrix = build_retrieval(train_pool["input"].tolist())
best_idx = retrieve(word_vec, word_matrix, char_vec, char_matrix, sel["input"].tolist())
sel_refs = sel["output"].tolist()

model, tokenizer = load(adapter_dir)

# ---- (a) pure retrieval, no model -- SKIPPED, already known: 0.5830 local (see day13 notes) ----
s_retrieval = {"composite": 0.5830, "bertscore": 0.6994, "token_f1": 0.5836, "rouge_l": 0.2913}

# ---- (b) retrieval + edit ----
edit_inputs = [
    EDIT_PROMPT.format(
        ex_input=train_pool.iloc[best_idx[j]]["input"][:RETRIEVED_INPUT_CHARS],
        ex_output=train_pool.iloc[best_idx[j]]["output"][:RETRIEVED_CONTEXT_CHARS],
        real_input=sel["input"].iloc[j],
    )
    for j in range(len(sel))
]
print("\n=== generating retrieval+edit ===", flush=True)
preds_edit = generate(model, tokenizer, edit_inputs)
s_edit = score(preds_edit, sel_refs, "retrieval+edit")

# ---- (c) plain generation, no retrieval ----
print("\n=== generating plain (no retrieval) ===", flush=True)
preds_plain = generate(model, tokenizer, sel["input"].tolist())
s_plain = score(preds_plain, sel_refs, "plain generation")

print(f"\n=== SUMMARY (local, N={N_SELECT}) ===")
print(f"pure retrieval:    {s_retrieval['composite']:.4f}")
print(f"retrieval+edit:    {s_edit['composite']:.4f}")
print(f"plain generation:  {s_plain['composite']:.4f}")

results = {"pure_retrieval": s_retrieval, "retrieval_edit": s_edit, "plain_generation": s_plain}
with open("/kaggle/working/day13_val_comparison.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

best_technique = max(results, key=lambda k: results[k]["composite"])
print(f"\nBEST: {best_technique} ({results[best_technique]['composite']:.4f})")

elapsed = time.time() - t_start
remaining_budget = MAX_HOURS * 3600 - elapsed
print(f"\n{elapsed/60:.1f}min elapsed, {remaining_budget/60:.1f}min budget left", flush=True)
print("Stopping here -- this run is a val-only diagnostic. Test-set generation "
      "happens in a follow-up run only if retrieval+edit clearly wins.", flush=True)
