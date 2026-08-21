"""
Day 12b — score + submit for the retrieval-augmented adapter.

Generation must use the SAME retrieval-enriched prompt format the model was
trained on (src/prepare_rag_data.py's format), or the model never sees the
retrieved-context pattern it learned to lean on. Retrieval here is against
the full curated train pool for every val/test input -- no self-exclusion
needed since val/test rows were never in the training pool.

Includes a cheap ablation: scores the SAME adapter both WITH and WITHOUT
retrieved context on the same 150 val rows, so we can tell whether the
retrieval-conditioning technique actually worked before trusting it for the
real submission, rather than assuming it did.

Inputs:
  dataset: ahsanulhoque48cu/nascenia-processed-data (train_curated.jsonl,
           val.jsonl, test.jsonl)
           ahsanulhoque48cu/nascenia-rag-adapter (the day12 trained adapter)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

RETRIEVED_INPUT_CHARS = 200
RETRIEVED_CONTEXT_CHARS = 300
REFERENCE_LABEL = (
    "নিচে একটি পূর্ববর্তী অনুরূপ প্রশ্ন ও তার উত্তরের উদাহরণ দেওয়া হলো, "
    "শুধুমাত্র ভাষা ও গঠনশৈলীর নির্দেশনা হিসেবে ব্যবহার করুন:\n"
    "উদাহরণ প্রশ্ন: {ex_input}\n"
    "উদাহরণ উত্তর: {ex_output}\n\n"
    "এখন নিচের প্রকৃত রোগীর প্রশ্নের উত্তর দিন:\n{real_input}"
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


def generate(model, tokenizer, inputs, batch_size=BATCH_SIZE, log_every=10):
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


def score(preds, refs):
    from bert_score import score as bs
    _, _, f = bs(preds, refs, model_type="bert-base-multilingual-cased",
                 lang="bn", batch_size=16, verbose=False, device="cuda")
    bertscore = float(f.mean())
    tf = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg = [_rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    mean_tf, mean_rg = sum(tf) / len(tf), sum(rg) / len(rg)
    return {
        "bertscore": bertscore, "token_f1": mean_tf, "rouge_l": mean_rg,
        "composite": 0.5 * bertscore + 0.3 * mean_tf + 0.2 * mean_rg,
        "mean_pred_chars": sum(len(p) for p in preds) / len(preds),
        "mean_ref_chars": sum(len(r) for r in refs) / len(refs),
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


def build_retrieval_index(train_inputs):
    vec = TfidfVectorizer(max_features=50000, dtype=np.float32)
    matrix = vec.fit_transform(train_inputs)
    return vec, matrix


def retrieve_and_enrich(vec, pool_matrix, pool_df, query_inputs, chunk=200):
    """Same format as training -- no self-exclusion needed, queries are
    val/test rows that were never in the training pool."""
    enriched = []
    for start in range(0, len(query_inputs), chunk):
        batch = query_inputs[start:start + chunk]
        q_matrix = vec.transform(batch)
        block = (q_matrix @ pool_matrix.T).toarray()
        best_idx = block.argmax(axis=1)
        for j, real_input in enumerate(batch):
            ex_input = pool_df.iloc[best_idx[j]]["input"][:RETRIEVED_INPUT_CHARS]
            ex_output = pool_df.iloc[best_idx[j]]["output"][:RETRIEVED_CONTEXT_CHARS]
            enriched.append(REFERENCE_LABEL.format(
                ex_input=ex_input, ex_output=ex_output, real_input=real_input))
    return enriched


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

print("building retrieval index over train pool...", flush=True)
vec, pool_matrix = build_retrieval_index(train_pool["input"].tolist())

print("retrieving + enriching val prompts...", flush=True)
sel_inputs_rag = retrieve_and_enrich(vec, pool_matrix, train_pool, sel["input"].tolist())
sel_refs = sel["output"].tolist()

model, tokenizer = load(adapter_dir)

# ---- ablation: same adapter, WITH vs WITHOUT retrieved context ----
print("\n=== scoring WITHOUT retrieval (plain prompt, ablation) ===", flush=True)
preds_plain = generate(model, tokenizer, sel["input"].tolist())
s_plain = score(preds_plain, sel_refs)
print(f"  composite {s_plain['composite']:.4f}  (no retrieval)", flush=True)

print("\n=== scoring WITH retrieval (trained format) ===", flush=True)
preds_rag = generate(model, tokenizer, sel_inputs_rag)
s_rag = score(preds_rag, sel_refs)
print(f"  composite {s_rag['composite']:.4f}  (with retrieval)", flush=True)

use_retrieval = s_rag["composite"] > s_plain["composite"]
print(f"\nRETRIEVAL {'HELPED' if use_retrieval else 'DID NOT HELP'} "
      f"({s_rag['composite']:.4f} vs {s_plain['composite']:.4f})", flush=True)

with open("/kaggle/working/rag_ablation.json", "w") as f:
    json.dump({"with_retrieval": s_rag, "without_retrieval": s_plain,
               "use_retrieval": use_retrieval}, f, indent=2, ensure_ascii=False)

# ---- generate the actual test submission with whichever won ----
elapsed = time.time() - t_start
remaining_budget = MAX_HOURS * 3600 - elapsed
print(f"\n{elapsed/60:.1f}min elapsed, {remaining_budget/60:.1f}min budget left "
      f"for {len(test)}-row test generation", flush=True)

print(f"\n=== generating test submission ({'WITH' if use_retrieval else 'WITHOUT'} retrieval) ===",
      flush=True)
if use_retrieval:
    test_inputs = retrieve_and_enrich(vec, pool_matrix, train_pool, test["input"].tolist())
else:
    test_inputs = test["input"].tolist()

t0 = time.time()
preds = generate(model, tokenizer, test_inputs)
print(f"generated {len(preds)} rows in {(time.time()-t0)/60:.1f} min")

mean_chars = sum(len(p) for p in preds) / len(preds)
pct_helo = 100 * sum(p.startswith("হেলো") for p in preds) / len(preds)
print(f"mean chars {mean_chars:.0f} (references ~619)")
print(f"opening with 'হেলো': {pct_helo:.1f}%  (references 76.3%)")

pd.DataFrame({"id": test["id"], "output": preds}).to_csv(
    "/kaggle/working/submission.csv", index=False
)
print("\nwrote /kaggle/working/submission.csv")
