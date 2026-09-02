"""
Day 16 -- full test-set submission using the Day 14 trained retrieval+edit
adapter, in the same format it was trained/tested on. Pushed "just to see"
the real leaderboard number, even though local signal (0.5496 on 100 val
rows, Day 15) already predicts this lands below the current best submission
(hybrid retrieval, 0.55756 real). Racing the Aug 24 00:00 BD deadline --
full 1000-row generation at these settings has historically taken ~2h+.

Inputs:
  dataset: ahsanulhoque48cu/nascenia-processed-data (train_curated.jsonl, test.jsonl)
           ahsanulhoque48cu/nascenia-edit-adapter (Day 14 trained adapter)
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "scikit-learn"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import glob
import time

import pandas as pd
import torch
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
BASE_MODEL = "Qwen/Qwen3-1.7B"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 900
MIN_NEW_TOKENS = 650
REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM = 6

RETRIEVED_INPUT_CHARS = 200
RETRIEVED_CONTEXT_CHARS = 400
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
        el = time.time() - t0
        print(f"    batch {bi}/{n_batches}  {el/60:.1f}min  "
              f"~{el/bi*(n_batches-bi)/60:.1f}min left", flush=True)
        # write partial progress every log_every batches so we can salvage a
        # submission even if the kernel gets cut off before finishing
        if bi % log_every == 0 or bi == n_batches:
            done = [i for i in range(len(preds)) if preds[i] is not None]
            pd.DataFrame({"id": test["id"].iloc[done], "output": [preds[i] for i in done]}
                        ).to_csv("/kaggle/working/submission_partial.csv", index=False)
    return preds


def normalize_rows(m):
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


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


t_start = time.time()

roots = [p for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
         if "/checkpoint-" not in p]
if not roots:
    raise SystemExit(f"no adapter found. /kaggle/input: {glob.glob('/kaggle/input/*')}")
adapter_dir = os.path.dirname(roots[0])
print(f"adapter: {adapter_dir}", flush=True)

train_pool = pd.read_json(f"{DATA_DIR}/train_curated.jsonl", lines=True)
test = pd.read_json(f"{DATA_DIR}/test.jsonl", lines=True)
print(f"test: {len(test)} rows", flush=True)

print("building hybrid TF-IDF retrieval index...", flush=True)
word_vec = TfidfVectorizer(max_features=50000)
word_matrix = word_vec.fit_transform(train_pool["input"].tolist())
char_vec = TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(3, 5))
char_matrix = char_vec.fit_transform(train_pool["input"].tolist())
word_q = word_vec.transform(test["input"].tolist())
char_q = char_vec.transform(test["input"].tolist())
combined = normalize_rows(cosine_similarity(word_q, word_matrix)) + normalize_rows(cosine_similarity(char_q, char_matrix))
best_idx = combined.argmax(axis=1)

model, tokenizer = load(adapter_dir)

edit_inputs = [
    EDIT_PROMPT.format(
        ex_input=train_pool.iloc[best_idx[j]]["input"][:RETRIEVED_INPUT_CHARS],
        ex_output=train_pool.iloc[best_idx[j]]["output"][:RETRIEVED_CONTEXT_CHARS],
        real_input=test["input"].iloc[j],
    )
    for j in range(len(test))
]
print("\n=== generating full test set with TRAINED retrieval+edit adapter ===", flush=True)
preds = generate(model, tokenizer, edit_inputs)

pd.DataFrame({"id": test["id"], "output": preds}).to_csv("/kaggle/working/submission.csv", index=False)
print(f"\nwrote /kaggle/working/submission.csv in {(time.time()-t_start)/60:.1f}min total", flush=True)
