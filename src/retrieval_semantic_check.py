"""
Retrieval baseline check using SEMANTIC (embedding) similarity instead of
lexical TF-IDF, for the retrieval step. BERTScore is 50% of the competition
metric and is semantic, not lexical -- TF-IDF retrieval (word or char
n-gram) can only ever optimize the other 50% (Token-F1 + ROUGE-L) well.
A retrieval signal that matches on meaning could find topically-correct
answers TF-IDF misses due to paraphrasing, and should lift BERTScore
specifically.

Reuses bert-base-multilingual-cased (already cached locally from
eval_metrics/bert-score) as a mean-pooled sentence encoder -- no new model
download.

Compares against: word TF-IDF 0.5757, char n-gram 0.5817, hybrid 0.5830
(all local, N_SELECT=150, seed=42).

Usage: python src/retrieval_semantic_check.py
"""

import json
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

SEED = 42
N_SELECT = 150
TRAIN_POOL_SAMPLE = 3000  # 15000 was too slow on this machine's CPU; directional signal only
MODEL_NAME = "bert-base-multilingual-cased"
BATCH_SIZE = 16

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def regex_tok(t):
    return _WORD_RE.findall(unicodedata.normalize("NFC", t))


def token_f1(pred, ref):
    p, r = regex_tok(pred), regex_tok(ref)
    if not p or not r:
        return 0.0
    ov = sum((Counter(p) & Counter(r)).values())
    if not ov:
        return 0.0
    prec, rec = ov / len(p), ov / len(r)
    return 2 * prec * rec / (prec + rec)


def embed(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Mean-pooled last-hidden-state embeddings, L2-normalized."""
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=128)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.numpy())
            print(f"  embedded {i}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def main():
    torch.set_num_threads(4)
    train = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    train = train.sample(n=TRAIN_POOL_SAMPLE, random_state=SEED).reset_index(drop=True)
    val = pd.read_json("data/processed/val.jsonl", lines=True)
    sel = val.sample(n=N_SELECT, random_state=SEED).reset_index(drop=True)
    print(f"train pool (subsampled): {len(train)} rows, val sample: {len(sel)} rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("embedding train pool inputs...")
    train_emb = embed(train["input"].tolist(), tokenizer, model)
    print("embedding val sample inputs...")
    val_emb = embed(sel["input"].tolist(), tokenizer, model)

    sims = val_emb @ train_emb.T
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    print(f"mean best-match semantic similarity: {best_sim.mean():.4f}")

    preds = [train.iloc[i]["output"] for i in best_idx]
    refs = sel["output"].tolist()

    from rouge_score import rouge_scorer

    class _Adapter:
        def tokenize(self, text):
            return regex_tok(text)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_Adapter())
    tf = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]

    from bert_score import score as bs
    _, _, f = bs(preds, refs, model_type="bert-base-multilingual-cased",
                 lang="bn", batch_size=16, verbose=False, device="cpu")
    bertscore = float(f.mean())

    mean_tf = sum(tf) / len(tf)
    mean_rg = sum(rg) / len(rg)
    composite = 0.5 * bertscore + 0.3 * mean_tf + 0.2 * mean_rg

    print(f"\nbertscore {bertscore:.4f}  token_f1 {mean_tf:.4f}  rouge_l {mean_rg:.4f}")
    print(f"SEMANTIC RETRIEVAL composite: {composite:.4f}")
    print("compare: word TF-IDF 0.5757, char n-gram 0.5817, hybrid 0.5830 (all lexical)")

    with open("/tmp/retrieval_semantic_result.json", "w") as f_out:
        json.dump({"bertscore": bertscore, "token_f1": mean_tf, "rouge_l": mean_rg,
                    "composite": composite}, f_out, indent=2)


if __name__ == "__main__":
    main()
