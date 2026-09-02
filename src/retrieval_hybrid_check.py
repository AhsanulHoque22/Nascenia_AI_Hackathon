"""
Retrieval baseline check: average word-level TF-IDF cosine similarity and
char n-gram TF-IDF cosine similarity to pick the nearest train neighbor.
Compares against word-level alone (0.5757 local) and char n-gram alone
(0.5817 local).

Usage: python src/retrieval_hybrid_check.py
"""

import json
import re
import unicodedata
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
N_SELECT = 150

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


def normalize_rows(m):
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


def main():
    train = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    val = pd.read_json("data/processed/val.jsonl", lines=True)
    sel = val.sample(n=N_SELECT, random_state=SEED).reset_index(drop=True)

    print(f"train pool: {len(train)} rows, val sample: {len(sel)} rows")

    word_vec = TfidfVectorizer(max_features=50000)
    word_train = word_vec.fit_transform(train["input"].tolist())
    word_val = word_vec.transform(sel["input"].tolist())
    word_sims = cosine_similarity(word_val, word_train)

    char_vec = TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(3, 5))
    char_train = char_vec.fit_transform(train["input"].tolist())
    char_val = char_vec.transform(sel["input"].tolist())
    char_sims = cosine_similarity(char_val, char_train)

    # Row-normalize each so word/char scales don't dominate each other, then average.
    combined = normalize_rows(word_sims) + normalize_rows(char_sims)
    best_idx = combined.argmax(axis=1)

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
    print(f"HYBRID RETRIEVAL composite: {composite:.4f}")
    print("compare: word-level alone 0.5757, char n-gram alone 0.5817")

    with open("/tmp/retrieval_hybrid_result.json", "w") as f_out:
        json.dump({
            "bertscore": bertscore, "token_f1": mean_tf, "rouge_l": mean_rg,
            "composite": composite,
        }, f_out, indent=2)


if __name__ == "__main__":
    main()
