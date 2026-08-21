"""
Cheap, no-training validation of the retrieval hypothesis: does answering
each val question with the REAL answer to the most similar TRAIN question
score dramatically higher than our fine-tuned model (~0.556-0.560 composite)?

If yes -> the leaderboard gap (0.524 vs top teams' 0.85-0.90) is plausibly
explained by retrieval-style near-verbatim overlap, and the one remaining
training run should be spent on retrieval-augmented fine-tuning.
If no -> the gap is something else, don't burn the training run on RAG.

Uses the EXACT same composite formula as kaggle_kernels/day11_select_submit
(0.5*BERTScore + 0.3*TokenF1(regex) + 0.2*ROUGE-L(regex)), same N_SELECT=150,
same seed=42, so the number is directly comparable to the model's 0.5597.
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


def main():
    train = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    val = pd.read_json("data/processed/val.jsonl", lines=True)
    sel = val.sample(n=N_SELECT, random_state=SEED).reset_index(drop=True)

    print(f"train pool: {len(train)} rows, val sample: {len(sel)} rows")

    vec = TfidfVectorizer(max_features=50000)
    train_matrix = vec.fit_transform(train["input"].tolist())
    val_matrix = vec.transform(sel["input"].tolist())

    sims = cosine_similarity(val_matrix, train_matrix)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    preds = [train.iloc[i]["output"] for i in best_idx]
    refs = sel["output"].tolist()

    print(f"mean best-match cosine similarity: {best_sim.mean():.4f}")
    print(f"exact input match (sim>0.999): {(best_sim > 0.999).sum()} / {len(sel)}")

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
    print(f"RETRIEVAL BASELINE composite: {composite:.4f}")
    print("compare: fine-tuned model composite was 0.5564-0.5597 (all 5 candidates)")
    print("compare: constant-string baseline composite was 0.5842")

    with open("/tmp/retrieval_baseline_result.json", "w") as f_out:
        json.dump({
            "bertscore": bertscore, "token_f1": mean_tf, "rouge_l": mean_rg,
            "composite": composite, "mean_best_sim": float(best_sim.mean()),
            "exact_match_count": int((best_sim > 0.999).sum()),
        }, f_out, indent=2)


if __name__ == "__main__":
    main()
