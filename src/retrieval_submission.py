"""
Diagnostic submission: no model at all. For each test question, answer with
the REAL train answer to the most similar train question (TF-IDF cosine
nearest neighbor). CPU-only, no GPU/training needed.

Purpose: the local retrieval check (src/retrieval_baseline_check.py) scored
0.5757 composite on val -- close to the fine-tuned model's own 0.5564-0.5597.
But the real leaderboard gap (0.524 vs top teams' 0.85-0.90, confirmed via
`kaggle competitions leaderboard -d`: rank 90/98) is far too large to be
explained by local-to-real calibration alone. This submission tests whether
near-verbatim retrieval overlap is what's actually driving the top of the
real leaderboard -- a real data point, not another local estimate.

Usage:
    python src/retrieval_submission.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = "submissions/retrieval_submission.csv"


def main():
    train = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    test = pd.read_json("data/processed/test.jsonl", lines=True)
    print(f"train pool: {len(train)} rows, test: {len(test)} rows")

    vec = TfidfVectorizer(max_features=50000)
    train_matrix = vec.fit_transform(train["input"].tolist())
    test_matrix = vec.transform(test["input"].tolist())

    sims = cosine_similarity(test_matrix, train_matrix)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    print(f"mean best-match cosine similarity: {best_sim.mean():.4f}")
    print(f"near-exact match (sim>0.999): {(best_sim > 0.999).sum()} / {len(test)}")

    preds = [train.iloc[i]["output"] for i in best_idx]
    out = pd.DataFrame({"id": test["id"], "output": preds})

    assert len(out) == len(test)
    assert out["output"].notna().all()
    assert not out["id"].duplicated().any()

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
