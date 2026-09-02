"""
Full test-set submission using the hybrid (word-level + char n-gram TF-IDF,
row-normalized and averaged) retrieval nearest-neighbor, which scored best
of the retrieval variants locally (0.5830 vs 0.5757 word-only / 0.5817
char-only -- see src/retrieval_hybrid_check.py).

Usage: python src/retrieval_hybrid_submission.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = "submissions/retrieval_hybrid_submission.csv"


def normalize_rows(m):
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


def main():
    train = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    test = pd.read_json("data/processed/test.jsonl", lines=True)
    print(f"train pool: {len(train)} rows, test: {len(test)} rows")

    word_vec = TfidfVectorizer(max_features=50000)
    word_train = word_vec.fit_transform(train["input"].tolist())
    word_test = word_vec.transform(test["input"].tolist())
    word_sims = cosine_similarity(word_test, word_train)

    char_vec = TfidfVectorizer(max_features=50000, analyzer="char_wb", ngram_range=(3, 5))
    char_train = char_vec.fit_transform(train["input"].tolist())
    char_test = char_vec.transform(test["input"].tolist())
    char_sims = cosine_similarity(char_test, char_train)

    combined = normalize_rows(word_sims) + normalize_rows(char_sims)
    best_idx = combined.argmax(axis=1)

    preds = [train.iloc[i]["output"] for i in best_idx]
    out = pd.DataFrame({"id": test["id"], "output": preds})

    assert len(out) == len(test)
    assert out["output"].notna().all()
    assert not out["id"].duplicated().any()

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
