"""
Train/test near-duplicate overlap check.

Motivation: leaderboard leaders sit at 0.87-0.90 composite, but that score
requires near-verbatim reproduction of reference responses (TokenF1 ~0.8,
ROUGE-L ~0.78) — a level normal open-ended generation does not reach. One
structural explanation is that test inputs have near-duplicates in the
training data, making retrieval/memorization far stronger than generation.

This checks that directly: for each test input, find its most similar train
input (TF-IDF cosine). Also checks train-internal duplication and val/train
leakage.

Memory notes: this runs on a laptop with ~2GB free RAM against a 98k-row
corpus, so it uses word unigrams (not char n-grams — those produce ~10x the
non-zeros and get the process OOM-killed), float32, and chunked query
scoring. Unigram cosine is plenty for near-duplicate detection: genuine
near-dups share vocabulary heavily.

Usage:
    python src/overlap_check.py
"""

import gc
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
TEST_PATH = "data/processed/test.jsonl"

MAX_FEATURES = 50_000
QUERY_CHUNK = 200  # keeps the dense (chunk x 98k) score block ~80MB
N_VAL_SAMPLE = 1000


def norm(s):
    return unicodedata.normalize("NFC", s).strip()


def best_matches(query_texts, corpus_X, vectorizer):
    """Best cosine similarity + corpus index for each query, chunked."""
    sims = np.zeros(len(query_texts), dtype=np.float32)
    idxs = np.zeros(len(query_texts), dtype=np.int64)
    for start in range(0, len(query_texts), QUERY_CHUNK):
        chunk = query_texts[start:start + QUERY_CHUNK]
        # TF-IDF rows are L2-normalised, so a plain dot product IS cosine.
        block = (vectorizer.transform(chunk) @ corpus_X.T).toarray()
        idxs[start:start + len(chunk)] = block.argmax(axis=1)
        sims[start:start + len(chunk)] = block.max(axis=1)
        del block
        gc.collect()
        print(f"    ...{min(start + QUERY_CHUNK, len(query_texts))}/{len(query_texts)}", flush=True)
    return sims, idxs


def report(name, sims):
    print(f"\n{name}", flush=True)
    print(f"  mean best-match similarity: {sims.mean():.4f}")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(sims, p):.4f}")
    for thr in [0.95, 0.90, 0.80, 0.70]:
        n = int((sims >= thr).sum())
        print(f"  >= {thr:.2f} similarity: {n} / {len(sims)} ({100 * n / len(sims):.1f}%)")


def main():
    print("Loading data...", flush=True)
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    train_inputs = [norm(s) for s in train_df["input"]]
    train_outputs = train_df["output"].tolist()
    del train_df
    gc.collect()

    test_df = pd.read_json(TEST_PATH, lines=True)
    test_inputs = [norm(s) for s in test_df["input"]]
    test_ids = test_df["id"].tolist()
    del test_df
    gc.collect()

    print(f"train={len(train_inputs)}  test={len(test_inputs)}", flush=True)

    # --- 1. exact duplicates (cheap, run first) ---
    train_set = set(train_inputs)
    n_exact = sum(1 for t in test_inputs if t in train_set)
    print(f"\nEXACT test-input matches in train: {n_exact} / {len(test_inputs)}", flush=True)
    n_dup_train = len(train_inputs) - len(train_set)
    print(f"Duplicate inputs WITHIN train: {n_dup_train} / {len(train_inputs)}", flush=True)
    del train_set
    gc.collect()

    # --- 2. fit TF-IDF on train inputs ---
    print("\nFitting TF-IDF over train inputs...", flush=True)
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, min_df=2, sublinear_tf=True, dtype=np.float32,
    )
    corpus_X = vectorizer.fit_transform(train_inputs)
    print(f"  matrix: {corpus_X.shape}, nnz={corpus_X.nnz:,}", flush=True)

    # --- 3. test -> train near-duplicates (the headline number) ---
    print("\nComputing test -> train nearest neighbours...", flush=True)
    sims, idxs = best_matches(test_inputs, corpus_X, vectorizer)
    report("TEST input -> nearest TRAIN input", sims)

    # --- 4. val -> train leakage sanity check (val was carved out of train) ---
    print("\nComputing val -> train nearest neighbours (leakage check)...", flush=True)
    val_df = pd.read_json(VAL_PATH, lines=True).sample(n=N_VAL_SAMPLE, random_state=42)
    val_inputs = [norm(s) for s in val_df["input"]]
    del val_df
    gc.collect()
    val_sims, _ = best_matches(val_inputs, corpus_X, vectorizer)
    report("VAL input -> nearest TRAIN input", val_sims)

    # --- 5. eyeball the closest test/train pairs ---
    order = np.argsort(-sims)[:5]
    print("\n" + "=" * 70)
    print("TOP 5 most-similar test/train input pairs")
    print("=" * 70)
    for rank, i in enumerate(order, 1):
        print(f"\n[{rank}] similarity={sims[i]:.4f}")
        print(f"  TEST : {test_inputs[i][:200]}")
        print(f"  TRAIN: {train_inputs[idxs[i]][:200]}")
        print(f"  TRAIN RESPONSE: {train_outputs[idxs[i]][:200]}")

    pd.DataFrame({
        "test_id": test_ids,
        "test_input": test_inputs,
        "best_train_sim": sims,
        "best_train_input": [train_inputs[i] for i in idxs],
        "best_train_output": [train_outputs[i] for i in idxs],
    }).to_csv("experiments/test_train_overlap.csv", index=False)
    print("\nSaved per-test-row detail to experiments/test_train_overlap.csv", flush=True)


if __name__ == "__main__":
    main()
