"""
Builds retrieval-augmented training data: for each row, retrieve the most
similar OTHER row (TF-IDF cosine on input, excluding self) and prepend a
truncated version of its real answer as an in-context reference example.
Teaches the model to lean on retrieved context (RAFT-style) rather than
generate purely from scratch -- targets the metric's heavy weight on
token/n-gram overlap with the hidden reference (0.3 TokenF1 + 0.2 ROUGE-L),
which favors output that closely mirrors real reference wording.

Sized for ONE ~8.5h Kaggle session: full-dataset 1-epoch training is ~34.5
GPU-hours (per kaggle_kernels/day11_train_teammate/script.py), so this
samples a subset and keeps retrieved-context short enough that the longer
per-example sequences still fit the time budget.

Usage: python src/prepare_rag_data.py
Output: data/processed/train_rag.jsonl
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
QUERY_CHUNK = 200   # keeps the dense (chunk x 81771) score block small
                    # (~62MB at float32) -- computing the full dense
                    # similarity matrix at once OOM-killed this same laptop
                    # doing the same thing in overlap_check.py earlier
N_TRAIN_ROWS = 7000        # subset size -- measured mean enriched length is
                            # ~3.2x the original (see module docstring for
                            # the GPU-hour math this is sized against)
RETRIEVED_INPUT_CHARS = 200     # truncate retrieved question -- reference
                                 # is for STYLE, not the exact original
                                 # question, so this doesn't need to be long
RETRIEVED_CONTEXT_CHARS = 300   # truncate retrieved answer to control
                                 # per-example token growth
REFERENCE_LABEL = (
    "নিচে একটি পূর্ববর্তী অনুরূপ প্রশ্ন ও তার উত্তরের উদাহরণ দেওয়া হলো, "
    "শুধুমাত্র ভাষা ও গঠনশৈলীর নির্দেশনা হিসেবে ব্যবহার করুন:\n"
    "উদাহরণ প্রশ্ন: {ex_input}\n"
    "উদাহরণ উত্তর: {ex_output}\n\n"
    "এখন নিচের প্রকৃত রোগীর প্রশ্নের উত্তর দিন:\n{real_input}"
)


def main():
    full = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    subset = full.iloc[:N_TRAIN_ROWS].reset_index(drop=True)

    print(f"curated pool: {len(full)} rows, training subset: {len(subset)} rows")

    # Retrieval pool is the FULL curated set (not just the subset) so there's
    # a wider variety of reference examples to draw from -- more realistic
    # stand-in for retrieval against the full pool at inference time too.
    vec = TfidfVectorizer(max_features=50000, dtype=np.float32)
    pool_matrix = vec.fit_transform(full["input"].tolist())
    subset_matrix = vec.transform(subset["input"].tolist())

    out_rows = []
    for start in range(0, len(subset), QUERY_CHUNK):
        chunk = subset.iloc[start:start + QUERY_CHUNK]
        block = (subset_matrix[start:start + len(chunk)] @ pool_matrix.T).toarray()
        # Exclude self -- subset is full's first N_TRAIN_ROWS rows in the
        # same order (both reset_index(drop=True) from the same shuffle),
        # so subset position i IS full position i. A row must never retrieve
        # itself as its own reference, or the model would learn to trivially
        # copy rather than the retrieval-conditioning BEHAVIOR that has to
        # transfer to genuinely different rows at test time.
        for j in range(len(chunk)):
            block[j, start + j] = -1.0
        best_idx = block.argmax(axis=1)

        for j, row in enumerate(chunk.itertuples()):
            ex_input = full.iloc[best_idx[j]]["input"][:RETRIEVED_INPUT_CHARS]
            ex_output = full.iloc[best_idx[j]]["output"][:RETRIEVED_CONTEXT_CHARS]
            enriched_input = REFERENCE_LABEL.format(
                ex_input=ex_input, ex_output=ex_output, real_input=row.input
            )
            out_rows.append({
                "id": getattr(row, "id", start + j),
                "input": enriched_input,
                "output": row.output,
            })
        if start % (QUERY_CHUNK * 10) == 0:
            print(f"  retrieved {start + len(chunk)}/{len(subset)}", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_df.to_json("data/processed/train_rag.jsonl", orient="records", lines=True, force_ascii=False)

    mean_len = out_df["input"].str.len().mean()
    print(f"wrote data/processed/train_rag.jsonl: {len(out_df)} rows")
    print(f"mean enriched input length: {mean_len:.0f} chars "
          f"(original mean was {subset['input'].str.len().mean():.0f} chars)")

    # sanity check: print one example
    print("\n--- example enriched input ---")
    print(out_rows[0]["input"][:800])
    print("--- target output (unchanged) ---")
    print(out_rows[0]["output"][:200])


if __name__ == "__main__":
    main()
