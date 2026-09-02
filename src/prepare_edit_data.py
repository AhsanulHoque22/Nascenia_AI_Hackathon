"""
Builds retrieval+edit training data: for each row, retrieve the most similar
OTHER row (hybrid word+char TF-IDF cosine on input, excluding self) and
build an EDIT_PROMPT input that presents the retrieved (question, answer)
pair and instructs adapting it to the real question -- but the TARGET is
the real, correct answer for the real question, not the retrieved answer.

This is deliberately different from prepare_rag_data.py's "style reference
only" framing (Day 12, failed its own ablation) and from Day 13's zero-shot
retrieval+edit prompting on an unrelated adapter (also failed: 0.5481
composite, worse than both pure retrieval 0.5830 and plain generation
0.5597). Neither prior attempt actually trained the model to DO the edit
task -- Day 12 trained a different behavior (lean on context for style),
and Day 13 just prompted a model that was never taught this task at all.
This is the first genuine test of whether the model CAN learn retrieval+edit
if actually supervised on it.

Usage: python src/prepare_edit_data.py
Output: data/processed/train_edit.jsonl
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
QUERY_CHUNK = 200
N_TRAIN_ROWS = 7000
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


def normalize_rows(m):
    mx = m.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return m / mx


def main():
    full = pd.read_json("data/processed/train_curated.jsonl", lines=True)
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    subset = full.iloc[:N_TRAIN_ROWS].reset_index(drop=True)
    print(f"curated pool: {len(full)} rows, training subset: {len(subset)} rows")

    word_vec = TfidfVectorizer(max_features=50000, dtype=np.float32)
    word_pool = word_vec.fit_transform(full["input"].tolist())
    word_subset = word_vec.transform(subset["input"].tolist())

    char_vec = TfidfVectorizer(max_features=50000, analyzer="char_wb",
                                ngram_range=(3, 5), dtype=np.float32)
    char_pool = char_vec.fit_transform(full["input"].tolist())
    char_subset = char_vec.transform(subset["input"].tolist())

    out_rows = []
    for start in range(0, len(subset), QUERY_CHUNK):
        end = start + QUERY_CHUNK
        word_block = (word_subset[start:end] @ word_pool.T).toarray()
        char_block = (char_subset[start:end] @ char_pool.T).toarray()
        block = normalize_rows(word_block) + normalize_rows(char_block)
        for j in range(word_block.shape[0]):
            block[j, start + j] = -1.0  # never retrieve self
        best_idx = block.argmax(axis=1)

        for j, row in enumerate(subset.iloc[start:end].itertuples()):
            ex_input = full.iloc[best_idx[j]]["input"][:RETRIEVED_INPUT_CHARS]
            ex_output = full.iloc[best_idx[j]]["output"][:RETRIEVED_CONTEXT_CHARS]
            enriched_input = EDIT_PROMPT.format(
                ex_input=ex_input, ex_output=ex_output, real_input=row.input
            )
            out_rows.append({
                "id": getattr(row, "id", start + j),
                "input": enriched_input,
                "output": row.output,  # the REAL correct answer, not the retrieved one
            })
        if start % (QUERY_CHUNK * 10) == 0:
            print(f"  retrieved {end}/{len(subset)}", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_df.to_json("data/processed/train_edit.jsonl", orient="records", lines=True, force_ascii=False)

    mean_len = out_df["input"].str.len().mean()
    print(f"wrote data/processed/train_edit.jsonl: {len(out_df)} rows")
    print(f"mean enriched input length: {mean_len:.0f} chars "
          f"(original mean was {subset['input'].str.len().mean():.0f} chars)")
    print("\n--- example enriched input ---")
    print(out_rows[0]["input"][:800])
    print("\n--- example target output ---")
    print(out_rows[0]["output"][:300])


if __name__ == "__main__":
    main()
