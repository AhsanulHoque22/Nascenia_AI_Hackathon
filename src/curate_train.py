"""
Curate the training set before the one-shot fine-tune.

The leaderboard metric is similarity-to-reference, so "quality" here means
"looks like the reference distribution", NOT "is a better medical answer".
That flips two instincts:

  * Boilerplate is kept, deliberately. 76% of reference responses open with
    "হেলো" and 48% carry the source-platform branding, none of which appears
    in any input. A base model never emits it; a fine-tuned one does, and
    every occurrence is a guaranteed token match. Stripping it would throw
    away score.
  * Rows are dropped for being TRUNCATED or degenerate, not for being
    unhelpful. A response cut mid-word teaches the model not to emit EOS,
    which is the exact failure that wrecks output length — and output length
    is the single largest lever on this metric.

Filters (applied in order), with row counts from the 97,853-row train split:

    F1  response < 150 chars                     -2,313   stubs
    F2  response > 1600 chars                      -554   p99 tail, off-target length
    F3  latin fraction > 0.05                      -585   broken/English-heavy rows
    F4  response lacks terminal punctuation      -6,429   TRUNCATED mid-sentence
    F5  duplicate input (keep first)               -163
    F6  duplicate output (keep first)            -1,181
    F7  input > ~850 chars (~900 tokens, p95)   -4,957   prompt-cost outliers
                                                 -------
                                                  81,671 survivors (83.5%)

F4 is the highest-value and least obvious: thousands of responses end in a
bare letter, a vowel sign, or a chopped "http…" — scrape damage.

Length-based token filtering is deliberately NOT done here; it depends on
max_seq_len, so train.py handles it via `min_prompt_tokens`.

Usage:
    python src/curate_train.py
    python src/curate_train.py --in data/processed/train.jsonl --out data/processed/train_curated.jsonl
"""

import argparse
import re

import pandas as pd

# Bengali averages ~1.06 Qwen3 tokens per character, so the p95 input cut of
# ~900 tokens lands at ~850 characters. Using chars keeps this script
# tokenizer-free and fast.
MAX_INPUT_CHARS = 850
MIN_OUTPUT_CHARS = 150
MAX_OUTPUT_CHARS = 1600
MAX_LATIN_FRAC = 0.05

# Bengali danda/double-danda plus western terminators. A response ending in
# anything else is almost always a mid-word truncation.
TERMINAL_PUNCT = tuple("।.?!॥)\"'")

_LATIN_RE = re.compile(r"[A-Za-z]")


def latin_frac(s: str) -> float:
    return len(_LATIN_RE.findall(s)) / max(len(s), 1)


def curate(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    steps = []

    def drop(mask, label):
        nonlocal df
        before = len(df)
        df = df[~mask].copy()
        steps.append((label, before - len(df), len(df)))

    out = df["output"].str.strip()
    drop(out.str.len() < MIN_OUTPUT_CHARS, f"F1 response < {MIN_OUTPUT_CHARS} chars")

    out = df["output"].str.strip()
    drop(out.str.len() > MAX_OUTPUT_CHARS, f"F2 response > {MAX_OUTPUT_CHARS} chars")

    drop(df["output"].map(latin_frac) > MAX_LATIN_FRAC, f"F3 latin frac > {MAX_LATIN_FRAC}")

    drop(~df["output"].str.strip().str.endswith(TERMINAL_PUNCT), "F4 no terminal punctuation")

    drop(df["input"].str.strip().duplicated(), "F5 duplicate input")
    drop(df["output"].str.strip().duplicated(), "F6 duplicate output")

    drop(df["input"].str.len() > MAX_INPUT_CHARS, f"F7 input > {MAX_INPUT_CHARS} chars")

    print(f"{'filter':<36}{'dropped':>9}{'remaining':>11}")
    print("-" * 56)
    print(f"{'(start)':<36}{'':>9}{n0:>11,}")
    for label, dropped, remaining in steps:
        print(f"{label:<36}{dropped:>9,}{remaining:>11,}")
    print("-" * 56)
    print(f"kept {len(df):,} / {n0:,} ({100*len(df)/n0:.1f}%)")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/train.jsonl")
    ap.add_argument("--out", dest="out", default="data/processed/train_curated.jsonl")
    args = ap.parse_args()

    df = pd.read_json(args.inp, lines=True)
    curated = curate(df)

    print(f"\nresponse chars: mean {curated['output'].str.len().mean():.0f}  "
          f"median {curated['output'].str.len().median():.0f}")
    print(f"input chars:    mean {curated['input'].str.len().mean():.0f}")
    # Sanity check that the formulaic opening survived — if this collapses,
    # a filter is eating the boilerplate that earns Token-F1 matches.
    pct_helo = 100 * curated["output"].str.strip().str.startswith("হেলো").mean()
    print(f"responses opening with 'হেলো': {pct_helo:.1f}% (expect ~76%)")

    curated.to_json(args.out, orient="records", lines=True, force_ascii=False)
    print(f"\nWrote {len(curated):,} rows to {args.out}")


if __name__ == "__main__":
    main()
