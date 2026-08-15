"""
Day 3 data cleaning, per notebooks/01_eda.ipynb findings:
  1. Drop rows with garbled/spam input (Bengali-char ratio < 0.3).
  2. Drop rows with placeholder/broken output (< 20 chars).
  3. NFC-normalize all input/output text (78% of raw outputs weren't NFC).
  4. Split train/val grouped by `input`, so a repeated prompt (205 in raw
     data) never appears on both sides.

Run from repo root: python src/data_prep.py
"""

import re
import unicodedata

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

BENGALI_RE = re.compile(r"[ঀ-৿]")
MIN_BENGALI_RATIO = 0.3
MIN_OUTPUT_CHARS = 20
VAL_FRACTION = 0.10
SEED = 42


def bengali_ratio(s: str) -> float:
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
    return len(BENGALI_RE.findall(s)) / len(s)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def clean(df: pd.DataFrame, has_output: bool) -> pd.DataFrame:
    df = df.copy()
    df["input"] = df["input"].map(nfc)
    if has_output:
        df["output"] = df["output"].map(nfc)

    before = len(df)
    df = df[df["input"].map(bengali_ratio) >= MIN_BENGALI_RATIO]
    dropped_garbled = before - len(df)

    dropped_placeholder = 0
    if has_output:
        before = len(df)
        df = df[df["output"].str.len() >= MIN_OUTPUT_CHARS]
        dropped_placeholder = before - len(df)

    print(
        f"  dropped {dropped_garbled} garbled-input rows, "
        f"{dropped_placeholder} placeholder-output rows "
        f"-> {len(df)} rows remain"
    )
    return df.reset_index(drop=True)


def grouped_train_val_split(df: pd.DataFrame, val_fraction: float, seed: int):
    unique_inputs = df["input"].drop_duplicates().sample(frac=1.0, random_state=seed)
    n_val_groups = int(len(unique_inputs) * val_fraction)
    val_inputs = set(unique_inputs.iloc[:n_val_groups])

    is_val = df["input"].isin(val_inputs)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def main():
    print("Loading raw data...")
    train_raw = pd.read_csv(f"{RAW_DIR}/train.csv")
    test_raw = pd.read_csv(f"{RAW_DIR}/test.csv")
    print(f"  train.csv: {len(train_raw)} rows, test.csv: {len(test_raw)} rows")

    print("Cleaning train...")
    train_clean = clean(train_raw, has_output=True)

    print("Cleaning test (input-only, no row dropping — every test id must "
          "get a prediction)...")
    test_clean = test_raw.copy()
    test_clean["input"] = test_clean["input"].map(nfc)

    print(f"Splitting train/val (grouped by input, {VAL_FRACTION:.0%} val, seed={SEED})...")
    train_split, val_split = grouped_train_val_split(train_clean, VAL_FRACTION, SEED)
    print(f"  train: {len(train_split)} rows, val: {len(val_split)} rows")

    overlap = set(train_split["input"]) & set(val_split["input"])
    assert not overlap, f"leakage: {len(overlap)} inputs in both train and val"
    print("  leakage check passed: no shared inputs between train and val")

    train_split[["id", "input", "output"]].to_json(
        f"{OUT_DIR}/train.jsonl", orient="records", lines=True, force_ascii=False
    )
    val_split[["id", "input", "output"]].to_json(
        f"{OUT_DIR}/val.jsonl", orient="records", lines=True, force_ascii=False
    )
    test_clean[["id", "input"]].to_json(
        f"{OUT_DIR}/test.jsonl", orient="records", lines=True, force_ascii=False
    )
    print(f"Wrote {OUT_DIR}/train.jsonl, {OUT_DIR}/val.jsonl, {OUT_DIR}/test.jsonl")


if __name__ == "__main__":
    main()
