"""
Local eval harness mirroring the Phase 1 leaderboard metric:

    Score = 0.5 * BERTScore_F1 + 0.3 * TokenF1 + 0.2 * ROUGE-L_F1

Token-F1 and ROUGE-L use a shared Unicode-aware word tokenizer (regex \\w+,
which matches Bengali letters under Python 3's default UNICODE mode) so
both sub-metrics tokenize consistently. BERTScore uses a multilingual BERT
backbone (bert-base-multilingual-cased by default — CPU-friendly for local
iteration; swap to a larger XLM-R checkpoint on Kaggle's GPU if it
correlates better with the real leaderboard once submission #1 lands).

Usage:
    from eval_metrics import evaluate

    result = evaluate(preds, refs)
    print(result["mean_composite"])
"""

import re
import unicodedata
from collections import Counter

from bert_score import score as bertscore_score
from rouge_score import rouge_scorer

DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str):
    text = unicodedata.normalize("NFC", text)
    return _WORD_RE.findall(text)


class _SharedTokenizer:
    """Adapter so rouge_score uses the same tokenizer as token_f1."""

    def tokenize(self, text):
        return simple_tokenize(text)


_rouge_scorer = rouge_scorer.RougeScorer(
    ["rougeL"], use_stemmer=False, tokenizer=_SharedTokenizer()
)


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = simple_tokenize(pred)
    ref_tokens = simple_tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, ref: str) -> float:
    return _rouge_scorer.score(ref, pred)["rougeL"].fmeasure


def evaluate(preds, refs, bertscore_model=DEFAULT_BERTSCORE_MODEL, batch_size=32, verbose=False):
    """
    preds, refs: lists of strings, same length, index-aligned.
    Returns per-example sub-scores/composite plus the mean composite
    (== what the Phase 1 leaderboard reports).
    """
    assert len(preds) == len(refs), "preds and refs must be the same length"

    _, _, bert_f1 = bertscore_score(
        preds, refs, model_type=bertscore_model, lang="bn", batch_size=batch_size, verbose=verbose
    )
    bert_f1 = bert_f1.tolist()

    tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
    rouge_f1 = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]

    composite = [
        0.5 * b + 0.3 * t + 0.2 * rg for b, t, rg in zip(bert_f1, tok_f1, rouge_f1)
    ]

    return {
        "bertscore_f1": bert_f1,
        "token_f1": tok_f1,
        "rouge_l_f1": rouge_f1,
        "composite": composite,
        "mean_bertscore_f1": sum(bert_f1) / len(bert_f1),
        "mean_token_f1": sum(tok_f1) / len(tok_f1),
        "mean_rouge_l_f1": sum(rouge_f1) / len(rouge_f1),
        "mean_composite": sum(composite) / len(composite),
    }


if __name__ == "__main__":
    import random

    import pandas as pd

    random.seed(42)
    N = 50
    val = pd.read_json("data/processed/val.jsonl", lines=True).sample(
        n=N, random_state=42
    ).reset_index(drop=True)
    refs = val["output"].tolist()

    print(f"=== Sanity check on {N} val examples ===\n")

    print("--- ref vs ref (should be ~1.0 on all sub-metrics) ---")
    result_self = evaluate(refs, refs, verbose=False)
    print(f"  mean BERTScore F1: {result_self['mean_bertscore_f1']:.4f}")
    print(f"  mean Token F1:     {result_self['mean_token_f1']:.4f}")
    print(f"  mean ROUGE-L F1:   {result_self['mean_rouge_l_f1']:.4f}")
    print(f"  mean composite:    {result_self['mean_composite']:.4f}\n")

    print("--- ref vs shuffled ref (mismatched pairs, should be low) ---")
    shuffled_refs = refs[1:] + refs[:1]  # rotate by 1, guarantees no self-pairing
    result_random = evaluate(refs, shuffled_refs, verbose=False)
    print(f"  mean BERTScore F1: {result_random['mean_bertscore_f1']:.4f}")
    print(f"  mean Token F1:     {result_random['mean_token_f1']:.4f}")
    print(f"  mean ROUGE-L F1:   {result_random['mean_rouge_l_f1']:.4f}")
    print(f"  mean composite:    {result_random['mean_composite']:.4f}")
