"""
Local eval harness mirroring the Phase 1 leaderboard metric:

    Score = 0.5 * BERTScore_F1 + 0.3 * TokenF1 + 0.2 * ROUGE-L_F1

Token-F1 and ROUGE-L are reported under TWO tokenizations, because the
rulebook (§4) fixes the weights but never specifies a tokenizer or a
BERTScore backbone, and the choice moves the absolute score enormously:

  * `regex`      — the original \\w+ tokenizer. NOTE: Python's \\w excludes
                   Unicode categories Mn/Mc, which is where every Bengali
                   vowel sign and hasant lives, so it SHATTERS Bengali words:
                   "হেলো" -> ['হ', 'ল']. It is a grapheme-fragment F1, not a
                   word F1, and runs ~2.4x inflated vs whitespace.
  * `whitespace` — plain .split(), i.e. real Bengali words.

Kept side by side rather than "fixed" because every plausible implementation
is monotone increasing in the same underlying quantity (lexical + semantic
overlap at roughly reference length), so relative deltas between our own
candidate systems are preserved either way. `composite` stays on `regex` so
numbers remain comparable with MODEL_SELECTION.md; trust the deltas, not the
absolute distance to the leaderboard.

BERTScore uses bert-base-multilingual-cased. bert_score's lang2model is a
defaultdict falling through to exactly that for "bn", and no Bengali
rescaling baseline ships with the package, so rescale_with_baseline is not
an option here.

Usage:
    from eval_metrics import evaluate

    result = evaluate(preds, refs)
    print(result["mean_composite"])              # regex tokenization
    print(result["mean_composite_whitespace"])   # whitespace tokenization
"""

import re
import unicodedata
from collections import Counter

from bert_score import score as bertscore_score
from rouge_score import rouge_scorer

DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str):
    """Original \\w+ tokenizer — fragments Bengali words (see module docstring)."""
    return _WORD_RE.findall(unicodedata.normalize("NFC", text))


def whitespace_tokenize(text: str):
    """Real Bengali words. Trailing punctuation stays attached, as in a plain split."""
    return unicodedata.normalize("NFC", text).split()


class _TokenizerAdapter:
    """Adapter so rouge_score uses the same tokenizer as token_f1."""

    def __init__(self, fn):
        self.fn = fn

    def tokenize(self, text):
        return self.fn(text)


_rouge_regex = rouge_scorer.RougeScorer(
    ["rougeL"], use_stemmer=False, tokenizer=_TokenizerAdapter(simple_tokenize)
)
_rouge_ws = rouge_scorer.RougeScorer(
    ["rougeL"], use_stemmer=False, tokenizer=_TokenizerAdapter(whitespace_tokenize)
)


def token_f1(pred: str, ref: str, tokenize=simple_tokenize) -> float:
    pred_tokens, ref_tokens = tokenize(pred), tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, ref: str, scorer=_rouge_regex) -> float:
    return scorer.score(ref, pred)["rougeL"].fmeasure


def evaluate(preds, refs, bertscore_model=DEFAULT_BERTSCORE_MODEL, batch_size=32, verbose=False):
    """
    preds, refs: lists of strings, same length, index-aligned.
    Returns per-example sub-scores/composite plus means. `composite` uses the
    regex tokenization; `*_whitespace` keys mirror it under plain .split().
    """
    assert len(preds) == len(refs), "preds and refs must be the same length"

    _, _, bert_f1 = bertscore_score(
        preds, refs, model_type=bertscore_model, lang="bn", batch_size=batch_size, verbose=verbose
    )
    bert_f1 = bert_f1.tolist()

    tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
    rouge_f1 = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
    tok_f1_ws = [token_f1(p, r, whitespace_tokenize) for p, r in zip(preds, refs)]
    rouge_f1_ws = [rouge_l_f1(p, r, _rouge_ws) for p, r in zip(preds, refs)]

    composite = [0.5 * b + 0.3 * t + 0.2 * g for b, t, g in zip(bert_f1, tok_f1, rouge_f1)]
    composite_ws = [
        0.5 * b + 0.3 * t + 0.2 * g for b, t, g in zip(bert_f1, tok_f1_ws, rouge_f1_ws)
    ]

    def mean(xs):
        return sum(xs) / len(xs)

    # Length ratio is the single most actionable diagnostic here: Token-F1 and
    # ROUGE-L are F1 measures, so under-generation caps recall outright.
    len_ratio = mean([len(p) for p in preds]) / max(mean([len(r) for r in refs]), 1)

    return {
        "bertscore_f1": bert_f1,
        "token_f1": tok_f1,
        "rouge_l_f1": rouge_f1,
        "composite": composite,
        "mean_bertscore_f1": mean(bert_f1),
        "mean_token_f1": mean(tok_f1),
        "mean_rouge_l_f1": mean(rouge_f1),
        "mean_composite": mean(composite),
        "mean_token_f1_whitespace": mean(tok_f1_ws),
        "mean_rouge_l_f1_whitespace": mean(rouge_f1_ws),
        "mean_composite_whitespace": mean(composite_ws),
        "mean_pred_chars": mean([len(p) for p in preds]),
        "mean_ref_chars": mean([len(r) for r in refs]),
        "length_ratio": len_ratio,
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
