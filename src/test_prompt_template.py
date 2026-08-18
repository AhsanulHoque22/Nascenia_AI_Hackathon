"""
Self-check for the SFT example builder — the logic most able to fail silently.

If prompt masking breaks, training still "succeeds" and the loss curve still
falls; the model just learns the wrong objective. If the length guard breaks,
we retrain on decapitated questions. Neither shows up in a metrics.json.

    python src/test_prompt_template.py
"""

from transformers import AutoTokenizer

from prompt_template import build_example, build_inference_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # cached locally; template shape matches Qwen3


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    q, a = "আমার মাথা ব্যথা করছে।", "হেলো, পর্যাপ্ত বিশ্রাম নিন এবং পানি পান করুন।"

    ex = build_example(tok, q, a, max_seq_len=512, min_prompt_tokens=8)

    assert ex["keep"], "short example should be kept"
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert ex["n_tokens"] == len(ex["input_ids"])
    assert all(m == 1 for m in ex["attention_mask"]), "unpadded example is fully attended"

    # The prompt span must be masked and the response span must not be.
    n_masked = sum(1 for l in ex["labels"] if l == -100)
    assert n_masked > 0, "prompt span is not masked"
    assert n_masked < len(ex["labels"]), "everything is masked — nothing to learn from"

    # Unmasked labels must decode back to exactly the target response.
    kept = [l for l in ex["labels"] if l != -100]
    decoded = tok.decode(kept, skip_special_tokens=True)
    assert decoded.strip() == a, f"response not recoverable from labels: {decoded!r}"

    # Labels must align with input_ids wherever they are not masked, or the
    # model is trained to predict tokens it was never shown.
    for tid, lab in zip(ex["input_ids"], ex["labels"]):
        assert lab in (-100, tid), "label/input mismatch on an unmasked position"

    # The response is the target and must survive; the prompt is what gives.
    long_q = "রোগীর ইতিহাস। " * 400
    ex2 = build_example(tok, long_q, a, max_seq_len=256, min_prompt_tokens=8)
    assert ex2["keep"]
    assert len(ex2["input_ids"]) <= 256, "max_seq_len not respected"
    assert tok.decode([l for l in ex2["labels"] if l != -100],
                      skip_special_tokens=True).strip() == a, "response was truncated"

    # A response too long to leave room for the question must be dropped, not
    # trained on with an erased prompt (this is the 28.9%-of-corpus bug).
    long_a = "চিকিৎসা পরামর্শ। " * 300
    ex3 = build_example(tok, q, long_a, max_seq_len=256, min_prompt_tokens=64)
    assert not ex3["keep"], "over-length response should be flagged for dropping"

    # Inference prompt must not leak the answer and must invite a completion.
    p = build_inference_prompt(tok, q)
    assert a not in p and q in p
    assert p.rstrip().endswith("assistant") or "assistant" in p[-80:], \
        "prompt does not end on an assistant turn"
    assert "<think>" not in p, "thinking block leaked into the prompt"

    print(f"all checks passed "
          f"({len(ex['input_ids'])} tokens, {n_masked} masked, {len(kept)} supervised)")


if __name__ == "__main__":
    main()
