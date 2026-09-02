"""
Locked instruction prompt template — used identically for zero-shot baseline
(Day 5/6), LoRA fine-tuning (Day 7+), and must stay identical at inference
and in the Phase 2 write-up.

Format: Qwen chat template (system + user turns), target completion is the
doctor's response followed by the tokenizer's EOS token. Loss is computed
only on the response span; the prompt span is masked with -100.

    <|im_start|>system
    {SYSTEM_PROMPT}<|im_end|>
    <|im_start|>user
    {patient_input}<|im_end|>
    <|im_start|>assistant
    {doctor_output}<|im_end|>

SYSTEM_PROMPT is unchanged from kaggle_kernels/model_shortlist/script.py
(Day 6 zero-shot eval) so baseline and fine-tuned numbers stay comparable.

enable_thinking=False is passed to apply_chat_template for Qwen3's hybrid
think/no-think template (Day 9 model swap to Qwen3-1.7B) — this task needs a
direct response, not a <think>...</think> reasoning block, and it wastes
generation budget otherwise. Qwen2.5's template silently ignores unknown
kwargs, so this is safe for both model families.
"""

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def build_messages(patient_input: str, doctor_output: str | None = None):
    """Chat-template messages. Pass doctor_output=None at inference time."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input},
    ]
    if doctor_output is not None:
        messages.append({"role": "assistant", "content": doctor_output})
    return messages


def build_example(tokenizer, patient_input: str, doctor_output: str, max_seq_len: int,
                  min_prompt_tokens: int = 0):
    """
    Tokenize one (patient_input, doctor_output) pair for SFT.

    The response is never truncated (it's the training target); if the
    prompt + response exceeds max_seq_len, the prompt is truncated from the
    left (oldest tokens first) to make room.

    That left-truncation is far more damaging than it looks. Measured on this
    corpus (Qwen3 tokenizer): responses average 674 tokens and inputs 467, so
    at max_seq_len=768 the prompt is squeezed down to ~64 tokens — the model
    trains on the tail of the question and the whole answer, and only 7.3% of
    examples fit untruncated. `min_prompt_tokens` guards against that: an
    example whose prompt would be cut below it is flagged `keep=False` so the
    caller can drop it instead of training on a decapitated question.

    Returns input_ids / attention_mask / labels (prompt span masked to -100),
    unpadded — padding happens in the collator — plus `keep` and `n_tokens`
    for filtering and length-grouping.
    """
    prompt_text = tokenizer.apply_chat_template(
        build_messages(patient_input), tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    response_text = doctor_output + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    keep = True
    max_prompt_len = max_seq_len - len(response_ids)
    if max_prompt_len < min_prompt_tokens:
        # Response is so long it leaves no usable room for the question.
        keep = False
        max_prompt_len = max(max_prompt_len, 1)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    response_ids = response_ids[:max_seq_len - len(prompt_ids)]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
        "keep": keep, "n_tokens": len(input_ids),
    }


def build_inference_prompt(tokenizer, patient_input: str) -> str:
    """Prompt string to feed the model at inference (no response turn)."""
    return tokenizer.apply_chat_template(
        build_messages(patient_input), tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
