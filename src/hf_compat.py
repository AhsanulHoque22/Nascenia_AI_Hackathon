"""
TrainingArguments compatibility across transformers 4.x and 5.x.

We develop against transformers 5.15 locally but Kaggle's image ships 4.x,
and the two disagree on argument names that matter here:

    4.x                       5.x
    group_by_length=True      train_sampling_strategy="group_by_length"
    warmup_ratio=0.03         (removed — compute warmup_steps yourself)

Passing the wrong one raises TypeError at construction. That is a fast
failure, but on a one-shot 8-hour run "fast" still costs a session slot, and
the failure happens on whichever machine you did NOT develop on. So rather
than pin a name, ask the installed class what it accepts.
"""

import inspect

from transformers import TrainingArguments


def supported_args():
    return set(inspect.signature(TrainingArguments.__init__).parameters)


def length_grouping(enabled=True):
    """Kwargs that batch similar-length examples together, on either version."""
    if not enabled:
        return {}
    sig = supported_args()
    if "train_sampling_strategy" in sig:        # transformers 5.x
        return {"train_sampling_strategy": "group_by_length"}
    if "group_by_length" in sig:                # transformers 4.x
        return {"group_by_length": True}
    return {}


def warmup(total_steps, ratio=0.03, explicit_steps=None):
    """warmup_steps on both versions — 5.x dropped warmup_ratio entirely.

    Computed from the real schedule length rather than hardcoded, so the
    warmup fraction stays correct when the dataset size changes.
    """
    steps = explicit_steps if explicit_steps else max(1, int(total_steps * ratio))
    return {"warmup_steps": steps}


def filter_kwargs(kwargs):
    """Drop kwargs this transformers version does not know about.

    Used for genuinely optional knobs (save_only_model, use_liger_kernel,
    gradient_checkpointing_kwargs). Anything load-bearing should NOT go
    through here — silently dropping it would change training semantics.
    """
    sig = supported_args()
    kept = {k: v for k, v in kwargs.items() if k in sig}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[hf_compat] transformers {_version()} does not support: "
              f"{', '.join(dropped)} — dropped", flush=True)
    return kept


def _version():
    import transformers
    return transformers.__version__
