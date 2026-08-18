"""
LoRA/QLoRA fine-tuning entrypoint.

Config-driven (see configs/*.yaml) so every run's hyperparameters are
reproducible and diffable — one YAML per experiment. Instruction pairs are
formatted with the locked prompt template in src/prompt_template.py, the
same template used for the zero-shot baselines.

Rewritten after a research pass (see TRAINING_NOTES.md) that found the
original Day 7 defaults were leaving most of the T4 idle and truncating the
patient question down to ~64 tokens. Headline changes:

  * 4-bit quantization is now OFF by default. NF4 has no hardware support —
    it dequantizes on FP32 CUDA cores (8.1 TFLOPS on a T4) before every
    fp16 tensor-core matmul, costing 20-40% throughput to save VRAM we do
    not need: Qwen3-1.7B in fp16 is ~4.1GB of a 16GB card.
  * gradient_checkpointing OFF by default (it trades ~30% speed for memory).
    When it IS enabled without 4-bit, enable_input_require_grads() must be
    called or PEFT silently receives zero gradients and the loss flatlines
    while the run "succeeds" — handled below.
  * Tokenization happens once, up front, in parallel, instead of re-rendering
    the chat template and tokenizing inside __getitem__ on every step.
  * group_by_length batches similar-length examples so cost tracks actual
    length rather than max_seq_len.
  * DDP-aware: device_map binds to LOCAL_RANK, so `torchrun --nproc_per_node 2`
    uses both T4s. (device_map="auto" instead SHARDS one model across both
    GPUs and crashes Trainer with a cuda:0/cuda:1 mismatch.)

Usage:
    python src/train.py --config configs/smoke_test.yaml
    python src/train.py --config configs/qwen3_full.yaml --max_steps 20   # probe
    torchrun --nproc_per_node 2 src/train.py --config configs/qwen3_full.yaml
"""

import argparse
import json
import os
import random
import time

# Tokenizing in worker processes warns (and can deadlock) unless this is off.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

# Kaggle hands out 2x T4. If we are NOT under torchrun, HF Trainer sees both
# and silently wraps the model in DataParallel, which blows up ("Caught
# RuntimeError in replica 0 on device 0") and can leave the context in an
# illegal-memory-access state. Single-process runs therefore pin to one GPU;
# `torchrun --nproc_per_node 2` sets LOCAL_RANK and gets both via real DDP.
if LOCAL_RANK < 0:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Measured peak is 13.8GB of a T4's 14.56GB — 95% utilization for ~11 hours.
# Allocator fragmentation alone could OOM the run late; expandable segments
# let the allocator grow blocks in place instead of stranding them.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from prompt_template import build_example

IS_MAIN = LOCAL_RANK in (-1, 0)


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AbortOnNaN(TrainerCallback):
    """
    fp16 + LoRA on Turing can diverge (no bf16 on sm_75). A NaN loss six
    hours into a one-shot run should stop it, not quietly waste the session.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({loss}) at step {state.global_step}. "
                "fp16 instability — lower learning_rate (2e-4 -> 1e-4) and rerun."
            )


class PadCollator:
    """Dynamic per-batch padding (input_ids/attention_mask/labels)."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(path, tokenizer, cfg, max_samples, seed, split_name):
    """Tokenize once, up front, in parallel; drop over-length examples."""
    t_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_seq_len = t_cfg["max_seq_len"]
    min_prompt_tokens = data_cfg.get("min_prompt_tokens", 0)

    df = pd.read_json(path, lines=True)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    ds = Dataset.from_pandas(df[["input", "output"]], preserve_index=False)

    def _tok(row):
        ex = build_example(
            tokenizer, row["input"], row["output"], max_seq_len, min_prompt_tokens
        )
        # Trainer's LengthGroupedSampler reads this column by name.
        ex["length"] = ex.pop("n_tokens")
        return ex

    n_proc = data_cfg.get("tokenize_num_proc", 4)
    ds = ds.map(
        _tok, remove_columns=["input", "output"], num_proc=n_proc,
        desc=f"tokenizing {split_name}",
    )
    before = len(ds)
    ds = ds.filter(lambda r: r["keep"], num_proc=n_proc).remove_columns(["keep"])
    dropped = before - len(ds)
    log(f"{split_name}: {len(ds)} examples "
        f"({dropped} dropped — response left < {min_prompt_tokens} tokens for the question)")
    if len(ds):
        lens = np.array(ds["length"])
        log(f"  token length: mean {lens.mean():.0f}  p50 {np.percentile(lens,50):.0f}  "
            f"p95 {np.percentile(lens,95):.0f}  max {lens.max()}")
    return ds


def load_model_and_tokenizer(cfg):
    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    want_4bit = cfg.get("quantization", {}).get("load_in_4bit", False)
    use_4bit = want_4bit and torch.cuda.is_available()
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif want_4bit:
        log("WARNING: load_in_4bit=true but no CUDA device — loading full precision.")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Bind the whole model to THIS rank's GPU. "auto" would shard one model
    # across both T4s and break Trainer with a device mismatch.
    device_map = None
    if torch.cuda.is_available():
        device_map = {"": LOCAL_RANK if LOCAL_RANK >= 0 else 0}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing; unused in training

    grad_ckpt = cfg["training"].get("gradient_checkpointing", False)
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt
        )
    elif grad_ckpt:
        # Without the kbit path, nothing marks the embedding output as
        # requiring grad, so checkpointing detaches the graph and every LoRA
        # gradient arrives as zero — training appears to run and learns nothing.
        model.enable_input_require_grads()

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg.get("use_rslora", False),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Keep the trainable adapter in fp32 while the frozen base stays fp16 —
    # standard mixed precision, and materially more stable than fp16 params
    # on a card with no bf16 fallback.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if IS_MAIN:
        model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run_id", default=None,
        help="experiments/<run_id> dir name; defaults to the config filename",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="override for short probe runs (throughput/sanity checks)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    run_id = args.run_id or os.path.splitext(os.path.basename(args.config))[0]
    run_dir = os.path.join("experiments", run_id)
    os.makedirs(run_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)

    data_cfg = cfg["data"]
    t_cfg = cfg["training"]
    train_ds = build_dataset(
        data_cfg["train_path"], tokenizer, cfg,
        data_cfg.get("max_train_samples"), seed, "train",
    )
    eval_ds = build_dataset(
        data_cfg["val_path"], tokenizer, cfg,
        data_cfg.get("max_eval_samples"), seed, "val",
    )

    collator = PadCollator(tokenizer.pad_token_id)

    # transformers 5.x dropped `warmup_ratio`, so derive warmup_steps from the
    # real schedule length (this is why we compute it rather than hardcode:
    # a fixed step count means a different warmup fraction per dataset size).
    bs = t_cfg["batch_size"] * max(t_cfg.get("grad_accum_steps", 1), 1)
    world = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
    total_steps = args.max_steps or max(
        1, int(len(train_ds) / (bs * world) * t_cfg["epochs"])
    )
    warmup_steps = t_cfg.get(
        "warmup_steps", max(1, int(total_steps * t_cfg.get("warmup_ratio", 0.03)))
    )
    log(f"schedule: ~{total_steps} optimizer steps, {warmup_steps} warmup")

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=t_cfg["epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=t_cfg["batch_size"],
        per_device_eval_batch_size=t_cfg.get("eval_batch_size", t_cfg["batch_size"]),
        gradient_accumulation_steps=t_cfg.get("grad_accum_steps", 1),
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        max_grad_norm=t_cfg.get("max_grad_norm", 1.0),
        logging_steps=t_cfg.get("logging_steps", 20),
        eval_strategy=t_cfg.get("eval_strategy", "steps"),
        eval_steps=t_cfg.get("eval_steps", 500),
        save_strategy=t_cfg.get("save_strategy", "steps"),
        save_steps=t_cfg.get("save_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # transformers 5.x: group_by_length=True became this enum. Batches
        # similar-length examples so step cost tracks actual length instead of
        # max_seq_len; "length" is precomputed at tokenization time.
        train_sampling_strategy=(
            "group_by_length" if t_cfg.get("group_by_length", True) else "random"
        ),
        length_column_name="length",
        remove_unused_columns=False,  # our collator picks its own keys; keeps "length" for the sampler
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        # Qwen3's 151,936-token vocab makes the logits tensor, not the model,
        # the memory hog: at batch 8 x 1152 tokens that is ~2.8GB of fp16
        # logits plus an fp32 upcast and a gradient. Liger's fused linear
        # cross-entropy chunks it and never materializes the full tensor,
        # which is what makes a larger batch fit at all.
        use_liger_kernel=t_cfg.get("use_liger_kernel", False),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[AbortOnNaN()],
    )

    t0 = time.time()
    train_result = trainer.train()
    train_time_s = time.time() - t0

    if IS_MAIN:
        adapter_dir = os.path.join(run_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        metrics = {
            "run_id": run_id,
            "config_path": args.config,
            "config": cfg,
            "train_time_s": round(train_time_s, 1),
            "train_loss": train_result.training_loss,
            "n_train_examples": len(train_ds),
            "log_history": trainer.state.log_history,
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        steps = max(train_result.global_step, 1)
        log(f"\nDone in {train_time_s/60:.1f} min "
            f"({train_time_s/steps:.2f}s/step over {steps} steps)")
        log(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
