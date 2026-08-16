"""
LoRA/QLoRA fine-tuning entrypoint (Day 7).

Config-driven (see configs/*.yaml) so every run's hyperparameters are
reproducible and diffable — one YAML per experiment. Instruction pairs are
formatted with the locked prompt template in src/prompt_template.py, the
same template used for the Day 5/6 zero-shot baseline.

4-bit quantization (QLoRA) only activates when `quantization.load_in_4bit`
is set AND a CUDA device is present — locally (CPU, no GPU) it silently
falls back to full precision so the pipeline is still runnable for
smoke-testing config/data/training-loop wiring before spending real Kaggle
GPU budget on it.

Usage:
    python src/train.py --config configs/smoke_test.yaml
    python src/train.py --config configs/qwen1p5b_lora_base.yaml --run_id day9_full_run
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from prompt_template import build_example


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, max_samples=None, seed=42):
        df = pd.read_json(path, lines=True)
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        self.rows = df[["input", "output"]].to_dict("records")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return build_example(self.tokenizer, row["input"], row["output"], self.max_seq_len)


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
        print(
            "WARNING: quantization.load_in_4bit=true but no CUDA device is "
            "available — loading in full precision instead. Expected on "
            "local CPU dev; run on the Kaggle T4 runtime for real QLoRA."
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.config.use_cache = False  # incompatible with gradient checkpointing; unused during training anyway

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run_id", default=None,
        help="experiments/<run_id> dir name; defaults to the config filename",
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
    train_ds = InstructionDataset(
        data_cfg["train_path"], tokenizer, t_cfg["max_seq_len"],
        max_samples=data_cfg.get("max_train_samples"), seed=seed,
    )
    eval_ds = InstructionDataset(
        data_cfg["val_path"], tokenizer, t_cfg["max_seq_len"],
        max_samples=data_cfg.get("max_eval_samples"), seed=seed,
    )
    print(f"train examples: {len(train_ds)} | eval examples: {len(eval_ds)}")

    collator = PadCollator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=t_cfg["epochs"],
        per_device_train_batch_size=t_cfg["batch_size"],
        per_device_eval_batch_size=t_cfg.get("eval_batch_size", t_cfg["batch_size"]),
        gradient_accumulation_steps=t_cfg.get("grad_accum_steps", 1),
        learning_rate=t_cfg["learning_rate"],
        warmup_steps=t_cfg.get("warmup_steps", 0),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=t_cfg.get("save_total_limit", 2),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    train_time_s = time.time() - t0

    adapter_dir = os.path.join(run_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "run_id": run_id,
        "config_path": args.config,
        "config": cfg,
        "train_time_s": round(train_time_s, 1),
        "train_loss": train_result.training_loss,
        "log_history": trainer.state.log_history,
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Adapter saved to {adapter_dir}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
