"""
Day 8 - Small-scale QLoRA pipeline validation on Kaggle T4.

The Day 7 local CPU smoke test (configs/smoke_test.yaml, Qwen2.5-0.5B, 4
examples) already validated data loading / prompt masking / LoRA wrapping /
Trainer loop / checkpoint save on CPU. This kernel validates the same
train.py pipeline logic on the *real* primary candidate (Qwen2.5-1.5B) with
*real* QLoRA (4-bit) on the Kaggle T4 GPU, on a 500-example subset -- so any
GPU/quantization-specific breakage surfaces here, before Day 9 burns a full
epoch over all ~98k training rows.

Not meant to produce a good model. Just: does it run, and are the
generations non-garbage (coherent Bengali, on-topic)?

Kernel has no local src/ import, so train.py + prompt_template.py logic is
inlined below (kept in sync by hand -- same target_modules, same masking,
same locked SYSTEM_PROMPT as src/prompt_template.py).

Data source: /kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)

import json
import random
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
OUT_DIR = "/kaggle/working/day8_run"
SEED = 42
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TRAIN_SAMPLES = 500
MAX_EVAL_SAMPLES = 20
MAX_SEQ_LEN = 768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---- locked prompt template (mirrors src/prompt_template.py) ----
SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def build_messages(patient_input, doctor_output=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input},
    ]
    if doctor_output is not None:
        messages.append({"role": "assistant", "content": doctor_output})
    return messages


def build_example(tokenizer, patient_input, doctor_output, max_seq_len):
    prompt_text = tokenizer.apply_chat_template(
        build_messages(patient_input), tokenize=False, add_generation_prompt=True
    )
    response_text = doctor_output + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    max_prompt_len = max_seq_len - len(response_ids)
    if max_prompt_len < 1:
        response_ids = response_ids[:max_seq_len]
        prompt_ids = []
    elif len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_inference_prompt(tokenizer, patient_input):
    return tokenizer.apply_chat_template(
        build_messages(patient_input), tokenize=False, add_generation_prompt=True
    )


class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len, max_samples=None, seed=SEED):
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


# ---- load model (QLoRA: 4-bit base + LoRA adapter) ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_config, dtype=torch.float16, device_map="auto"
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_ds = InstructionDataset(
    f"{DATA_DIR}/train.jsonl", tokenizer, MAX_SEQ_LEN, max_samples=MAX_TRAIN_SAMPLES
)
eval_ds = InstructionDataset(
    f"{DATA_DIR}/val.jsonl", tokenizer, MAX_SEQ_LEN, max_samples=MAX_EVAL_SAMPLES
)
print(f"train examples: {len(train_ds)} | eval examples: {len(eval_ds)}")

collator = PadCollator(tokenizer.pad_token_id)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=5,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to=[],
    seed=SEED,
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

adapter_dir = f"{OUT_DIR}/adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

metrics = {
    "run_id": "day8_pipeline_validation",
    "base_model": BASE_MODEL,
    "train_time_s": round(train_time_s, 1),
    "train_loss": train_result.training_loss,
    "log_history": trainer.state.log_history,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\nTrained. Adapter saved to {adapter_dir}")

# ---- sanity-check generations on a handful of val examples ----
model.eval()
val_sample = pd.read_json(f"{DATA_DIR}/val.jsonl", lines=True).sample(
    n=5, random_state=SEED
).reset_index(drop=True)

rows = []
for _, r in val_sample.iterrows():
    prompt = build_inference_prompt(tokenizer, r["input"])
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=150, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    rows.append({"input": r["input"], "reference": r["output"], "generation": gen})
    print(f"\nINPUT: {r['input'][:80]}\nGEN: {gen[:200]}")

pd.DataFrame(rows).to_csv(f"{OUT_DIR}/day8_sample_generations.csv", index=False)
print(f"\nDone. Outputs in {OUT_DIR}")
