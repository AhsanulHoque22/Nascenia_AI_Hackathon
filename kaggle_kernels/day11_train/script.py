"""
Day 11 - THE RUN. QLoRA fine-tune of Qwen3-1.7B, single Kaggle T4.

Training only. Inference and checkpoint selection are a SEPARATE kernel
(day11_select_submit) because they do not fit: training alone is ~7.6h,
generation on 1,000 test rows is another 30-45 min, and scoring several
checkpoints another ~30 min — against a HARD 12h session kill that does not
reliably persist /kaggle/working.

Every value here is measured, not assumed. See TRAINING_NOTES.md. The ones
that would surprise someone reading this cold:

  * 4-bit is ON. The calibration probe measured 6.08s/step with it and
    6.24s/step without, and every fp16/no-checkpointing variant OOM'd at
    seq 1152. The widely-repeated "drop 4-bit for 1.25-1.4x" advice assumes
    headroom this configuration does not have.
  * gradient_checkpointing is ON for the same reason — peak VRAM is already
    13.8GB of 14.56GB with it.
  * max_seq_len 1152, not 768. At 768 the response (mean 674 tokens) crowded
    the patient's question down to ~64 of its 467 tokens and 28.9% of rows
    lost the question entirely.
  * No eval loop. Checkpoints are chosen by COMPOSITE score in the next
    kernel, not by val loss, so an eval pass here would only risk a fresh
    OOM for a number we discard.
  * Greeting/branding boilerplate is deliberately KEPT in the data. 76% of
    references open with "হেলো" and it appears in zero inputs — it is score,
    not noise.

Self-stops at 8.5h so the adapter is always written.

Data: /kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/
"""

import os

# Kaggle allocates 2x T4; HF Trainer then silently wraps the model in
# DataParallel and crashes ("Caught RuntimeError in replica 0 on device 0"),
# poisoning the CUDA context. Pin to one card before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 95% VRAM utilisation for hours; let the allocator grow blocks in place
# rather than stranding them and OOM-ing late.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1"],
    check=True,
)
# Kaggle preinstalls torchao 0.10.0; peft's LoRA dispatcher RAISES (not
# returns False) on a version it considers too old, killing model loading.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import json
import random
import time

import numpy as np
import pandas as pd
import torch
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

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
OUT_DIR = "/kaggle/working/run"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42

MAX_SEQ_LEN = 1152
MIN_PROMPT_TOKENS = 64
MAX_TRAIN_SAMPLES = 18000     # ~7.6h at the measured 0.658 ex/s
BATCH_SIZE = 4
GRAD_ACCUM = 4                # effective batch 16
LR = 2e-4
WARMUP_RATIO = 0.03
LORA_R, LORA_ALPHA = 32, 64
SAVE_STEPS = 250
MAX_HOURS = 8.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def build_example(tokenizer, patient_input, doctor_output, max_seq_len, min_prompt_tokens):
    """Response is the target and is never truncated; the prompt gives way.
    If the response leaves too little room for the question, the example is
    flagged for dropping rather than trained on with a decapitated prompt."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": patient_input}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(doctor_output + tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"]
    keep = True
    max_prompt_len = max_seq_len - len(response_ids)
    if max_prompt_len < min_prompt_tokens:
        keep = False
        max_prompt_len = max(max_prompt_len, 1)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    response_ids = response_ids[:max_seq_len - len(prompt_ids)]
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + list(response_ids),
        "keep": keep,
        "length": len(input_ids),
    }


class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ii, am, lb = [], [], []
        for b in batch:
            p = max_len - len(b["input_ids"])
            ii.append(b["input_ids"] + [self.pad_token_id] * p)
            am.append(b["attention_mask"] + [0] * p)
            lb.append(b["labels"] + [-100] * p)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels": torch.tensor(lb, dtype=torch.long),
        }


class AbortOnNaN(TrainerCallback):
    """fp16 on sm_75 has no bf16 fallback; a NaN at hour 6 should stop, not
    quietly burn the rest of the session."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state.global_step}")


class StopAfterHours(TrainerCallback):
    """Kaggle kills at 12h with no grace period and a killed kernel may not
    persist /kaggle/working. Stopping ourselves guarantees a saved adapter."""

    def __init__(self, hours):
        self.limit_s, self.t0 = hours * 3600, time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.t0 > self.limit_s:
            print(f"\n*** {MAX_HOURS}h wall-clock limit at step {state.global_step} "
                  f"— saving and stopping ***", flush=True)
            control.should_training_stop = True
            control.should_save = True
        return control


class Heartbeat(TrainerCallback):
    """Kaggle streams logs but shows no ETA; print one so a stalled run is
    visible from the outside."""

    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and state.global_step:
            el = time.time() - self.t0
            frac = state.global_step / max(state.max_steps, 1)
            eta = el / max(frac, 1e-9) - el
            print(f"    step {state.global_step}/{state.max_steps}  "
                  f"loss {logs['loss']:.4f}  {el/3600:.2f}h elapsed  "
                  f"~{eta/3600:.2f}h left", flush=True)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

train_path = f"{DATA_DIR}/train_curated.jsonl"
if not os.path.exists(train_path):
    raise SystemExit(f"curated data not found at {train_path} — "
                     f"dataset contains: {os.listdir(DATA_DIR)}")

df = pd.read_json(train_path, lines=True)
print(f"curated pool: {len(df):,} rows", flush=True)
df = df.sample(n=min(MAX_TRAIN_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)

ds = Dataset.from_pandas(df[["input", "output"]], preserve_index=False)
ds = ds.map(
    lambda r: build_example(tokenizer, r["input"], r["output"], MAX_SEQ_LEN, MIN_PROMPT_TOKENS),
    remove_columns=["input", "output"], num_proc=4, desc="tokenizing",
)
before = len(ds)
ds = ds.filter(lambda r: r["keep"], num_proc=4).remove_columns(["keep"])
lens = np.array(ds["length"])
print(f"train: {len(ds):,} examples ({before-len(ds)} dropped)", flush=True)
print(f"  tokens: mean {lens.mean():.0f}  p50 {np.percentile(lens,50):.0f}  "
      f"p95 {np.percentile(lens,95):.0f}  max {lens.max()}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    ),
    dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
))
# Trainable adapter in fp32 while the frozen base stays quantized — standard
# mixed precision, and materially more stable on a card with no bf16.
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()

total_steps = len(ds) // (BATCH_SIZE * GRAD_ACCUM)
warmup = max(1, int(total_steps * WARMUP_RATIO))
print(f"\n~{total_steps} optimizer steps, {warmup} warmup, "
      f"est {len(ds)/0.658/3600:.1f}h at measured throughput\n", flush=True)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=warmup,
    max_grad_norm=1.0,
    logging_steps=25,
    eval_strategy="no",          # selection is by composite score, in the next kernel
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    save_only_model=True,        # we never resume; optimizer state doubles checkpoint size
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    train_sampling_strategy="group_by_length",   # transformers 5.x name
    length_column_name="length",
    remove_unused_columns=False,
    optim="adamw_torch_fused",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to=[],
    seed=SEED,
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=PadCollator(tokenizer.pad_token_id),
    callbacks=[AbortOnNaN(), StopAfterHours(MAX_HOURS), Heartbeat()],
)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

adapter_dir = f"{OUT_DIR}/adapter_final"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump({
        "base_model": BASE_MODEL, "n_train": len(ds),
        "max_seq_len": MAX_SEQ_LEN, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lr": LR, "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "train_time_s": round(elapsed, 1), "steps": result.global_step,
        "train_loss": result.training_loss,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log_history": trainer.state.log_history,
    }, f, ensure_ascii=False, indent=2)

print(f"\nDONE in {elapsed/3600:.2f}h — {result.global_step} steps, "
      f"loss {result.training_loss:.4f}, peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print(f"final adapter: {adapter_dir}")
print("checkpoints:", sorted(d for d in os.listdir(OUT_DIR) if d.startswith("checkpoint-")))
