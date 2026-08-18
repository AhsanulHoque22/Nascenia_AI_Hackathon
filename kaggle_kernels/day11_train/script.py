"""
THE RUN — QLoRA fine-tune of Qwen3-1.7B over the FULL curated dataset,
as a chain of resumable shards. Single Kaggle T4.

WHY SHARDED. One epoch over the 81,771 curated rows costs ~34.5 GPU-hours at
the measured 0.658 ex/s. Kaggle kills a session at 12h and gives ~30 GPU-h a
week, so a single run is impossible on free tier. Instead each session trains
one ~16.4k-row shard and RESUMES from the previous session's adapter, so
training accumulates across the whole dataset:

    session 0:  shard 0, fresh adapter          rows      0- 16,354
    session 1:  shard 1, resumes session 0      rows 16,354- 32,708
    session 2:  shard 2, resumes session 1      rows 32,708- 49,062
    session 3:  shard 3, resumes session 2      rows 49,062- 65,416
    session 4:  shard 4, resumes session 3      rows 65,416- 81,770

Shards are disjoint slices of one fixed shuffle, so the chain sees every row
exactly once — a genuine single epoch, just serialized. Stopping early is
safe and simply means training on the first N shards.

To advance: bump SHARD_INDEX, upload the previous adapter as the dataset in
RESUME_DATASET, re-push. The adapter is ~140MB, so this is cheap.

Every other value is measured, not assumed (see TRAINING_NOTES.md). The
surprising ones:
  * 4-bit is ON — the probe measured it FASTER than fp16 here (6.08 vs 6.24
    s/step), and every fp16/no-checkpointing variant OOM'd at seq 1152.
  * checkpointing ON — peak is already 13.8GB of 14.56GB with it.
  * max_seq_len 1152, not 768 — at 768 the response crowded the question down
    to ~64 of 467 tokens and 28.9% of rows lost the question entirely.
  * no eval loop — checkpoints are ranked by COMPOSITE score in the next
    kernel, so an eval pass here is only a fresh OOM risk.
  * greeting/branding boilerplate is KEPT in the data: 76% of references open
    with "হেলো" and it appears in zero inputs. It is score, not noise.
"""

import os

# Kaggle allocates 2x T4; HF Trainer then silently wraps the model in
# DataParallel and crashes ("Caught RuntimeError in replica 0 on device 0"),
# poisoning the CUDA context. Pin to one card before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ~95% VRAM utilisation for hours; let the allocator grow blocks in place
# instead of stranding them and OOM-ing late in the run.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1"],
    check=True,
)
# Kaggle preinstalls torchao 0.10.0; peft's LoRA dispatcher RAISES (not
# returns False) on a version it considers too old, which kills model loading.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import glob
import inspect
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ============================ SESSION CONTROL ============================
SHARD_INDEX = 0        # driven by src/run_chain.py; bump per session
N_SHARDS = 5
RESUME = False         # True for sessions 1+; the adapter is found by glob
                       # because Kaggle's mount path varies by dataset form
# =========================================================================

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data"
OUT_DIR = "/kaggle/working/run"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42

MAX_SEQ_LEN = 1152
MIN_PROMPT_TOKENS = 64
BATCH_SIZE = 4
GRAD_ACCUM = 4               # effective batch 16
LR = 2e-4
WARMUP_RATIO = 0.03
LORA_R, LORA_ALPHA = 32, 64
SAVE_STEPS = 250
MAX_HOURS = 8.5              # self-stop well inside Kaggle's hard 12h kill

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"transformers {transformers.__version__}", flush=True)
print(f"SHARD {SHARD_INDEX}/{N_SHARDS}  resume={RESUME}", flush=True)

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def hf_supports(name):
    return name in inspect.signature(TrainingArguments.__init__).parameters


def compat_kwargs(group_by_length=True):
    """Kaggle ships transformers 4.x, local dev is on 5.x, and they disagree
    on argument names. Passing the wrong one is a TypeError at construction —
    a fast failure, but one that still costs a session slot. So ask the
    installed class what it accepts rather than pinning a name."""
    kw = {}
    if group_by_length:
        if hf_supports("train_sampling_strategy"):      # transformers 5.x
            kw["train_sampling_strategy"] = "group_by_length"
        elif hf_supports("group_by_length"):            # transformers 4.x
            kw["group_by_length"] = True
    for k, v in [("save_only_model", True),
                 ("gradient_checkpointing_kwargs", {"use_reentrant": False})]:
        if hf_supports(k):
            kw[k] = v
    return kw


def build_example(tokenizer, patient_input, doctor_output, max_seq_len, min_prompt_tokens):
    """Response is the target and is never truncated; the prompt gives way.
    If the response leaves too little room for the question, flag the example
    for dropping rather than train on a decapitated prompt."""
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
    """fp16 on sm_75 has no bf16 fallback; a NaN at hour 6 should stop the run,
    not quietly burn the rest of the session."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state.global_step}")


class StopAfterHours(TrainerCallback):
    """Kaggle kills at 12h with no grace period and may not persist
    /kaggle/working; stopping ourselves guarantees a saved adapter."""

    def __init__(self, hours):
        self.limit_s, self.t0 = hours * 3600, time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.t0 > self.limit_s:
            print(f"\n*** {MAX_HOURS}h limit at step {state.global_step} — "
                  f"saving and stopping ***", flush=True)
            control.should_training_stop = True
            control.should_save = True
        return control


class Heartbeat(TrainerCallback):
    """Kaggle streams logs but shows no ETA; print one so a stall is visible."""

    def __init__(self):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and state.global_step:
            el = time.time() - self.t0
            frac = state.global_step / max(state.max_steps, 1)
            print(f"    step {state.global_step}/{state.max_steps}  "
                  f"loss {logs['loss']:.4f}  {el/3600:.2f}h elapsed  "
                  f"~{el/max(frac,1e-9)-el:.0f}s left", flush=True)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

train_path = f"{DATA_DIR}/train_curated.jsonl"
if not os.path.exists(train_path):
    raise SystemExit(f"curated data missing at {train_path}; DATA_DIR holds "
                     f"{os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'nothing'}")

full = pd.read_json(train_path, lines=True)
# One fixed shuffle, then disjoint contiguous slices — so the chain of
# sessions covers every row exactly once.
full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
shard_size = len(full) // N_SHARDS + 1
lo, hi = SHARD_INDEX * shard_size, min((SHARD_INDEX + 1) * shard_size, len(full))
df = full.iloc[lo:hi].reset_index(drop=True)
print(f"curated pool {len(full):,} -> shard {SHARD_INDEX}: rows {lo:,}-{hi:,} "
      f"({len(df):,} examples)", flush=True)

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

resume_dir = None
if RESUME:
    # Search all mounted inputs rather than a fixed path: Kaggle mounts
    # datasets at both /kaggle/input/<slug> and
    # /kaggle/input/datasets/<user>/<slug> depending on how they were added.
    hits = [h for h in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)]
    if not hits:
        raise SystemExit(
            "RESUME=True but no adapter_config.json under /kaggle/input; "
            f"mounted: {glob.glob('/kaggle/input/*')}")
    resume_dir = os.path.dirname(sorted(hits)[0])

if resume_dir:
    # Continue training the SAME adapter rather than starting a new one, so
    # the chain of shards behaves like one continuous epoch.
    print(f"resuming adapter from {resume_dir}", flush=True)
    model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
else:
    print("fresh adapter", flush=True)
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

total_steps = max(1, len(ds) // (BATCH_SIZE * GRAD_ACCUM))
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
    eval_strategy="no",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=4,
    fp16=True,
    gradient_checkpointing=True,
    length_column_name="length",
    remove_unused_columns=False,
    optim="adamw_torch_fused",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    report_to=[],
    seed=SEED,
    **compat_kwargs(),
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

rows_done = hi if result.global_step >= total_steps else lo + result.global_step * BATCH_SIZE * GRAD_ACCUM
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump({
        "shard_index": SHARD_INDEX, "n_shards": N_SHARDS,
        "shard_rows": [lo, hi], "resumed_from": resume_dir,
        "n_train": len(ds), "rows_covered_cumulative": rows_done,
        "max_seq_len": MAX_SEQ_LEN, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lr": LR, "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "train_time_s": round(elapsed, 1), "steps": result.global_step,
        "train_loss": result.training_loss,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log_history": trainer.state.log_history,
    }, f, ensure_ascii=False, indent=2)

print(f"\nSHARD {SHARD_INDEX} DONE in {elapsed/3600:.2f}h — {result.global_step} steps, "
      f"loss {result.training_loss:.4f}, peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print(f"cumulative dataset coverage: ~{rows_done:,}/{len(full):,} rows "
      f"({100*rows_done/len(full):.0f}%)")
print(f"adapter: {adapter_dir}")
print(f"\nNEXT: shard {SHARD_INDEX+1} (src/run_chain.py handles this)"
      if SHARD_INDEX + 1 < N_SHARDS else "\nAll shards complete — full epoch done.")
