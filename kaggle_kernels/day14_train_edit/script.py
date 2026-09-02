"""
Day 14 -- retrieval+EDIT fine-tune. Two prior attempts at combining
retrieval with generation both failed their own ablations:
  - Day 12 RAG (style-reference framing): 0.5562 vs 0.5584 without retrieval.
  - Day 13 zero-shot edit prompting (unrelated adapter, never trained for
    this): 0.5481, worse than both pure retrieval (0.5830) and plain
    generation (0.5597).
Neither actually trained the model to DO the edit task. This is the first
genuine test: explicitly supervise on (retrieved example + real question) ->
REAL correct answer (not the retrieved one), via src/prepare_edit_data.py.

SIZED FOR A SHORTER SESSION (MAX_HOURS below) than prior runs -- pushed with
~7.5h left before the Aug 24 00:00 BD deadline, need real buffer left for
eval + submission afterward. StopAfterHours saves a checkpoint and stops
cleanly regardless of how many of the N_TRAIN_ROWS rows it gets through.

SETUP: dataset owner must share "nascenia-processed-data" (now includes
train_edit.jsonl) with your Kaggle username first.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import inspect
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers
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

DATA_DIR = "/kaggle/input/datasets/ahsanulhoque48cu/nascenia-edit-data"
OUT_DIR = "/kaggle/working/run"
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42
MAX_SEQ_LEN = 1664          # same proven-safe value as Day 12's RAG run
MIN_PROMPT_TOKENS = 64
BATCH_SIZE = 2              # same OOM-fix-proven config as Day 12
GRAD_ACCUM = 8
LR = 2e-4
WARMUP_RATIO = 0.03
LORA_R, LORA_ALPHA = 32, 64
SAVE_STEPS = 250
MAX_HOURS = 3.0             # shorter than Day 12's 8.5h -- deadline pressure,
                             # need buffer left for eval + submission after
                             # (further cut from 4.0h: data-prep slippage ate
                             # into the ~7.5h budget available at 4:25pm)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"transformers {transformers.__version__}", flush=True)
print("Day 14 retrieval+edit fine-tune -- fresh adapter", flush=True)

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def hf_supports(name):
    return name in inspect.signature(TrainingArguments.__init__).parameters


def compat_kwargs():
    kw = {}
    if hf_supports("train_sampling_strategy"):
        kw["train_sampling_strategy"] = "group_by_length"
    elif hf_supports("group_by_length"):
        kw["group_by_length"] = True
    for k, v in [("save_only_model", True),
                 ("gradient_checkpointing_kwargs", {"use_reentrant": False})]:
        if hf_supports(k):
            kw[k] = v
    return kw


def build_example(tokenizer, patient_input, doctor_output, max_seq_len, min_prompt_tokens):
    # patient_input is already the full EDIT_PROMPT-enriched text baked in
    # by prepare_edit_data.py.
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
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {state.global_step}")


class StopAfterHours(TrainerCallback):
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

train_path = f"{DATA_DIR}/train_edit.jsonl"
if not os.path.exists(train_path):
    raise SystemExit(
        f"edit training data not found at {train_path}. Has the dataset owner "
        f"pushed the updated 'nascenia-processed-data' (with train_edit.jsonl)? "
        f"DATA_DIR holds: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'nothing'}")

df = pd.read_json(train_path, lines=True)
print(f"edit training set: {len(df):,} examples", flush=True)

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
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()

total_steps = max(1, len(ds) // (BATCH_SIZE * GRAD_ACCUM))
warmup = max(1, int(total_steps * WARMUP_RATIO))
print(f"\n~{total_steps} optimizer steps, {warmup} warmup\n", flush=True)

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
    save_total_limit=2,
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

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump({
        "technique": "retrieval_edit_fine_tune",
        "n_train": len(ds), "max_seq_len": MAX_SEQ_LEN,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "train_time_s": round(elapsed, 1), "steps": result.global_step,
        "train_loss": result.training_loss,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log_history": trainer.state.log_history,
    }, f, ensure_ascii=False, indent=2)

print(f"\nDONE in {elapsed/3600:.2f}h — {result.global_step} steps, "
      f"loss {result.training_loss:.4f}, peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print(f"adapter: {adapter_dir}")
