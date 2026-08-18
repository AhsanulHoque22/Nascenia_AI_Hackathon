"""
Day 9 (model swap) - Qwen3-1.7B validation on Kaggle T4.

Decision to move off Qwen2.5-1.5B-Instruct (Day 6 primary candidate) onto
Qwen3-1.7B, made after Day 8's pipeline was already validated on Qwen2.5.
Before committing the full ~98k-row fine-tune to the new model, this kernel
redoes the two checks that justified the original choice, on Qwen3-1.7B:

  Part A - Day 6-equivalent: param count, tokenizer fragmentation, zero-shot
  composite score on the SAME 15 val examples (seed=42) used for every other
  candidate in MODEL_SELECTION.md, for a directly comparable number.

  Part B - Day 8-equivalent: small-scale QLoRA fine-tune (500 train examples,
  1 epoch) to confirm the pipeline (LoRA target modules, masking, Trainer
  loop) works on Qwen3's architecture, and that post-fine-tune generations
  are direct answers (no leftover <think> tags) and non-garbage.

Qwen3 uses a hybrid think/no-think chat template. enable_thinking=False is
passed to apply_chat_template everywhere (matches src/prompt_template.py) --
this task needs a direct empathetic response, not a reasoning block, and
untamed thinking would burn generation budget and pollute the scored output.

Kernel has no local src/ import; prompt template + train.py logic inlined.

Data source: /kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/
"""

import os

# Kaggle T4 sessions sometimes allocate 2 GPUs; device_map="auto" would then
# shard the model across both, which breaks Trainer's forward pass with a
# cuda:0/cuda:1 device-mismatch error under bitsandbytes 4-bit quantization.
# Pin everything to a single GPU before torch/transformers ever see the rest.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1",
     "bert-score", "rouge-score"],
    check=True,
)

import gc
import json
import random
import re
import time
import unicodedata
from collections import Counter

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
OUT_DIR = "/kaggle/working/day9_qwen3_run"
SEED = 42
CAP = 3_000_000_000
BASE_MODEL = "Qwen/Qwen3-1.7B"
N_ZEROSHOT = 15
MAX_TRAIN_SAMPLES = 500
MAX_EVAL_SAMPLES = 20
MAX_SEQ_LEN = 768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


def build_inference_prompt(tokenizer, patient_input):
    return tokenizer.apply_chat_template(
        build_messages(patient_input), tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def build_example(tokenizer, patient_input, doctor_output, max_seq_len):
    prompt_text = build_inference_prompt(tokenizer, patient_input)
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


# ---- eval_metrics logic, inlined (Day 6 methodology) ----
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text):
    return _WORD_RE.findall(unicodedata.normalize("NFC", text))


class _SharedTokenizer:
    def tokenize(self, text):
        return simple_tokenize(text)


from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_SharedTokenizer())


def token_f1(pred, ref):
    p, r = simple_tokenize(pred), simple_tokenize(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l_f1(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].fmeasure


# =========================================================================
# Part A: Day 6-equivalent -- param count, tokenizer fragmentation, zero-shot
# =========================================================================
print(f"{'='*70}\nPart A: Day 6-equivalent validation -- {BASE_MODEL}\n{'='*70}")

val = pd.read_json(f"{DATA_DIR}/val.jsonl", lines=True).sample(
    n=N_ZEROSHOT, random_state=SEED
).reset_index(drop=True)
refs = val["output"].tolist()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
zs_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map="cuda")
zs_model.eval()

n_params = sum(p.numel() for p in zs_model.parameters())
under_cap = n_params < CAP
print(f"params: {n_params:,} ({n_params/1e9:.4f}B) -- under 3B cap: {under_cap}")

char_counts = val["input"].str.len().sum()
token_counts = sum(len(tokenizer.encode(t)) for t in val["input"])
frag = token_counts / char_counts
print(f"tokenizer fragmentation: {frag:.4f} tokens/char (over {N_ZEROSHOT} val inputs)")

preds = []
t0 = time.time()
for _, r in val.iterrows():
    prompt = build_inference_prompt(tokenizer, r["input"])
    inputs = tokenizer(prompt, return_tensors="pt").to(zs_model.device)
    with torch.no_grad():
        out = zs_model.generate(
            **inputs, max_new_tokens=200, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    preds.append(gen)
gen_time = time.time() - t0
print(f"generated {N_ZEROSHOT} responses in {gen_time:.1f}s ({gen_time/N_ZEROSHOT:.2f}s/example)")
print(f"contains leftover <think> tag: {sum('<think>' in p for p in preds)}/{N_ZEROSHOT}")

from bert_score import score as bertscore_score

_, _, bert_f1 = bertscore_score(
    preds, refs, model_type="bert-base-multilingual-cased", lang="bn",
    batch_size=16, verbose=False, device="cuda",
)
bert_f1 = bert_f1.tolist()
tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
rg_f1 = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
composite = [0.5 * b + 0.3 * t + 0.2 * rg for b, t, rg in zip(bert_f1, tok_f1, rg_f1)]

zeroshot_result = {
    "model_id": BASE_MODEL,
    "params": n_params,
    "under_3b_cap": under_cap,
    "tokens_per_char": round(frag, 4),
    "mean_bertscore_f1": round(sum(bert_f1) / len(bert_f1), 4),
    "mean_token_f1": round(sum(tok_f1) / len(tok_f1), 4),
    "mean_rouge_l_f1": round(sum(rg_f1) / len(rg_f1), 4),
    "mean_composite": round(sum(composite) / len(composite), 4),
    "leftover_think_tags": sum("<think>" in p for p in preds),
}
print(f"\nZero-shot composite: {zeroshot_result['mean_composite']}")
print("(Day 5 Qwen2.5-0.5B floor: 0.4409 | Day 6 Qwen2.5-1.5B: 0.4979)")

pd.DataFrame({"id": val["id"], "input": val["input"], "output": val["output"], "pred": preds}).to_csv(
    "/kaggle/working/day9_qwen3_zeroshot_predictions.csv", index=False
)
with open("/kaggle/working/day9_qwen3_zeroshot_results.json", "w", encoding="utf-8") as f:
    json.dump(zeroshot_result, f, ensure_ascii=False, indent=2)

del zs_model
gc.collect()
torch.cuda.empty_cache()

# =========================================================================
# Part B: Day 8-equivalent -- small-scale QLoRA pipeline validation
# =========================================================================
print(f"\n{'='*70}\nPart B: Day 8-equivalent QLoRA pipeline validation\n{'='*70}")


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
    "run_id": "day9_qwen3_pipeline_validation",
    "base_model": BASE_MODEL,
    "train_time_s": round(train_time_s, 1),
    "train_loss": train_result.training_loss,
    "log_history": trainer.state.log_history,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"\nTrained. Adapter saved to {adapter_dir}")

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

pd.DataFrame(rows).to_csv(f"{OUT_DIR}/day9_sample_generations.csv", index=False)
print(f"\nDone. Outputs in {OUT_DIR} and /kaggle/working/")
