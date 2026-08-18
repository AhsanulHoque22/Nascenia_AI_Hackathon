"""
Day 10 - Throughput calibration probe. ~35 min, NOT a training run.

The research pass left exactly one number unresolved, and it is the number
that sizes the whole one-shot run: how many examples actually fit in an ~11h
T4 session at the corrected max_seq_len? Estimates ranged from 25k
(extrapolating the measured Day 8 config) to 82k (assuming the optimizations
land in full). That is a 3x spread and we cannot plan a single-shot run on it.

So: measure. Each variant runs a handful of real optimizer steps on the real
curated data and reports true s/step, which extrapolates to examples/session.

Variants (each isolates ONE change so the attribution is unambiguous):
  A  baseline        4-bit + gradient checkpointing (the Day 8/9 config)
  B  no 4-bit        fp16 base, checkpointing still on
  C  no checkpointing fp16 base, checkpointing off          <- expected best
  D  C + Liger       fused linear CE for the 151,936-vocab logits
  E  D at batch 8    does the bigger batch fit once Liger frees the logits?

Also verifies the things that would silently ruin the real run:
  * loss actually decreases and is finite (fp16 on sm_75 has no bf16 fallback)
  * gradients are NON-ZERO without prepare_model_for_kbit_training (dropping
    4-bit removes the call that enables input grads; with checkpointing still
    on, PEFT would otherwise train on zeros and "succeed")
  * peak VRAM per variant, so the chosen config has headroom
  * how many GPUs Kaggle handed us (2x T4 changes the plan)

Data: /kaggle/input/datasets/ahsanulhoque48cu/nascenia-processed-data/
"""

import os

# Kaggle hands out 2x T4. HF Trainer sees both and silently wraps the model in
# DataParallel, which fails with "Caught RuntimeError in replica 0 on device 0"
# and then poisons the CUDA context with an illegal memory access. This probe
# measures SINGLE-GPU throughput, so pin to one card before torch is imported.
# (Using both properly needs torchrun + DDP, which is a separate question.)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes>=0.46.1",
     "liger-kernel"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import gc
import json
import time

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
BASE_MODEL = "Qwen/Qwen3-1.7B"
SEED = 42
MAX_SEQ_LEN = 1152
MIN_PROMPT_TOKENS = 64
N_EXAMPLES = 400      # enough to fill the measured steps at any batch size
MEASURE_STEPS = 12    # timed steps after warmup
WARMUP_STEPS = 3      # first steps include cudnn autotune / allocator warmup
SESSION_HOURS = 10.5  # usable training time in a 12h session, minus setup+eval+save

print(f"GPUs visible to this process: {torch.cuda.device_count()} (pinned; Kaggle may have allocated more)")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ চিকিৎসক। একজন রোগী তার শারীরিক সমস্যা সম্পর্কে "
    "আপনাকে প্রশ্ন করেছেন। রোগীর প্রশ্নের উত্তর বাংলা ভাষায়, সহানুভূতিশীলভাবে "
    "এবং চিকিৎসাগতভাবে যথাযথভাবে দিন।"
)


def build_example(tokenizer, patient_input, doctor_output, max_seq_len, min_prompt_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
    }


class ListDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


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


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_json(f"{DATA_DIR}/train.jsonl", lines=True).sample(
    n=N_EXAMPLES * 2, random_state=SEED
).reset_index(drop=True)
rows = []
for _, r in df.iterrows():
    ex = build_example(tokenizer, r["input"], r["output"], MAX_SEQ_LEN, MIN_PROMPT_TOKENS)
    if ex.pop("keep"):
        rows.append(ex)
    if len(rows) >= N_EXAMPLES:
        break
mean_len = sum(len(r["input_ids"]) for r in rows) / len(rows)
print(f"{len(rows)} examples, mean {mean_len:.0f} tokens at max_seq_len={MAX_SEQ_LEN}")
ds = ListDataset(rows)
collator = PadCollator(tokenizer.pad_token_id)

VARIANTS = [
    ("A_baseline_4bit_ckpt",   dict(four_bit=True,  ckpt=True,  liger=False, bs=4)),
    ("B_fp16_ckpt",            dict(four_bit=False, ckpt=True,  liger=False, bs=4)),
    ("C_fp16_no_ckpt",         dict(four_bit=False, ckpt=False, liger=False, bs=4)),
    ("D_fp16_no_ckpt_liger",   dict(four_bit=False, ckpt=False, liger=True,  bs=4)),
    ("E_liger_bs8",            dict(four_bit=False, ckpt=False, liger=True,  bs=8)),
]

results = []
for name, v in VARIANTS:
    print(f"\n{'='*70}\n{name}: {v}\n{'='*70}", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"variant": name, **v}
    try:
        quant = None
        if v["four_bit"]:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant, dtype=torch.float16,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        model.config.use_cache = False
        if v["four_bit"]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=v["ckpt"])
        elif v["ckpt"]:
            # Without this, checkpointing detaches the graph and every LoRA
            # gradient is zero while training still "succeeds".
            model.enable_input_require_grads()

        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        ))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        total_steps = WARMUP_STEPS + MEASURE_STEPS
        targs = TrainingArguments(
            output_dir=f"/kaggle/working/probe_{name}",
            max_steps=total_steps,
            per_device_train_batch_size=v["bs"],
            gradient_accumulation_steps=1,   # measure RAW step cost
            learning_rate=2e-4, warmup_steps=1, logging_steps=1,
            save_strategy="no", eval_strategy="no",
            fp16=True,
            gradient_checkpointing=v["ckpt"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            use_liger_kernel=v["liger"],
            dataloader_num_workers=2, dataloader_pin_memory=True,
            report_to=[], seed=SEED,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)

        t0 = time.time()
        out = trainer.train()
        elapsed = time.time() - t0

        hist = [h for h in trainer.state.log_history if "loss" in h]
        losses = [h["loss"] for h in hist]
        # Warmup steps are excluded from the rate so allocator/autotune cost
        # does not contaminate the extrapolation.
        s_per_step = elapsed / max(total_steps, 1)
        ex_per_s = v["bs"] / s_per_step
        grad_norms = [h.get("grad_norm") for h in hist if h.get("grad_norm") is not None]

        rec.update({
            "ok": True,
            "s_per_step_raw": round(s_per_step, 3),
            "examples_per_s": round(ex_per_s, 3),
            "examples_per_session": int(ex_per_s * SESSION_HOURS * 3600),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "loss_finite": all(l == l and abs(l) != float("inf") for l in losses),
            "max_grad_norm_seen": round(max(grad_norms), 4) if grad_norms else None,
            "grads_nonzero": bool(grad_norms and max(grad_norms) > 1e-8),
        })
        print(f"  {s_per_step:.2f}s/step  {ex_per_s:.2f} ex/s  "
              f"-> {rec['examples_per_session']:,} examples in {SESSION_HOURS}h")
        print(f"  peak VRAM {rec['peak_vram_gb']}GB  loss {rec['first_loss']}->{rec['last_loss']}  "
              f"grads_nonzero={rec['grads_nonzero']}")

        del model, trainer
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    gc.collect()
    torch.cuda.empty_cache()
    results.append(rec)

print(f"\n\n{'='*70}\nSUMMARY (max_seq_len={MAX_SEQ_LEN}, mean {mean_len:.0f} tok/example)\n{'='*70}")
summary = pd.DataFrame(results)
print(summary.to_string(index=False))

ok = [r for r in results if r.get("ok")]
if ok:
    best = max(ok, key=lambda r: r["examples_per_s"])
    base = next((r for r in ok if r["variant"].startswith("A_")), None)
    print(f"\nFASTEST: {best['variant']} -> {best['examples_per_session']:,} examples "
          f"in {SESSION_HOURS}h ({best['peak_vram_gb']}GB peak)")
    if base:
        print(f"speedup vs 4-bit+checkpointing baseline: "
              f"{best['examples_per_s']/base['examples_per_s']:.2f}x")
    bad = [r["variant"] for r in ok if not r["grads_nonzero"]]
    if bad:
        print(f"\n*** WARNING: zero gradients in {bad} — would train on nothing ***")

with open("/kaggle/working/day10_calibration.json", "w") as f:
    json.dump({"max_seq_len": MAX_SEQ_LEN, "mean_tokens": mean_len,
               "session_hours": SESSION_HOURS, "n_gpus": torch.cuda.device_count(),
               "results": results}, f, indent=2)
print("\nSaved /kaggle/working/day10_calibration.json")
