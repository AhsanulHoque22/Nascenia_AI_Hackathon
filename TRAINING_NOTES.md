# Training design notes — the one-shot run

Every configuration value in `configs/qwen3_full.yaml` traces to a
measurement on this corpus or a cited ablation. This file is the record of
why, and doubles as the source for the Phase 2 write-up (rulebook §5.1.3).

Context: Kaggle free tier only. GPU sessions cap at 12h and the weekly quota
is ~30 GPU-hours, so this is **one ~11h run**, not an iterative campaign.
Phase 1 (leaderboard) is 80% of the final score; Phase 2 (LLM-judge) is 20%.

---

## 1. The two bugs that dominated everything else

Both were found during the research pass, both predate any hyperparameter
question, and either alone was worth more than the entire tuning surface.

### 1.1 Output was capped at ~31% of reference length

`max_new_tokens=200` had been in every kernel since Day 5. Bengali tokenizes
at ~1.06 Qwen3 tokens/char, so 200 tokens ≈ 192 characters. Reference doctor
responses average **619 chars (~654 tokens)**.

Measured on the Day 9 fine-tuned adapter: `mean_pred_chars 192.2` vs
`mean_ref_chars 619.2`.

Token-F1 and ROUGE-L are F1 measures, so a length ratio of 0.31 caps recall
at 0.31, which caps Token-F1 at **0.47 even with perfect precision**. We
measured 0.367 — i.e. ~78% of a ceiling imposed by the cap itself. Every
score in `MODEL_SELECTION.md`, including the 0.5037 zero-shot baseline, was
measured under this handicap.

The Token-F1-vs-length curve is sharply **asymmetric** (measured by scoring
an unrelated real doctor response, truncated to length N, against 400 val
references — i.e. "right register, wrong content", a proxy for a fine-tuned
model):

| generated chars | Token-F1 |
|---|---|
| 192 (what we were doing) | ~0.345 |
| 450 | 0.526 |
| 600 | 0.551 |
| **750 (peak)** | **0.555** |
| 1200 | 0.551 |

Under-generating to 192 costs **38%**. Over-generating from 750 → 1200 costs
**0.7%**. So the target is deliberately set *above* the 601-char median.

BERTScore F1 is greedy-matched and normalized by both lengths, so it is
essentially length-invariant (measured spread 0.001 between a short and a
full-length candidate). That means length is tuned purely for Token-F1;
ROUGE-L peaks at the same place, and BERTScore comes along for free.

**Fix:** `max_new_tokens=900`, `min_new_tokens=400`. Expected **+0.05 to
+0.09 composite** for three config values.

### 1.2 At `max_seq_len=768`, the model never saw the patient's question

`build_example` reserves space for the response first (correct — it is the
target and must never be truncated), then left-truncates the prompt. With
responses averaging 674 tokens against a 768 budget, the prompt gets ~64 of
its 467 tokens.

Measured breakdown at 768 over the 97,853-row train split:

| condition | rows |
|---|---|
| response alone ≥ 768 → prompt erased **and** response chopped | **28,271 (28.9%)** |
| fits, prompt left-truncated | 63,130 (64.5%) |
| whole example fits | **6,452 (6.6%)** |

The 28.9% bucket is actively harmful twice over: it trains the model to
answer with no question visible, and to stop mid-sentence with no EOS —
which attacks the exact behaviour §1.1 depends on.

| max_seq_len | fits whole | prompt destroyed | mean seq (cost) |
|---|---|---|---|
| 768 | 6.6% | 28.9% | 758 |
| **1152** | **54.5%** | **4.6%** | **1035** |
| 1536 | 86.3% | 1.0% | 1137 |

**Fix:** `max_seq_len=1152` (+37% compute/example, removes 84% of the
corrupted rows) and **drop** rows whose response leaves under
`min_prompt_tokens=64` for the question rather than training on a decapitated
example.

---

## 2. Metric implementation — why the leaderboard gap is not what it looks like

Leaders sit at 0.87–0.90; our zero-shot measured 0.5037. We verified this is
**not** train/test leakage (`src/overlap_check.py`: 0 exact matches, mean
best-match TF-IDF cosine 0.47, only 0.3% of test inputs above 0.90 similarity).

The rulebook (§4) fixes the weights but specifies **no tokenizer and no
BERTScore backbone**, and both choices move the absolute score enormously.
Measured on 100 val rows, mismatched in-domain pairs:

| Token-F1 tokenizer | score |
|---|---|
| whitespace `.split()` | 0.169 |
| regex `\w+` (ours) | 0.577 |
| Qwen subword | 0.740 |

A plausible reconstruction of a 0.87 leaderboard using different but entirely
reasonable choices:

```
0.5 × 0.97  (XLM-R raw — extreme anisotropy, cosine >0.99 between randoms)
0.3 × 0.80  (subword-level token F1)
0.2 × 0.62  (subword ROUGE-L)
          = 0.849
```

Also: `bert_score`'s `lang2model` is a defaultdict falling through to
`bert-base-multilingual-cased` for "bn", and **no Bengali rescaling baseline
ships with the package**, so `rescale_with_baseline` is not available — and
rescaling moves scores *down* anyway, the wrong direction to explain the gap.

**Conclusion: proceed, do not redesign.** Every plausible implementation is
monotone increasing in the same latent quantity (lexical + semantic overlap
at roughly reference length), so the *ranking of our own candidate systems*
is preserved. Optimize against the local harness, trust relative deltas,
ignore absolute distance to the leaderboard.

### 2.1 Our own tokenizer was fragmenting Bengali

`_WORD_RE = re.compile(r"\w+")` — Python's `\w` excludes Unicode categories
`Mn`/`Mc`, which is where every Bengali vowel sign and hasant lives:

```
"হেলো, আপনার সমস্যাটি বুঝতে পেরেছি।"
  \w+   -> ['হ', 'ল', 'আপন', 'র', 'সমস', 'য', 'ট', 'ব', 'ঝত', 'প', 'র', 'ছ']
  split -> ['হেলো,', 'আপনার', 'সমস্যাটি', 'বুঝতে', 'পেরেছি।']
```

So our "Token-F1" is a grapheme-fragment F1 running ~2.4x inflated versus a
whitespace implementation. `src/eval_metrics.py` now reports **both**
tokenizations. `composite` stays on `regex` so numbers remain comparable with
`MODEL_SELECTION.md`; `composite_whitespace` is reported alongside.

### 2.2 Sobering baselines

Measured on 200–300 val rows:

| system | Token-F1 | est. composite |
|---|---|---|
| ours (Qwen3-1.7B zero-shot, 200-token cap) | 0.388 | **0.504** |
| 1-NN TF-IDF retrieval from train | 0.571 | ~0.56 |
| **one fixed Bengali reply, same for all 1000 rows** | **0.633** | **~0.60** |

A constant string beats our model, purely on length and format. **~0.60 is
the floor, not the target.**

---

## 3. Data curation (`src/curate_train.py`)

Because the metric is similarity-to-reference, "quality" means "looks like
the reference distribution", not "is better medical advice". Two inverted
instincts follow:

**Boilerplate is kept deliberately.** 76.3% of references open with `হেলো`
and 48.25% carry the source-platform branding — which appears in **zero**
inputs, train or test. A base model never emits it; a fine-tuned one does,
and each occurrence is a guaranteed token match on half the test set. A fixed
constant string scores Token-F1 0.633 largely on format alone, so roughly
60% of achievable Token-F1 comes from format and length before any medical
content is correct. Stripping boilerplate would throw away score.

**Rows are dropped for being truncated, not unhelpful.**

| filter | dropped | remaining |
|---|---|---|
| F1 response < 150 chars | 2,313 | 95,540 |
| F2 response > 1600 chars | 554 | 94,986 |
| F3 latin fraction > 0.05 | 585 | 94,401 |
| **F4 no terminal punctuation** | **6,428** | 87,973 |
| F5 duplicate input | 158 | 87,815 |
| F6 duplicate output | 1,181 | 86,634 |
| F7 input > 850 chars (~p95) | 4,863 | **81,771** |

F4 is the highest-value and least obvious: thousands of responses end in a
bare letter, a vowel sign, or a chopped `http…` — scrape damage that teaches
the model to stop mid-sentence.

Verified post-curation: `হেলো` opening rate **76.3%** (unchanged), response
mean 654 chars. If a fine-tuned model does **not** open with `হেলো` at ~76%
on val, the pipeline is broken — cheap sanity check.

**Deliberately NOT filtered:** rows where the response appears not to address
the input (IDF-weighted overlap is 0.0 for 26.4% of rows — a translation
artifact, not corruption; filtering here would delete a quarter of the data
on a measurement error).

---

## 4. Throughput

The baseline was leaving ~88% of the T4 idle: 16×768 tokens/step at 18.4s
works out to ~8.1 effective TFLOPS against the T4's 65 TFLOPS fp16 peak.

| lever | effect | taken |
|---|---|---|
| Drop 4-bit → fp16 LoRA | **1.25–1.4x** | yes |
| Drop gradient checkpointing | **1.25–1.4x** | yes |
| Liger fused linear CE | 1.05–1.15x, **−4 to −9 GB** | yes |
| Larger micro-batch, less grad_accum | 1.0–1.1x | yes |
| Pre-tokenize + workers + pin_memory | 1.0–1.15x | yes |
| `group_by_length` | matters once seq_len > 768 | yes |
| 2× T4 DDP | 1.7–1.9x | probe-gated |
| FlashAttention / better attention kernels | ~1% | **no** |
| Sequence packing | ~1.02x | **no** |
| 8-bit / paged optimizers | ~1.0x | **no** |
| torch.compile | 1.0–1.15x, high risk | **no** |

Three of these are cargo-cult in this specific setting, and skipping them is
as important as taking the others:

- **Attention is ~2% of FLOPs** at these sequence lengths (2·S²·h·L vs the
  matmuls). FlashAttention-class kernels cannot deliver 3–5x here — and FA2
  needs sm_80+ anyway, while T4 is sm_75.
- **Packing recovers ~2%**, because 87% of examples already filled the 768
  window. The "31 GPU-hours/epoch" figure was real compute, not padding.
- **Optimizer state is 139 MB** for 17.4M trainable params. Paged/8-bit
  optimizers solve a full-fine-tuning problem we do not have.

**4-bit is the counterintuitive one.** NF4 has no hardware support: it
dequantizes on FP32 CUDA cores (8.1 TFLOPS on a T4) before every fp16
tensor-core matmul (65 TFLOPS). It costs 20–40% throughput to save VRAM we
have spare — Qwen3-1.7B in fp16 is ~4.1 GB of a 16 GB card.

**The trap this creates:** `prepare_model_for_kbit_training()` is what calls
`enable_input_require_grads()`. Drop 4-bit while keeping gradient
checkpointing and PEFT silently receives **zero gradients** — loss flatlines
while the run reports success. `src/train.py` calls it explicitly, and the
calibration probe asserts `grad_norm > 0` per variant.

---

## 5. Accuracy techniques — including what was rejected

| technique | verdict | why |
|---|---|---|
| LoRA r=32, alpha=64 | **yes** | Bengali + medical is a double distribution shift, the regime where adapter capacity binds. ~2–4% step cost. alpha=2r keeps the LR valid. |
| All 7 target modules | **yes** | Attention-only underperforms MLP-only at matched parameter count. LoRA params are ~2% of FLOPs, so dropping MLP saves ~3% wall-clock for a real accuracy hit. |
| Response-only loss masking | **yes** | Standard for prompt→response SFT. Caveat: select checkpoints on *completion* loss, not full-sequence loss. |
| 1 epoch, maximize unique examples | **yes** | At fixed compute, up to 4 epochs of repeats ≈ unique data on loss — but BERTScore is 50% of the score and rewards topical coverage, which comes from unique examples. Also no epoch-2 overfitting cliff to fall off with one shot. |
| cosine + `warmup_ratio` 0.03, lr 2e-4 | **yes** | LoRA optimum is ~10x full-FT LR. Warmup's value here is robustness, which is what a one-shot fp16 run needs. |
| **NEFTune** | **NO** | Its own paper (Fig. 5) shows NEFTune-trained models score **lower** on ROUGE-L and BLEU vs ground truth — that is the paper's selling point ("does not lock in to exact wording") and it is precisely what our metric rewards. Every reported gain is LLM-judge win-rate. Clearest yes/no in the entire design. |
| **rsLoRA** | **NO** | Its benefit is at r≥128. At r=32/alpha=64 it would jump the update scale from 2.0 to 11.3, silently invalidating the LR — on fp16, on a T4, with one attempt. |
| **DoRA** | **NO** | ~+1% on commonsense/multiple-choice, for 20–40% throughput. Under a fixed wall-clock that trades ~25% of our data for a gain on a different metric family. |
| **LoRA+** | **NO** | 1–2% claimed, largest on GLUE-style classification, and interacts with fp16 overflow risk in a way nobody has characterized for Qwen3 on T4. Inside the run-to-run variance we cannot measure with one run. |
| **Beam search** | **NO** | The beam-search curse is strongest for long sequences and biases toward *shorter* hypotheses — the exact direction that hurts us — at 2.5–4x runtime. |

---

## 6. Inference

```python
max_new_tokens       = 900    # past the plateau start (~630) with headroom
min_new_tokens       = 400    # blocks the catastrophic under-generation mode
repetition_penalty   = 1.05   # >1.2 causes bn/en language mixing in Qwen3
no_repeat_ngram_size = 6      # NOT 3 — the summarisation default blocks
                              # legitimate trigrams the reference itself contains
num_beams            = 1
```

Batching is mandatory, not an optimization: 1,000 test rows × 900 tokens
unbatched on a T4 is ~8 hours; batched with left-padding and length-sorted
batches it is ~30–45 minutes. Decoder-only models **require**
`padding_side="left"` for batched generation — with right padding the model
continues from pad tokens and emits garbage.

Merge the LoRA adapter before generating, and do **not** run 4-bit at
inference: NF4 dequant makes generation slower than fp16 on a T4.

`enable_thinking=False` is correct for this metric — `<think>` traces are
pure precision loss, since every reasoning token is unmatched against the
reference.

---

## 7. Checkpoint selection

Save several checkpoints and pick by **composite score on a held-out sample**,
not by validation loss. The two are different objectives, and the gap between
best and last checkpoint is frequently larger than any hyperparameter effect
above. Budget ~30 min of the session for this.
