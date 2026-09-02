# Nascenia AI Hackathon — Phase 2 write-up

Bengali medical dialogue generation: single-turn patient question -> doctor
response, in Bengali, empathetic and medically appropriate.

## IMPORTANT: what actually reproduces the final Phase 1 submission

The final/last Kaggle submission (per rulebook §8, last submission is what
counts unless another is explicitly selected) is **not** the LoRA fine-tuned
model described below. It is `src/retrieval_hybrid_submission.py` — a
zero-model TF-IDF (word + character n-gram) nearest-neighbor retrieval over
`train_curated.jsonl`, no GPU/training involved. It scored **0.55756** real,
beating the LoRA model's **0.52418** and an earlier word-only retrieval
variant's 0.55462. See "Retrieval — the actual best submission" below for
the full comparison and rationale.

The LoRA fine-tuning work below is the substantive ML approach that was
explored first and remains the more defensible methodology, but it did not
score highest on the competition's actual metric — that's disclosed
explicitly, not glossed over. `inference_script.py` /
`download_weights.sh` in this package reproduce the **LoRA model**, not the
retrieval submission; if Phase 2 is ever reached (currently rank ~90/98,
nowhere near the top-10 cutoff of ~0.878, so unlikely) the retrieval script
is what would need to be packaged instead, since Phase 2 requires
reproducing the leaderboard-submitted outputs specifically.

## Approach (LoRA fine-tuning — not the final submission, see above)

Base model **Qwen/Qwen3-1.7B**, fine-tuned with a single LoRA adapter via
PEFT/SFT on the competition's own `train.csv` (cleaned — see Data below).
No RAG/retrieval, no ensembling, no distillation: one adapter, straightforward
supervised fine-tuning, selected as the best of 5 independently-trained
candidates.

**Why Qwen3-1.7B:** shortlisted from Qwen2.5-0.5B/1.5B/3B, gemma-2-2b, and
sarvam-2b-v0.5 on zero-shot Bengali generation quality and the 3B parameter
cap (`MODEL_SELECTION.md`). Qwen2.5-3B was disqualified outright (3.086B, over
cap). The initial pick was Qwen2.5-1.5B-Instruct; Qwen3-1.7B was swapped in
before fine-tuning began (Day 9) for materially better Bengali coherence at a
similar parameter budget, still comfortably under the 3B cap with room for
adapter overhead.

**Multi-account parallel training:** Kaggle's free tier caps GPU sessions at
12h with a ~30 GPU-hour weekly quota per account — not enough for iterative
tuning on one account. Training was split across 3 teammates' Kaggle
accounts, each training an independent LoRA adapter from the *same* random
seed and initialization on a disjoint shard of the training data (so the
adapters share a compatible init for weight-averaging, FedAvg-style — see
`src/merge_chain.py`). 5 candidates were produced this way: 4 independent
shard adapters plus 1 weight-merged average of them. All 5 were scored on
the same held-out validation split under the competition's exact composite
metric, and the single best-scoring adapter was selected for submission —
not the merge, and not by training loss.

## LoRA configuration

```
r = 32, alpha = 64, dropout = 0.05
target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # all 7, attention + MLP
bias = "none"
learning_rate = 2e-4, cosine schedule, warmup_ratio-based warmup
per_device_batch_size = 4, grad_accum = 4  -> effective batch 16
max_seq_len = 1152, num_train_epochs = 1
optimizer = adamw_torch_fused, fp16, gradient_checkpointing = True
```

Rationale (full detail in `TRAINING_NOTES.md`): r=32/alpha=64 because
Bengali + medical is a double distribution shift, the regime where adapter
capacity actually binds; all 7 target modules because attention-only
underperforms MLP-only at matched parameter count, and LoRA's own params are
only ~2% of FLOPs, so including MLP costs little for a real accuracy gain.

## Parameter count (rulebook §compliance, inference-time)

Computed exactly from the Qwen3-1.7B architecture config (`hidden_size=2048,
num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
head_dim=128, intermediate_size=6144, vocab_size=151936, tied embeddings`)
and the LoRA config actually used (`r=32`, all 7 target modules, all 28
layers) — cross-checked against Qwen's officially published non-embedding
count for this model (1.4B), which the base-model figure below matches:

| | params |
|---|---|
| base model (Qwen3-1.7B) | 1,720,574,976 (1.7206B) |
| LoRA adapter (r=32, 7 modules × 28 layers) | 34,865,152 (34.87M) |
| **total, inference-time** | **1,755,440,128 (1.7554B)** |
| 3B cap | 3,000,000,000 |
| **headroom** | **1,244,559,872 (~1.245B)** |

Well under the 3,000,000,000 cap. LoRA adapter weights add negligibly (order
tens of millions of parameters against a ~2B base).

## Selected checkpoint

`sanzidislam/nascenia-shard-2-adapter` (one of the 3 teammate-trained shards
described above), selected by `src/merge_chain.py` /
`kaggle_kernels/day11_select_submit` because it scored highest composite
among all 5 candidates on held-out validation:

| metric | value |
|---|---|
| BERTScore F1 | 0.6770 |
| Token-F1 | 0.5581 |
| ROUGE-L | 0.2690 |
| **local composite** (0.5·BERTScore + 0.3·TokenF1 + 0.2·ROUGE-L) | **0.5597** |
| **real leaderboard score** | **0.52418** |

The ~0.035 gap between local and real leaderboard score is a known, measured
calibration offset (see Known limitations) — not evidence of overfitting to
the local val split; the same offset direction and rough magnitude was
observed again on a second, independently-trained adapter (Day 12).

## Retrieval — the actual best submission

After the LoRA model's real leaderboard score (0.52418) put the team at rank
90/98 — nowhere near the top-10 cutoff (~0.878) — a diagnostic question was
tested directly on the real leaderboard rather than guessed at locally:
does near-verbatim retrieval overlap explain the gap to top teams? If it
did, pure retrieval (no model at all) should score close to 0.85-0.90.

It did not. Three real submissions, in order:

| technique | local composite | **real leaderboard** |
|---|---|---|
| LoRA fine-tuned model (above) | 0.5597 | 0.52418 |
| word-level TF-IDF retrieval (1-NN, no model) | 0.5757 | **0.55462** |
| word+char n-gram hybrid TF-IDF retrieval (1-NN, no model) | 0.5830 | **0.55756** |

Findings:
- **Retrieval beats fine-tuning here.** Copying the real answer to the
  nearest training-set question by TF-IDF cosine similarity — zero model,
  zero GPU — outscored the trained LoRA adapter by +0.030 real. The
  fine-tuning did not add value over reusing an existing real answer, on
  this metric.
- **But retrieval alone doesn't explain the top of the leaderboard.** 0.557
  is far short of 0.85-0.90. If surface/near-verbatim overlap were what top
  teams were exploiting, pure retrieval would land much closer to that
  range. It plateaus instead at roughly the same ceiling the constant-string
  baseline analysis already predicted (~0.58 local) for anything driven by
  boilerplate/format overlap rather than genuine content quality. Whatever
  the top ~29 teams (all above 0.84) are doing, it is not explained by
  either of the two hypotheses this project could cheaply test (fine-tuning
  harder, or exploiting overlap harder) — most likely a fundamentally
  different technique or pipeline, not investigated further given exhausted
  GPU quota and the Aug 24 00:00 deadline.
- A follow-up test tried semantic (mBERT mean-pooled embedding) retrieval
  instead of lexical TF-IDF, on the theory that BERTScore (50% of the
  metric) rewards semantic match specifically, which TF-IDF can't target
  directly (`src/retrieval_semantic_check.py`). Result: **0.5762 local**
  composite — statistically the same as word-level TF-IDF (0.5757) and
  slightly below the char n-gram/hybrid variants (0.5817/0.5830), despite
  using a smaller 3,000-row train subsample (vs. the hybrid's full 81,771
  rows) that should have handicapped it further. Lexical and semantic
  retrieval converge on the same ~0.57-0.58 ceiling — confirms this whole
  family of nearest-neighbor techniques is exhausted, not just under-tuned.

**Decision**: the hybrid retrieval submission (0.55756 real, submitted
2026-08-22) is the team's final Phase 1 submission, since it is the
highest-scoring entry produced and it is the last submission chronologically
(rulebook §8 default). 2 of the 5 allowed submissions remain unused. No
further retrieval-tuning is worth pursuing — every variant tried lands in
the same narrow band.

## Retrieval + generation combination — also exhausted

Separately from pure retrieval, three attempts were made at combining
retrieval with the model rather than using either alone, on the theory that
the model could fix retrieval's biggest weakness (returning a real answer to
the wrong case) while keeping its lexical-overlap advantage. All three
landed in the same ~0.55-0.56 band, at or below plain generation:

| technique | local composite | how it was tested |
|---|---|---|
| Day 12: RAG-style ("use as style reference", trained) | 0.5562 | real training run, own ablation vs. no-retrieval (0.5584) |
| Day 13: zero-shot edit prompting (untrained adapter) | 0.5481 | prompted the existing LoRA adapter to edit a retrieved answer, no training for this task |
| Day 14/15: dedicated retrieval+edit LoRA (trained from base) | 0.5496 | ~3h GPU training explicitly supervised on (retrieved example + question) → real answer, then evaluated in the same format |

The Day 14 result is the important one: training the model specifically on
this exact task (not just prompting it) produced a score statistically
indistinguishable from the untrained zero-shot attempt (0.5496 vs. 0.5481).
That rules out "the model just needed to be taught this" — three different
implementations of retrieval-conditioned generation (style-reference
training, zero-shot editing, dedicated edit training) all independently
converge on the same ceiling, below pure retrieval. Combining retrieval with
generation, in every form tried, is worse than either technique alone.

## Ensembling — go/no-go: **NO-GO**

~1.245B of param headroom remains under the 3B cap (enough room, in
principle), but the cheapest form of ensembling was already tried and it
lost:

`ahsanulhoque48cu/nascenia-merged-adapter` — a weight-averaged "model soup"
of the 4 independently-trained shard adapters (the FedAvg-style merge
described under Approach) — scored **0.5590 composite**, *below* the single
best shard's 0.5597. The 4 shards share the same base model, same seed/init,
and overlapping training data, so they're too correlated for averaging to
add useful diversity; it just pulled the best candidate toward the mean of
weaker ones.

Output-space ensembling (generate with multiple adapters per test row, then
vote/select) is untested and could behave differently, but: (a) the
model-soup result is a real negative signal about how correlated these
candidates are, (b) it would need fresh GPU inference time against an
already-strained shared quota (both team-lead Kaggle accounts are
quota/verification-blocked; training and eval runs have been going through
teammate accounts), and (c) it competes directly with finishing
`phase2_package/` and locking the final submission before the Aug 23
last-working-day / Aug 24 00:00 deadline. Per the roadmap's own criterion —
skip if the single model is already solid and time is short — this is a
clear skip. **Decision: no-go, documented here rather than spending
remaining time/quota testing it.**

## Data

Source: competition-provided `train.csv` only. **No external data used** —
no public Bengali medical corpora, no scraped or synthetically generated
training examples.

Cleaning applied (`src/curate_train.py`): removed rows with scrape damage
(truncated URLs, bare stray characters/vowel signs left over from source
HTML). Boilerplate openings were deliberately **kept**, not stripped: 76.3%
of real reference responses open with `হেলো` (a greeting convention in the
source data) — filtering it out would have made training distribution
diverge from the true target distribution the metric is measured against.

## Generation settings at inference

Greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=900`,
`min_new_tokens=650`, `repetition_penalty=1.05`, `no_repeat_ngram_size=6`.
Beam search was deliberately avoided — for long-form decoder-only generation
it systematically prefers shorter hypotheses, which is exactly the wrong
direction here (see `src/infer.py` module docstring for the measured
Token-F1-vs-length curve that motivated the length floor).

## Known limitations

- **Output-opening mode collapse.** The fine-tuned model opens essentially
  every response with `হেলো`, versus 76.3% in the real reference
  distribution — the model over-learned the single most common opening
  token rather than reproducing its true frequency. Content after the
  opening still varies per-input.
- **Slight over-length.** Mean generation length (~712-714 chars on val) runs
  somewhat longer than the reference mean (~634 chars), a deliberate tradeoff
  since Token-F1/ROUGE-L are recall-sensitive under-generation was measured
  as the far more damaging failure mode.
- **Local-vs-real metric calibration gap.** Local composite scores
  consistently overestimate the real leaderboard score by roughly 0.03-0.04.
  Investigated and most likely explained by an undocumented difference in
  tokenizer or BERTScore backbone between our local harness and the official
  scorer (the rulebook fixes the metric *weights* but not those
  implementation choices) rather than by data leakage, which was checked and
  ruled out. Relative deltas between our own candidates remain trustworthy;
  absolute distance to other teams' leaderboard scores should not be
  over-interpreted.
- **A follow-up retrieval-augmented (RAG-style) fine-tuning run (Day 12) did
  not improve on this checkpoint** — its own ablation showed retrieval
  context conditioning made generations *worse* (0.5562 vs 0.5584 composite
  without retrieval), and even in its best mode it scored slightly below
  this checkpoint (0.5584 vs 0.5597) — so it was not submitted. This
  checkpoint remains the best-performing candidate produced.
