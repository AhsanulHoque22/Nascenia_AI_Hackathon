# Daily Plan — Nascenia AI Hackathon (Learning + Development, merged)

Today: **Aug 6**. Phase 1 deadline is **Aug 24, 00:00 BD** — meaning your last real
working day is **Aug 23**. Phase 2 deadline **Aug 25, 00:00** (only ~24h after
Phase 1 closes, if you make top 10). Results **Aug 26**.

This merges `LEARNING_PLAN.md` and `DEVELOPMENT_ROADMAP.md` into one schedule:
each day's **Learn** block covers only what that day's **Build** block needs —
learn it, then immediately use it. No topic is studied more than ~1 day before
it's applied.

Assume ~1–1.5h learning + ~3–5h building per day. Adjust hours to your actual
availability, but keep the *order* — it's built so nothing blocks on something
not yet learned.

---

## Week 1 — Foundations, data, eval harness, baseline

### Day 1 — Aug 6 (Thu)
- **Learn**: Skim Transformer basics if rusty (attention, encoder-decoder vs
  decoder-only) — 30-60min refresher, skip if solid already.
- **Build**: Phase 0 setup — Kaggle account/join competition/accept rules,
  download `data/raw/`, create the repo scaffold (see `DEVELOPMENT_ROADMAP.md`
  structure), venv + install `transformers datasets peft accelerate
  bitsandbytes bert-score rouge-score sentencepiece sacremoses`, freeze
  `requirements.txt` now, `git init`.
- **Deliverable**: working env, raw data downloaded, empty scaffold committed.

### Day 2 — Aug 7 (Fri)
- **Learn**: Tokenization (BPE/SentencePiece/WordPiece) — why this matters
  disproportionately for Bengali. HF `transformers`/`datasets` basics (loading
  models/tokenizers, `Dataset` objects).
- **Build**: Phase 1 — load train data, inspect columns/encoding/row count,
  confirm single-turn vs multi-turn framing, scan for label noise, note
  external-data disclosure plan.
- **Deliverable**: `notebooks/01_eda.ipynb`.

### Day 3 — Aug 8 (Sat)
- **Learn**: Bengali Unicode normalization (NFC, conjuncts, zero-width
  joiners), code-mixing handling (Bengali+English drug/anatomy terms).
- **Build**: Phase 2 — normalize text, dedupe, filter garbled rows (keep
  legit code-mixing), token-length analysis, carve out local val split
  (never leaks into training).
- **Deliverable**: `src/data_prep.py`, `data/processed/{train,val}.jsonl`.

### Day 4 — Aug 9 (Sun)
- **Learn**: How BERTScore, Token-F1, and ROUGE-L are actually computed —
  what embedding backbone BERTScore needs for Bengali, why tokenizer choice
  moves Token-F1, why ROUGE-L rewards ordering.
- **Build**: Phase 3 — implement `src/eval_metrics.py` combining
  `0.5×BERTScore + 0.3×TokenF1 + 0.2×ROUGE-L`. Sanity check: ref-vs-ref ≈1.0,
  ref-vs-random ≈low.
- **Deliverable**: `src/eval_metrics.py`, sanity results logged.

### Day 5 — Aug 10 (Mon)
- **Learn**: Generation strategies — greedy/beam search, top-k/top-p
  sampling, temperature; causal LM vs conditional generation.
- **Build**: Phase 4 — pick 1-2 off-the-shelf ≤3B multilingual/Bengali
  models, run zero-shot/few-shot prompting, score with your Day 4 harness.
  This is your floor + a live check the harness produces sane numbers.
- **Deliverable**: `notebooks/02_baseline.ipynb`, baseline score recorded.

### Day 6 — Aug 11 (Tue)
- **Learn**: Model landscape for ≤3B Bengali capability — BanglaT5,
  BanglaBERT, IndicBART, IndicTrans2, mT5-small/base, mBART, Gemma-2B,
  Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, SmolLM. Check tokenizer fragmentation on
  real Bengali text (token count vs char count) as a first filter.
- **Build**: Phase 5 — shortlist 2-4 candidates, run `param_count.py` on
  each (verify ≤3B including any adapter overhead), run same zero-shot eval
  for apples-to-apples comparison.
- **Deliverable**: comparison table (model, params, baseline score) —
  `MODEL_SELECTION.md` or in the baseline notebook. Pick 1 primary + 1 backup.

### Day 7 — Aug 12 (Wed)
- **Learn**: PEFT — LoRA/QLoRA mechanics (rank, alpha, target modules), why
  this is the main lever for staying under 3B while still getting real gains.
  Instruction/SFT formatting (prompt → response pairs, template design).
- **Build**: Phase 6 start — build `src/train.py` (LoRA/QLoRA), format data
  as instruction pairs, lock a prompt template (write it down — you'll need
  it identically at inference and in the Phase 2 write-up).
- **Deliverable**: `src/train.py` skeleton, prompt template documented.

---

## Week 2 — Fine-tuning + iteration (bulk of the time budget)

### Day 8 — Aug 13 (Thu)
- **Learn**: Mixed precision (fp16/bf16), gradient checkpointing/
  accumulation for small-GPU fine-tuning; Kaggle T4/P100 session limits —
  plan your compute budget around them.
- **Build**: Config-driven runs (`configs/*.yaml`). Train a *small-scale*
  run (data subset, 1 epoch) purely to validate the pipeline runs end-to-end
  and outputs non-garbage — don't burn compute on an untested pipeline.
- **Deliverable**: validated pipeline, first sane (if weak) generations.

### Day 9 — Aug 14 (Fri)
- **Build**: First full fine-tuning run on primary candidate. Checkpoint to
  `experiments/<run_id>/`, log local score.
- **Deliverable**: first full checkpoint beating the Day 5 baseline.

### Day 10 — Aug 15 (Sat)
- **Learn**: Error-analysis technique — what patterns to look for (truncated
  responses, wrong tone, hallucinated diagnoses, tokenization artifacts,
  code-mixing failures).
- **Build**: **Kaggle submission #1** (confirm local score correlates with
  leaderboard score — fix the harness now if wildly off, before spending more
  submissions). Start error analysis on worst-scoring val examples.
- **Deliverable**: submission #1 logged (local vs LB score), error-analysis
  notes.

### Day 11 — Aug 16 (Sun)
- **Build**: Iteration loop — apply data-cleaning fixes found in error
  analysis, retrain, re-score. Re-run `param_count.py` after any change.

### Day 12 — Aug 17 (Mon)
- **Build**: Iteration — LoRA hyperparameter sweep (rank/alpha/lr/epochs),
  track every run's config + score in `experiments/`.

### Day 13 — Aug 18 (Tue)
- **Build**: Iteration — generation config tuning (beam vs sampling, max new
  tokens, repetition penalty). If a run clears a real local-score
  improvement, use **Kaggle submission #2**.

### Day 14 — Aug 19 (Wed)
- **Learn**: Quantization vs distillation vs ensembling basics — only so
  you can make an informed go/no-go later, not to use yet.
- **Build**: Give the backup candidate a fine-tuning pass (hedge). Keep
  iterating on primary if it's still improving; stop tweaks that aren't
  moving the local metric.

### Day 15 — Aug 20 (Thu)
- **Build**: Continue iteration on whichever candidate (primary/backup) is
  ahead. If improved, **Kaggle submission #3**.

---

## Week 3 — Polish, Phase 2 prep, lock submission

### Day 16 — Aug 21 (Fri)
- **Learn**: LLM-as-judge mechanics for Phase 2 (verbosity/position bias,
  what "clinically appropriate" + tone/completeness/clarity judging rewards —
  distinct from Phase 1's string-similarity metrics).
- **Build**: Decide go/no-go on ensembling/quantization (only if headroom
  under 3B and time permits — skip if the single model is already solid).
  Start `phase2_package/`: finalize `src/infer.py` → `inference_script.py`.

### Day 17 — Aug 22 (Sat)
- **Build**: Finish `phase2_package/`: weights or HF Hub download script,
  `writeup.md` (approach, exact param count, LoRA config, disclosed external
  data, known limitations), frozen `requirements.txt`/`environment.yml`.
  Verify reproducibility from a clean environment now, not later. If a run
  improved, **Kaggle submission #4**.

### Day 18 — Aug 23 (Sun) — **last working day before Phase 1 closes**
- **Build**: Final polish pass. Ethics/safety spot-check a batch of
  generations (Section 11 — no harmful/unsafe medical advice). Re-run
  `param_count.py` on the exact final checkpoint. Select and mark your
  intended **final Kaggle submission (#5)** explicitly — don't leave it
  ambiguous. Leave real buffer before midnight, don't submit your only real
  attempt at the last minute.
- **Deliverable**: final Phase 1 submission locked in before Aug 24 00:00.

### Day 19 — Aug 24 (Mon) — buffer / Phase 2 finalize
- Phase 1 is closed. Use today to double-check `phase2_package/` end-to-end
  (clean-environment run reproducing your submitted outputs) regardless of
  whether top-10 is announced yet — you only have until Aug 25 00:00 to act
  if you make it.

### Day 20 — Aug 25 (Tue)
- If top 10: submit `phase2_package/` before 00:00. Otherwise, nothing due.

### Aug 26 — Results.

---

## Rules-compliance checklist (recheck before every submission)
- [ ] Model param count at inference ≤ 3,000,000,000 (incl. adapters)
- [ ] No private test set inputs used in training/fine-tuning
- [ ] No manually-labeled outputs submitted in place of model generations
- [ ] Any external data used is disclosed
- [ ] Local validation split never leaked into training data
- [ ] Generated outputs spot-checked for harmful/unsafe medical content
