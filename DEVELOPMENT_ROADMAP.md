# Development Roadmap — Nascenia AI Hackathon
### Bengali Medical Dialogue Generation

Goal: maximize `Final Score = 0.8 × Phase1Score + 0.2 × Phase2Score` where
`Phase1Score = 0.5×BERTScore + 0.3×TokenF1 + 0.2×ROUGE-L`, under a hard ≤3B
parameter inference constraint.

Today: **Aug 5**. Phase 1 closes **Aug 24, 00:00 BD**. Phase 2 closes **Aug 25, 00:00 BD**. Results **Aug 26**.

Reference: `RuleBook_Nascenia_Hackathon.pdf`, `LEARNING_PLAN.md` (this folder).

---

## Repository structure

Set this up before anything else — every later phase assumes it exists.

```
Hack_A_Thone/
├── data/
│   ├── raw/                # untouched Kaggle download
│   ├── processed/          # cleaned/normalized train+val splits
│   └── external/           # any disclosed external data, kept separate
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
├── src/
│   ├── data_prep.py        # cleaning, normalization, train/val split
│   ├── eval_metrics.py      # local BERTScore/TokenF1/ROUGE-L harness
│   ├── train.py             # fine-tuning entrypoint (LoRA/QLoRA)
│   ├── infer.py             # inference script (doubles as Phase 2 deliverable)
│   └── param_count.py       # verifies model stays under 3B before every submit
├── configs/
│   └── *.yaml               # one config per experiment run
├── experiments/
│   └── <run_id>/            # checkpoints, logs, metrics.json per run
├── submissions/
│   └── kaggle_submission_*.csv
├── phase2_package/
│   ├── inference_script.py
│   ├── download_weights.sh (or weights/ if small enough)
│   ├── writeup.md
│   └── requirements.txt / environment.yml
├── LEARNING_PLAN.md
├── DEVELOPMENT_ROADMAP.md
└── RuleBook_Nascenia_Hackathon.pdf
```

Keep `experiments/<run_id>/metrics.json` for every run — you'll need to
compare runs quickly and pick the best one deliberately (only 5 Kaggle
submissions allowed).

---

## Phase 0 — Environment setup (Day 1, ~half day)

1. Kaggle account + join competition, accept rules, download `data/raw/` via Kaggle API (`kaggle competitions download`).
2. Local/Colab/Kaggle-notebook environment: Python venv or conda, install `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `bert-score`, `rouge-score`, `sentencepiece`, `sacremoses`.
3. Freeze this into `phase2_package/requirements.txt` **now** — you'll need it later anyway, and it documents what you actually used as you go instead of reconstructing it under deadline pressure.
4. Confirm GPU access (Kaggle notebook T4/P100, or Colab) and note session time limits — this bounds how large/how many experiments you can run.
5. Git-init this folder locally (even without a remote) so you can diff/revert experiment configs.

**Deliverable**: working environment, raw data downloaded, empty scaffold committed.

---

## Phase 1 — Data collection & understanding (Day 1-2)

1. Load `train.csv` (or whatever the provided format is): inspect columns, row count, encoding (must be UTF-8 for Bengali).
2. Check for the exact task framing: is it single-turn (one patient prompt → one doctor response) or multi-turn dialogue? This determines your input/output format for the whole pipeline.
3. Look for label noise: empty responses, mismatched language, HTML/markup leftovers, duplicate rows.
4. Confirm what "external data" would even look like for this task (public Bengali medical QA corpora, Indic health datasets) — if you plan to use any, note it now so disclosure isn't an afterthought (rulebook requires disclosure).
5. **Do not touch or peek at private test set contents** — you likely only have train + public test inputs (no public test labels). Treat this as a hard boundary from day one.

**Deliverable**: `notebooks/01_eda.ipynb` with data shape, samples, language/encoding checks, anomaly list.

---

## Phase 2 — Data preprocessing & cleaning (Day 2-3)

1. Unicode normalization for Bengali (NFC normalization, strip zero-width joiners/non-joiners inconsistencies).
2. Deduplicate exact/near-duplicate prompt-response pairs.
3. Filter or fix non-Bengali / code-mixed rows that are clearly broken (garbled encoding), but **keep legitimate Bengali-English code-mixing** (common in medical text — drug names, anatomical terms) since your model must handle it.
4. Length analysis: token length distribution of prompts and responses (using your chosen tokenizer) — sets your max sequence length config and flags outliers to cap/truncate sensibly.
5. Create a held-out **local validation split** from train (e.g. 90/10 or stratified if there's a category field) — this is your proxy for the private leaderboard and is what phase 4's eval harness runs against. Never let this leak into fine-tuning.

**Deliverable**: `src/data_prep.py`, `data/processed/{train,val}.jsonl`, a short data-quality note in the EDA notebook.

---

## Phase 3 — Local evaluation harness (Day 3-4, build before serious modeling)

This has to exist before you trust any model comparison — build it early, not after you have a model you like.

1. Implement `src/eval_metrics.py`:
   - BERTScore (`bert-score` package) — pick a multilingual/Bengali-capable embedding backbone (test 2-3 candidates, e.g. `bert-base-multilingual-cased`, XLM-R based scorer, since default English models under-score Bengali).
   - Token-level F1 (SQuAD-style precision/recall over tokenized overlap) — tokenize consistently with whatever your generation model uses, or a fixed reference tokenizer for stability across model comparisons.
   - ROUGE-L F1 (`rouge-score`, with Bengali-aware tokenization — the default English tokenizer will misbehave on Bengali script).
   - Combine into the exact competition formula: `0.5×BERTScore + 0.3×TokenF1 + 0.2×ROUGE-L`.
2. Sanity-check the harness: run it on (reference vs reference) → should score ~1.0; (reference vs random unrelated Bengali text) → should score low.
3. Wrap it so every future experiment run auto-logs this score against your local validation split into `experiments/<run_id>/metrics.json`.

**Deliverable**: `src/eval_metrics.py`, sanity-check results recorded, reusable scoring function for every later phase.

---

## Phase 4 — Baseline (Day 4-5)

1. Pick 1-2 off-the-shelf ≤3B multilingual/Bengali-capable models, run **zero-shot / few-shot prompting** (no fine-tuning) as your floor.
2. Score against the local validation harness from Phase 3.
3. This gives you (a) a sanity floor to beat, (b) an early check that your eval harness produces sane, differentiable numbers, (c) a first param-count sanity check.

**Deliverable**: `notebooks/02_baseline.ipynb`, baseline score recorded — this is your reference point for every later improvement.

---

## Phase 5 — Model candidate shortlist (Day 5-7)

1. Shortlist 2-4 ≤3B candidates based on real Bengali pretraining coverage (not just "multilingual" marketing) — check tokenizer vocab for Bengali script coverage directly (encode a Bengali medical sentence, check token count vs character count — high fragmentation is a red flag).
2. For each candidate, run `src/param_count.py` to log exact inference-time parameter count **before** investing fine-tuning time in it — leaves headroom for LoRA adapter params, which do count toward the cap.
3. Run the same zero-shot baseline eval (Phase 4 method) on each candidate for an apples-to-apples comparison.
4. Pick 1 primary + 1 backup candidate to carry into fine-tuning.

**Deliverable**: comparison table (model, params, baseline score) in `notebooks/02_baseline.ipynb` or a short `MODEL_SELECTION.md`.

---

## Phase 6 — Fine-tuning pipeline (Day 7-10)

1. Build `src/train.py`: LoRA/QLoRA fine-tuning (start here over full fine-tune — cheaper, faster iteration, easier to stay under param cap since base weights can stay frozen/quantized).
2. Format data as instruction pairs: patient prompt → doctor response, with a consistent prompt template (log the exact template — you'll need it identically at inference and in the Phase 2 write-up).
3. Config-driven runs (`configs/*.yaml`): LoRA rank/alpha, learning rate, epochs, max seq length — one YAML per experiment so runs are reproducible and diffable.
4. Train first run at small scale (subset of data, 1 epoch) purely to validate the pipeline runs end-to-end and produces non-garbage output — don't burn full compute budget on an untested pipeline.
5. Full run once pipeline is validated; checkpoint to `experiments/<run_id>/`.

**Deliverable**: working fine-tuning pipeline, first full fine-tuned checkpoint, local eval score beating baseline.

---

## Phase 7 — Iteration loop (Day 10-16, the bulk of the time budget)

Repeat, tracking every run's config + local score in `experiments/`:

1. Error analysis: pull worst-scoring validation examples, read them — look for patterns (truncated responses, wrong tone, hallucinated diagnoses, tokenization artifacts, code-mixing failures).
2. Iterate on: LoRA hyperparameters, data cleaning fixes found via error analysis, prompt template tweaks, generation config (beam search vs sampling, max new tokens, repetition penalty).
3. Re-run `param_count.py` after every architecture/adapter change — easy to accidentally cross 3B when stacking adapters or changing rank.
4. Keep the backup candidate model warm (at least one more fine-tuning pass) as a hedge — don't over-commit to a single model until you have real comparative numbers.
5. Track local score trend over time; stop iterating on a change if it's not moving the local metric — avoid rabbit-holing on marginal tweaks with limited days left.

**Deliverable**: a ranked list of experiment runs by local score, best checkpoint identified.

---

## Phase 8 — Optional: ensembling / quantization (Day 16-18, only if headroom remains)

1. If your best single model leaves meaningful room under the 3B cap, evaluate whether a second small model ensembled (e.g. averaging/voting or a lightweight reranker) improves local score enough to justify the added inference complexity.
2. Post-training quantization (int8/int4) if it helps you fit a larger/better base model under budget without hurting quality much — re-validate local score after quantizing, don't assume it's free.
3. Skip this phase entirely if the single fine-tuned model is already solid and time is short — a working simple submission beats an unfinished complex one.

**Deliverable**: go/no-go decision on ensembling, documented either way.

---

## Phase 9 — Kaggle submission strategy (ongoing from Day ~10, finalized by Day 18)

You get **5 submissions total** for Phase 1 — spend them deliberately:
1. Submission 1 (~Day 10-12): first fine-tuned model, mainly to confirm your local eval harness correlates with the actual public leaderboard score (compare local score vs leaderboard score — if wildly off, your harness has a bug worth fixing before spending more submissions).
2. Submissions 2-4 (~Day 14-20): your best iterations as they clear local-score improvements, spaced out rather than burned early.
3. Submission 5 (~Day 22-23, before the Aug 24 deadline): final best candidate, selected explicitly as your final submission on Kaggle (rulebook: final rank uses the last/selected submission — don't leave this ambiguous, actively mark your intended final).
4. Leave at least 1-2 days of buffer before Aug 24 00:00 for the leaderboard-confirmation step — don't submit your only real attempt on the last day.

**Deliverable**: submission log (date, run_id, local score, leaderboard score) so you can pick the final one with evidence, not guesswork.

---

## Phase 10 — Phase 2 prep (start early, don't wait for top-10 announcement — Day ~18-22)

Prepare this **before** you know if you made top 10, since the Phase 2 deadline (Aug 25) is only ~1 day after Phase 1 closes (Aug 24):

1. `phase2_package/inference_script.py`: clean, documented, runs end-to-end from raw input to prediction — this should just be `src/infer.py` finalized, not written from scratch under time pressure.
2. `phase2_package/` weights: either the checkpoint itself (if small enough to package) or a reproducible download script (e.g. pulling from Hugging Face Hub if you pushed your fine-tuned adapter there).
3. `phase2_package/writeup.md`: approach, base model + exact params, fine-tuning method (LoRA config), any external data/tools used (must match earlier disclosure), known limitations.
4. `phase2_package/requirements.txt` / `environment.yml`: exact dependency versions used for the winning run — freeze from your actual training environment, don't hand-write from memory.
5. Verify reproducibility yourself: run the packaged inference script from a clean environment and confirm it reproduces your leaderboard-submitted outputs within tolerance — this is exactly what organizers will check in Section 5.2, so catch mismatches yourself first.

**Deliverable**: complete, tested `phase2_package/` ready to submit the moment top 10 is announced.

---

## Phase 11 — Final checks & submission (Day 22-24)

1. Re-run `param_count.py` one last time on the exact final checkpoint being submitted.
2. Ethics/safety spot-check: sample a batch of generated responses, manually read for harmful/unsafe medical advice per Section 11 — this can get you penalized in Phase 2 judging even if Phase 1 score is strong.
3. Confirm the selected-as-final submission on Kaggle is actually your intended best run.
4. If top 10 is reached: submit the `phase2_package/` before Aug 25, 00:00 BD.

**Deliverable**: final Phase 1 submission locked in; Phase 2 package submitted if applicable.

---

## Timeline summary

| Days | Phase | Focus |
|---|---|---|
| 1 | 0 | Environment + repo setup |
| 1-2 | 1 | Data collection & understanding |
| 2-3 | 2 | Data cleaning & local val split |
| 3-4 | 3 | Local eval harness (BERTScore/TokenF1/ROUGE-L) |
| 4-5 | 4 | Baseline (zero/few-shot) |
| 5-7 | 5 | Model candidate shortlist + param counting |
| 7-10 | 6 | Fine-tuning pipeline (LoRA/QLoRA) built + first run |
| 10-16 | 7 | Iteration loop (bulk of the work) |
| 16-18 | 8 | Optional ensembling/quantization |
| 10-23 | 9 | Kaggle submissions (5 total, spaced deliberately) |
| 18-22 | 10 | Phase 2 package prep (in parallel with late Phase 7-9) |
| 22-24 | 11 | Final checks, lock submission |
| Aug 24 | — | Phase 1 closes |
| Aug 25 | — | Phase 2 closes (if top 10) |
| Aug 26 | — | Results |

---

## Rules-compliance checklist (check before every submission)

- [ ] Model param count at inference ≤ 3,000,000,000 (verified via `param_count.py`, including any adapters)
- [ ] No private test set inputs used in training/fine-tuning
- [ ] No manually-labeled outputs submitted in place of model generations
- [ ] Any external data used is disclosed in submission notes
- [ ] Local validation split never leaked into training data
- [ ] Generated outputs spot-checked for harmful/unsafe medical content
