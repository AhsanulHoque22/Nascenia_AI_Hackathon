# Nascenia AI Hackathon — Learning Plan

**Task**: Bengali Medical Dialogue Generation (Kaggle) — generate a doctor's Bengali response to a patient's prompt, model ≤3B params at inference.

**Timeline**: Today Aug 5 → Phase 1 closes Aug 24, 00:00 BD time → Phase 2 closes Aug 25 → results Aug 26.

Full rules: `RuleBook_Nascenia_Hackathon.pdf` in this folder.

---

## 1. Core ML/DL foundations
- Python for ML (NumPy, pandas), PyTorch basics (tensors, autograd, training loop)
- Neural net fundamentals: backprop, loss functions, optimizers (Adam/AdamW), learning rate schedules
- Overfitting/regularization, train/val/test splits, cross-validation

## 2. NLP & Transformer fundamentals
- Tokenization (BPE, SentencePiece, WordPiece) — critical since Bengali tokenizes very differently than English
- Embeddings, attention mechanism, self-attention vs cross-attention
- Transformer architecture: encoder-decoder (T5/BART-style) vs decoder-only (GPT-style) — you'll choose one paradigm for this task
- Seq2seq generation basics: greedy/beam search, sampling (top-k, top-p/nucleus, temperature), causal LM vs conditional generation

## 3. Hugging Face ecosystem (your main toolkit)
- `transformers`: loading models/tokenizers, `Trainer`/`Seq2SeqTrainer` API, `generate()` config
- `datasets`: loading/preprocessing Kaggle CSVs into HF Dataset objects, batching, dynamic padding
- `accelerate`: multi-GPU/mixed-precision training abstraction
- `peft`: LoRA/QLoRA adapters — the single most important library for staying under the 3B cap while still getting good fine-tuning gains
- `bitsandbytes`: 4-bit/8-bit quantization for training and inference

## 4. Fine-tuning strategy (competition allows all of these — know the tradeoffs)
- Full fine-tuning vs LoRA vs QLoRA (rank, alpha, target modules)
- Instruction/SFT-style fine-tuning: formatting patient prompt → doctor response as instruction pairs
- Prompt engineering as a fallback/complement (few-shot prompting a base model, since it's explicitly permitted)
- Post-training quantization vs distillation (both permitted, only matters if you need extra headroom under 3B)
- Basics of ensembling small models under a combined param budget

## 5. Model selection (the ≤3B constraint is a hard boundary — research this early)
- Candidates worth evaluating for Bengali capability at ≤3B: multilingual encoder-decoders (mT5-small/base, mBART), Bengali-specific models (BanglaT5, BanglaBERT, IndicBART, IndicTrans2), and small multilingual decoder LLMs (Gemma 2B, Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, SmolLM) — check their actual Bengali pretraining coverage, not just "multilingual" claims
- How to count parameters precisely at inference time (including embedding/LM-head sharing, adapter overhead) so you don't accidentally bust the cap after adding LoRA weights

## 6. Bengali-language specifics (low-resource language handling)
- Bengali Unicode normalization (conjuncts, zero-width joiners — common source of silent bugs)
- Tokenizer vocabulary coverage for Bengali script (a tokenizer trained mostly on Latin scripts will badly over-fragment Bengali, hurting both quality and effective context length)
- Code-mixing: Bengali medical text often mixes in English drug/anatomy terms — your model needs to handle that gracefully
- Low-resource fine-tuning tricks: data augmentation, back-translation, transfer from related Indic languages

## 7. Medical dialogue domain knowledge
- Structure of doctor-patient dialogue (triage tone, follow-up questions, appropriate hedging vs overconfidence)
- Basic medical terminology in Bengali (enough to sanity-check outputs, not clinical expertise)
- Hallucination risks in medical generation and why Phase 2's "clinically appropriate" judge criterion will penalize confident-but-wrong answers
- The rulebook's ethics clause (Section 11) — outputs must avoid harmful/unsafe generated advice

## 8. The exact evaluation metrics (build a local harness that mirrors these before submitting)
- **BERTScore** (50% weight): contextual embeddings + cosine similarity — need to know which underlying model computes it for Bengali reasonably, since that affects what "good" scores look like
- **Token-level F1** (30%): precision/recall over token overlap (SQuAD-style) — sensitive to tokenizer segmentation, so Bengali tokenization choices directly move this number
- **ROUGE-L** (20%): longest common subsequence based — rewards structural/ordering similarity to reference
- Implement all three locally (`bert-score`, `rouge-score` packages) so you can validate before burning Kaggle submissions (only 5 allowed)

## 9. LLM-as-judge mechanics (Phase 2, only matters if you place top 10)
- How LLM-as-judge scoring works, common biases (verbosity bias, position bias) and why the rulebook uses blind + randomized order
- Writing a response so a judge model scores it well on "clinically appropriate" + "tone/completeness/clarity" — a distinct skill from optimizing Phase 1's string-similarity metrics, and the two can pull in different directions

## 10. Training infrastructure & efficiency
- Kaggle notebook GPU constraints (T4/P100 quotas, session time limits) — plan compute budget
- Mixed precision (fp16/bf16), gradient checkpointing, gradient accumulation for small-GPU fine-tuning
- Reproducibility: seeding, deterministic runs (needed for Phase 2's "reproduces leaderboard outputs" requirement)

## 11. Competition mechanics
- Kaggle submission format, notebook-based vs file-based submission for this competition
- Reading the leaderboard correctly (public vs private split — real target is private leaderboard rank)
- Preparing Phase 2 deliverables early: inference script, weights/download script, write-up, environment/dependency file (`requirements.txt` or similar) — don't leave this to the last day

---

## Suggested order given the timeline
1. **Days 1-3**: Transformer/tokenization fundamentals (if rusty) + HF ecosystem basics + set up local eval harness (BERTScore/TokenF1/ROUGE-L) — needed before judging any model
2. **Days 3-6**: Model survey — shortlist 2-3 ≤3B candidates with real Bengali coverage, run baseline (zero-shot/few-shot) to get a floor score
3. **Days 6-14**: LoRA/QLoRA fine-tuning iterations, Bengali-specific data cleaning, error analysis against local metrics
4. **Days 14-18**: Ensembling/quantization experiments if headroom remains under 3B, submission hygiene (use the 5 Kaggle submissions deliberately, not early)
5. **Days 18-19**: Freeze best entry, pre-write the Phase 2 write-up and environment file so there's no scramble if top 10 is reached
