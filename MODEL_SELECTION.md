# Day 6 — Model Shortlist & Selection

All numbers are zero-shot composite scores (`0.5*BERTScore + 0.3*TokenF1 + 0.2*ROUGE-L`)
on the same 15 val examples (seed=42), same system prompt, greedy decoding.
Day 5's Qwen2.5-0.5B ran locally (CPU); Day 6's candidates ran on Kaggle
(T4 GPU) via `kaggle_kernels/model_shortlist/`.

| Model | Params | Under 3B cap? | Tokens/char (bn) | Composite | Notes |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.494B | Yes | - | 0.4409 | Day 5 floor. Often incoherent/non-answers. |
| **Qwen2.5-1.5B-Instruct** | **1.544B** | **Yes** | 1.060 | **0.4979** | Best real candidate. Coherent, on-topic. |
| Qwen2.5-3B-Instruct | 3.086B | **No** | 1.060 | 0.5050 | Disqualified — over the 3B cap by ~86M params. Marginal gain over 1.5B anyway. |
| google/gemma-2-2b-it | 2.614B | Yes | - | - | Blocked: gated on HF, needed a personal token, HF required password re-confirmation to generate one — stopped rather than handle a password. Not evaluated. |
| sarvamai/sarvam-2b-v0.5 | 2.509B | Yes | 0.289 | 0.2845 | **Broken as tested** — ignored the Bengali prompt entirely and generated unrelated generic English text (looked like base-model free-association, not instruction-following) for every example. Token F1 and ROUGE-L were exactly 0.0. Likely a chat-template/prompt-format mismatch for this checkpoint, not necessarily a dead end, but not worth further debugging time right now. |

## Decision

**Primary candidate: Qwen2.5-1.5B-Instruct.**
- Best real composite score (0.4979) among viable (under-cap) candidates.
- Coherent, on-topic Bengali generations (see
  `kaggle_kernels/model_shortlist/output/day6_Qwen_Qwen2.5-1.5B-Instruct.csv`
  for the raw predictions).
- Leaves meaningful headroom under the 3B cap for LoRA adapter overhead.
- Well-documented, actively maintained, standard `transformers`/`peft`
  fine-tuning support.

**Backup candidate: Qwen2.5-0.5B-Instruct.**
- Already validated end-to-end (Day 5). Fallback if 1.5B fine-tuning runs
  into compute/time constraints on Kaggle's session limits.

**Not used further right now:**
- Qwen2.5-3B-Instruct — hard-disqualified by the parameter cap; scaling
  within the Qwen family gave only +0.007 composite over 1.5B anyway, not
  worth chasing given the disqualification.
- Gemma-2-2b-it — access blocked (see notes above). Could revisit later if
  there's spare time and the user completes the HF token step themselves.
- sarvam-2b-v0.5 — technically under cap and specifically Indic-focused,
  but broken in zero-shot form as tested. Could revisit with a corrected
  prompt template if Qwen2.5-1.5B fine-tuning underperforms.

## Competitive context

Kaggle leaderboard top scores are currently 0.87-0.90 (checked Aug 15). All
zero-shot numbers above are far below that — as expected, since fine-tuning
(not model choice) is what will close most of the gap. Day 7+ starts
fine-tuning Qwen2.5-1.5B-Instruct.

## Infra notes (for future Kaggle kernel runs)

- Dataset mount path inside a kernel is
  `/kaggle/input/datasets/<username>/<dataset-slug>/`, not
  `/kaggle/input/<dataset-slug>/` — the shorter path does not exist.
- Push a kernel only after `kaggle datasets status <ref>` reports `ready`
  (a freshly-created dataset needs a few seconds to process; pushing too
  early mounts an empty/missing directory).
- `bert-score` and `rouge-score` are not preinstalled on Kaggle's default
  image — install them at the top of the script.
- Kaggle can assign either a T4 or a P100 GPU. The currently preinstalled
  PyTorch build has dropped P100 (sm_60) kernel support, which fails with
  "no kernel image is available for execution on the device". Force
  `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json` to avoid
  this. Use `float16`, not `bfloat16`, for T4 compatibility.
