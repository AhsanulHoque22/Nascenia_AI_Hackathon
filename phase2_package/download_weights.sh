#!/usr/bin/env bash
# Downloads the exact winning LoRA adapter (base model is pulled automatically
# by transformers from Qwen/Qwen3-1.7B on first run of inference_script.py).
#
# Requires Kaggle API credentials (~/.kaggle/kaggle.json) with access to the
# adapter dataset below. The dataset is private; ask the team for a
# collaborator invite if this 403s.
set -euo pipefail

ADAPTER_DATASET="sanzidislam/nascenia-shard-2-adapter"  # winning candidate, local composite 0.5597, real leaderboard 0.52418
OUT_DIR="weights/adapter"

mkdir -p "$OUT_DIR"
kaggle datasets download -d "$ADAPTER_DATASET" -p "$OUT_DIR" --unzip

echo "adapter downloaded to $OUT_DIR"
echo "base model (Qwen/Qwen3-1.7B) downloads automatically on first inference_script.py run"
