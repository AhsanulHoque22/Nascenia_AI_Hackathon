#!/usr/bin/env bash
# One-command setup for the final RAG training run, on a teammate's account
# (using their available GPU quota since the team lead's accounts are used up).
#
#   ./setup_and_push.sh <your-kaggle-username>
#
# Requires the kaggle CLI already configured with YOUR OWN token
# (~/.kaggle/kaggle.json) and access to ahsanulhoque48cu/nascenia-processed-data
# (should already be shared from earlier shard training -- if not, ask the
# team lead to share it via Settings -> Sharing -> Add collaborator).

set -euo pipefail

USERNAME="${1:-}"
if [ -z "$USERNAME" ]; then
  echo "usage: $0 <your-kaggle-username>"
  exit 1
fi

if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "No Kaggle credentials found (~/.kaggle/kaggle.json missing, KAGGLE_API_TOKEN unset)."
  echo "Get your token: Kaggle -> profile -> Settings -> API -> Create New Token"
  echo "Then put the downloaded file at ~/.kaggle/kaggle.json and re-run this script."
  exit 1
fi

sed -i.bak "s#\"id\": \".*\"#\"id\": \"${USERNAME}/nascenia-day12-train-rag\"#" kernel-metadata.json
sed -i.bak "s#\"title\": \".*\"#\"title\": \"nascenia-day12-train-rag\"#" kernel-metadata.json
rm -f kernel-metadata.json.bak

echo "Checking access to the training dataset..."
if ! kaggle datasets files ahsanulhoque48cu/nascenia-processed-data >/dev/null 2>&1; then
  echo "Can't access the dataset yet. Has the team lead shared"
  echo "ahsanulhoque48cu/nascenia-processed-data with your Kaggle username?"
  exit 1
fi

echo "Pushing RAG training kernel as ${USERNAME}..."
kaggle kernels push -p .

cat <<EOF

Pushed. Track it at:
  https://www.kaggle.com/code/${USERNAME}/nascenia-day12-train-rag

This is the FINAL training run -- estimated ~5-6h based on the sized-down
7000-row retrieval-augmented dataset, with an 8.5h hard safety cutoff.

When it's done, send the team lead:
  kaggle kernels output ${USERNAME}/nascenia-day12-train-rag -p ./out
  mkdir -p ./out/run/adapter_final
  cat > ./out/run/adapter_final/dataset-metadata.json <<META
{"title": "nascenia-rag-adapter", "id": "${USERNAME}/nascenia-rag-adapter", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}
META
  kaggle datasets create -p ./out/run/adapter_final

Then share that new dataset with the team lead's Kaggle username
(dataset page -> Settings -> Sharing -> Add collaborator).
EOF
