#!/usr/bin/env bash
# One-command setup for a teammate's shard. Run from inside this folder.
#
#   ./setup_and_push.sh <your-kaggle-username> <shard-index>
#
# example: ./setup_and_push.sh jane_doe 2
#
# Does exactly the two edits TEAMMATE_SETUP.md describes by hand, then pushes.
# Requires the kaggle CLI already configured with YOUR OWN token
# (~/.kaggle/kaggle.json) — see TEAMMATE_SETUP.md step 2 for that part; this
# script can't do it for you since it's your credential.

set -euo pipefail

USERNAME="${1:-}"
SHARD="${2:-}"

if [ -z "$USERNAME" ] || [ -z "$SHARD" ]; then
  echo "usage: $0 <your-kaggle-username> <shard-index (2, 3, or 4)>"
  exit 1
fi
if ! [[ "$SHARD" =~ ^[2-4]$ ]]; then
  echo "shard-index must be 2, 3, or 4 — check with the team lead which one is yours"
  exit 1
fi

if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "No Kaggle credentials found (~/.kaggle/kaggle.json missing, KAGGLE_API_TOKEN unset)."
  echo "Get your token: Kaggle -> profile -> Settings -> API -> Create New Token"
  echo "Then put the downloaded file at ~/.kaggle/kaggle.json and re-run this script."
  exit 1
fi

sed -i.bak "s/^SHARD_INDEX = .*/SHARD_INDEX = ${SHARD}   # set by setup_and_push.sh/" script.py
sed -i.bak "s#\"id\": \".*\"#\"id\": \"${USERNAME}/nascenia-day11-shard-${SHARD}\"#" kernel-metadata.json
# "title" must also match, or Kaggle can reject the push with a 409 conflict
# (found the hard way: a stale placeholder title caused exactly this).
sed -i.bak "s#\"title\": \".*\"#\"title\": \"nascenia-day11-shard-${SHARD}\"#" kernel-metadata.json
rm -f script.py.bak kernel-metadata.json.bak

echo "Checking access to the training dataset..."
# `datasets status` only resolves for datasets you own -- it 404s even with
# valid shared access. `datasets files` actually requires read access to
# succeed, so it's the real check here.
if ! kaggle datasets files ahsanulhoque48cu/nascenia-processed-data >/dev/null 2>&1; then
  echo "Can't access the dataset yet. Has the team lead shared"
  echo "ahsanulhoque48cu/nascenia-processed-data with your Kaggle username?"
  exit 1
fi

echo "Pushing shard ${SHARD} as ${USERNAME}..."
kaggle kernels push -p .

cat <<EOF

Pushed. Track it at:
  https://www.kaggle.com/code/${USERNAME}/nascenia-day11-shard-${SHARD}

Takes about 7 hours. When it's done, run this to send the adapter back:

  kaggle kernels output ${USERNAME}/nascenia-day11-shard-${SHARD} -p ./out
  mkdir -p ./out/run/adapter_final
  cat > ./out/run/adapter_final/dataset-metadata.json <<META
{"title": "nascenia-shard-${SHARD}-adapter", "id": "${USERNAME}/nascenia-shard-${SHARD}-adapter", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}
META
  kaggle datasets create -p ./out/run/adapter_final

Then share that new dataset with the team lead's Kaggle username
(dataset page -> Settings -> Sharing -> Add collaborator). That's it.
EOF
