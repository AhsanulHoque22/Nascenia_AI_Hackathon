#!/usr/bin/env bash
# One-command setup to run the final scoring+submission kernel on a
# teammate's account (used when the team lead's own accounts are blocked --
# quota exhausted or phone-verification gated).
#
#   ./setup_and_push.sh <your-kaggle-username>
#
# Ranks all 5 real candidates (own chain, shard2, shard3, shard4, merge)
# on held-out val and generates the 1000-row test submission with the
# winner. Requires the kaggle CLI already configured with YOUR OWN token
# (~/.kaggle/kaggle.json) -- this script can't do that part for you.
#
# Needs these 6 datasets shared with your username first (team lead adds
# you as a collaborator: dataset page -> Settings -> Sharing):
#   ahsanulhoque48cu/nascenia-processed-data
#   ahsanulhoque48cu/nascenia-adapter
#   ahsanulhoque48cu/nascenia-merged-adapter
#   ahsanulhoque48cu/nascenia-shard-4-adapter
#   sanzidislam/nascenia-shard-2-adapter
#   sanzid65/nascenia-shard-3-adapter

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

sed -i.bak "s#\"id\": \".*\"#\"id\": \"${USERNAME}/nascenia-day11-select-submit\"#" kernel-metadata.json
rm -f kernel-metadata.json.bak

echo "Checking access to all 6 required datasets..."
missing=0
for ref in \
  ahsanulhoque48cu/nascenia-processed-data \
  ahsanulhoque48cu/nascenia-adapter \
  ahsanulhoque48cu/nascenia-merged-adapter \
  ahsanulhoque48cu/nascenia-shard-4-adapter \
  sanzidislam/nascenia-shard-2-adapter \
  sanzid65/nascenia-shard-3-adapter
do
  if ! kaggle datasets files "$ref" >/dev/null 2>&1; then
    echo "  MISSING ACCESS: $ref"
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "Ask the team lead / the relevant teammate to share the dataset(s)"
  echo "listed above with your username (${USERNAME}), then re-run this script."
  exit 1
fi

echo "All datasets accessible. Pushing scoring kernel as ${USERNAME}..."
kaggle kernels push -p .

cat <<EOF

Pushed. Track it at:
  https://www.kaggle.com/code/${USERNAME}/nascenia-day11-select-submit

This scores 5 candidates on held-out val then generates the 1000-row test
submission with the winner -- estimate ~3.5-4 hours total. It has an 8.5h
internal safety cutoff and saves its ranking incrementally, so it won't
silently lose everything even if it runs long.

When it's done, send the team lead:
  kaggle kernels output ${USERNAME}/nascenia-day11-select-submit -p ./out
The files they need are ./out/submission.csv and ./out/checkpoint_selection.json.
EOF
