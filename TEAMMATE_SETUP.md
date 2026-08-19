# Training one shard on your own Kaggle account

You're training **one independent piece** of the team's model, in parallel
with the other teammates, so we finish the full dataset in ~7 hours of wall
clock instead of ~35. Each of you gets a different, non-overlapping slice —
no coordination needed once you've got your `SHARD_INDEX`.

## 0. Which shard is yours?

The team lead will tell you your number: **2**, **3**, or **4**.

## 1. Get access to the training data (one-time)

Ask the team lead to share the `nascenia-processed-data` Kaggle dataset with
your Kaggle username: Dataset page → **Settings → Sharing → Add
collaborator**, Viewer access is enough.

## 2. Get your Kaggle API token

Kaggle → your profile picture → **Settings** → **API** → **Create New
Token**. This downloads `kaggle.json`. Put it at `~/.kaggle/kaggle.json`
(Linux/Mac) — Kaggle CLI reads it automatically. Then:

```bash
pip install kaggle
kaggle datasets status ahsanulhoque48cu/nascenia-processed-data
# should print "ready" — if it errors, step 1 hasn't gone through yet
```

## 3. Get the training script

Either clone the team repo, or just grab these two files from the team lead:
- `kaggle_kernels/day11_train_teammate/script.py`
- `kaggle_kernels/day11_train_teammate/kernel-metadata.json`

Put them in a folder together, e.g. `my_shard/`.

## 4. Edit exactly two things

**`script.py`**, near the top:
```python
SHARD_INDEX = 2   # <- change to YOUR assigned number (2, 3, or 4)
```

**`kernel-metadata.json`**:
```json
"id": "your-kaggle-username/nascenia-day11-shard-2"
```
Replace `your-kaggle-username` with your actual username, and the `2` at the
end with your shard number. Nothing else in either file needs to change —
the LoRA config and random seed must stay identical across everyone's copy
for the results to merge correctly later.

## 5. Push it

```bash
cd my_shard
kaggle kernels push -p .
```

Check progress at `https://www.kaggle.com/code/your-username/nascenia-day11-shard-N`,
or:
```bash
kaggle kernels status your-username/nascenia-day11-shard-N
```

Takes about **7 hours**. It self-stops at 8.5h even if something runs long,
so it won't blow past your session limit.

## 6. When it finishes: send the adapter back

```bash
kaggle kernels output your-username/nascenia-day11-shard-N -p ./out
kaggle datasets create -p ./out/run/adapter_final
```

This creates a new **private** Kaggle dataset under your account containing
just the adapter (~140MB, not the training data). Share *that* dataset back
with the team lead's Kaggle username the same way as step 1 (Settings →
Sharing → Add collaborator).

Tell the team lead it's ready — that's the whole handoff.

## If something errors

Send the team lead:
```bash
kaggle kernels status your-username/nascenia-day11-shard-N
kaggle kernels output your-username/nascenia-day11-shard-N -p ./errored
```
and the `.log` file from `./errored/`. Don't try to fix and re-push blind —
GPU sessions are quota-limited, so a wasted 7-hour retry on a guess is
expensive.
