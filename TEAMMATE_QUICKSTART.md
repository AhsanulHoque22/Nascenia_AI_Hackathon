# Quickstart — train your shard in 5 minutes of actual work

This is the fast path. Read `TEAMMATE_SETUP.md` only if something here breaks
or you want the full explanation of what's happening.

You do 3 things. Then you wait ~7 hours (Kaggle trains in the background —
close your laptop, doesn't matter). Then you do 1 more thing.

## What you need before starting

- A Kaggle account (free)
- Your **shard number** (2, 3, or 4) — ask the team lead, don't guess
- Python installed (`python3 --version` or `python --version` — if that
  errors, install Python first, anything 3.9+ works)

## Step 1 — Get your Kaggle token (2 min)

1. Go to kaggle.com → click your profile picture (top right) → **Settings**
2. Scroll to **API** → click **Create New Token**
3. A file called `kaggle.json` downloads. Move it to:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<you>\.kaggle\kaggle.json`
     (create the `.kaggle` folder if it doesn't exist)

4. Install the CLI:
   ```bash
   pip install kaggle
   ```
   (Windows: same command, in Command Prompt or PowerShell)

## Step 2 — Ask the team lead to share the dataset with you

Tell them your Kaggle **username**. They'll add you as a Viewer on
`ahsanulhoque48cu/nascenia-processed-data`. Without this, everything below
fails at the "checking access" step.

## Step 3 — Run one script

Get `kaggle_kernels/day11_train_teammate/` from the team lead (repo clone,
zip, whatever's easiest), then from inside that folder:

```bash
./setup_and_push.sh <your-kaggle-username> <your-shard-number>
```

Windows without a bash shell (no WSL/Git Bash): use the manual 2-edit version
in `TEAMMATE_SETUP.md` steps 4-5 instead — same result, just by hand.

If it prints "Pushed. Track it at: ..." — you're done for now. Go do
something else for 7 hours.

## Step 4 — When it's done, send the adapter back

Check progress at the URL it printed, or:
```bash
kaggle kernels status <your-username>/nascenia-day11-shard-<N>
```
Once it says `COMPLETE`, run the block of commands the script printed at the
end of Step 3 (it already has your username/shard number filled in — just
copy-paste). Last step is sharing the resulting dataset with the team lead's
Kaggle username. That's it — no message needed, automation picks it up
within the hour.

## Troubleshooting — copy the error, don't guess

| Error | Fix |
|---|---|
| `kaggle: command not found` | `pip install kaggle`, then re-open your terminal |
| "No Kaggle credentials found" | `kaggle.json` isn't at the right path — recheck Step 1.3 |
| "Can't see the dataset as ready" | Team lead hasn't shared the dataset with your username yet (Step 2) |
| Anything else | Send the team lead the exact command you ran + the exact error text. Don't retry blind — a wasted 7-hour GPU run is expensive on the free tier. |
