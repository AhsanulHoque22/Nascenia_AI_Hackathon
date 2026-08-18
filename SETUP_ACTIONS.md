# Hands-off training chain via GitHub Actions

Runs the shard chain on GitHub's servers, so it advances whether or not your
laptop is on. Setup is ~5 minutes.

## What it does

Hourly, a short job checks the Kaggle kernel and takes at most one action:

| kernel state | action |
|---|---|
| shard still running | nothing, exit |
| shard COMPLETE | download adapter, publish it as a Kaggle dataset, advance the counter |
| next shard not yet pushed | push it |
| all shards done | run select+submit, upload `submission.csv` as an artifact |
| shard ERROR | dump the remote log, fail the run loudly, halt the chain |

It never submits to the competition. That stays a human decision.

## 1. Get your Kaggle token

```fish
cat ~/.kaggle/access_token
```

Copy the whole string (starts with `KGAT_`).

## 2. Create the repo and push

The repo currently has no remote. Create a **private** repo on GitHub, then:

```fish
git remote add origin git@github.com:<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

Nothing sensitive goes up — `.gitignore` already excludes `data/`,
`experiments/*/`, `submissions/*.csv`, and the Kaggle upload mirrors. Only
code, configs, and notes are committed.

## 3. Add the secret

Repo → **Settings → Secrets and variables → Actions → New repository secret**

- Name: `KAGGLE_API_TOKEN`
- Value: the `KGAT_...` string from step 1

## 4. Turn it on

Repo → **Actions** tab → enable workflows if prompted → select
**training chain** → **Run workflow** to fire the first one immediately
(otherwise it waits for the next hour boundary).

## Watching it

- **Actions** tab shows each run; the summary line says what it did.
- `experiments/chain_state.json` is committed back after every advance, so
  the commit history *is* the progress log.
- The finished `submission.csv` appears as a downloadable **artifact** on the
  final run.

## Notes and limits

**Actions minutes.** Private repos get 2,000 free minutes/month. Hourly runs
at ~1.5 min each is ~1,100/month, comfortably inside. Do **not** drop the
cron to every 15 minutes on a private repo — that would exceed the free tier.
A public repo has unlimited Actions minutes, but publishes your approach
during the competition.

**Hourly granularity** means up to ~1h idle after a shard finishes. Across
four shards that is a few hours of slack — much better than an overnight gap,
and not worth burning minutes to shave.

**Kaggle quota still binds.** ~30 GPU-h/week against ~7h per shard means
about 4 shards (~65k rows, 80% coverage). The workflow cannot conjure quota;
use `max_shards` on manual dispatch to stop earlier.

**Do not run the local orchestrator at the same time.** Both would push the
next shard. Pick one: Actions, or `venv/bin/python src/run_chain.py` locally.
