"""
Merge-chain orchestrator — the automated half of the multi-account plan.

What it does, once teammates' adapters are shared back:
    1. wait for your own chain to finish (shard 1) and each teammate's
       adapter dataset to be shared + ready
    2. download all of them
    3. merge them (src/merge_adapters.py — weight-averaged LoRA soup)
    4. publish the merge as a Kaggle dataset
    5. push the scoring kernel (kaggle_kernels/day11_select_submit),
       now pointed at ALL candidates: your own adapter, every teammate's,
       AND the merge — it ranks all of them by the real composite metric
       on held-out val and generates the test submission with the winner
    6. retrieve submission.csv

What it explicitly does NOT need: teammates' Kaggle credentials. Dataset
sharing (Settings -> Sharing -> Add collaborator) grants your account
download access to what they publish; nothing here ever touches their
tokens. It also never submits to the competition — that stays a decision
you make by hand, same as run_chain.py.

CONFIGURE TEAMMATES BELOW before running. Dataset names must match
TEAMMATE_SETUP.md's naming convention exactly (nascenia-shard-N-adapter) or
this can't find them.

Usage:
    python src/merge_chain.py --once     # single non-blocking step, for cron
    python src/merge_chain.py            # blocking, polls until submission.csv exists
    python src/merge_chain.py --status
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# The bin dir with `kaggle` in it may not be on PATH depending on how this
# is launched (see run_chain.py — same fix, same reason).
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

# ============================ TEAMMATES ============================
# shard_index -> Kaggle username. Must match what each teammate set in
# their own kernel-metadata.json and dataset-metadata.json.
TEAMMATES = {
    2: "REPLACE_ME_teammate_a_username",
    3: "REPLACE_ME_teammate_b_username",
    4: "REPLACE_ME_teammate_c_username",
}
# =====================================================================

USER = "ahsanulhoque48cu"
TRAIN_KERNEL = f"{USER}/nascenia-day11-train"
SUBMIT_KERNEL = f"{USER}/nascenia-day11-select-submit"
SUBMIT_DIR = "kaggle_kernels/day11_select_submit"
MERGED_DATASET = f"{USER}/nascenia-merged-adapter"
MERGE_STAGE = "kaggle_merged_adapter"
STATE_PATH = "experiments/merge_chain_state.json"
POLL_S = 300


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=3600)
    if check and r.returncode != 0:
        detail = ((r.stdout or "") + (r.stderr or ""))[-1500:]
        raise RuntimeError(f"command failed: {cmd}\n{detail}")
    return (r.stdout or "") + (r.stderr or "")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"phase": "waiting", "merged_published": False}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)


def dataset_ready(ref):
    out = run(f"kaggle datasets status {ref}", check=False).lower()
    return "ready" in out


OWN_CHAIN_STATE = "experiments/chain_state.json"
OWN_TARGET_SHARDS = 2  # must match train-chain.yml's max_shards


def own_chain_done():
    """True once run_chain.py has fully harvested through OWN_TARGET_SHARDS.

    Deliberately NOT "is the latest kernel status COMPLETE" — that's a race:
    shard 0 finishing leaves the kernel COMPLETE for the whole window before
    shard 1 gets pushed (train-chain.yml and merge-chain.yml poll on
    independent schedules), and reading only kernel status there would merge
    shard 0 alone, silently dropping 16,355 rows nobody would notice missing.
    chain_state.json's `shard` counter only advances after a harvest, and
    `pushed_shard` is None between "harvested" and "next shard pushed" — both
    together are the actual authoritative signal.
    """
    if not os.path.exists(OWN_CHAIN_STATE):
        return False
    with open(OWN_CHAIN_STATE) as f:
        cs = json.load(f)
    return cs.get("shard", 0) >= OWN_TARGET_SHARDS and cs.get("pushed_shard") is None


def teammates_ready():
    """{shard: (username, dataset_ref, ready_bool)} for every configured teammate."""
    result = {}
    for shard, username in TEAMMATES.items():
        ref = f"{username}/nascenia-shard-{shard}-adapter"
        result[shard] = (username, ref, dataset_ready(ref))
    return result


def download_dataset(ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    run(f"kaggle datasets download {ref} -p {out_dir} --unzip")
    return out_dir


def merge_and_publish():
    log("all adapters ready — downloading")
    own_dir = "/tmp/merge_chain/own"
    shutil.rmtree(own_dir, ignore_errors=True)
    run(f"kaggle kernels output {TRAIN_KERNEL} -p {own_dir}")
    own_hits = [os.path.dirname(p) for p in
                glob.glob(f"{own_dir}/**/adapter_config.json", recursive=True)
                if "adapter_final" in p]
    if not own_hits:
        raise RuntimeError(f"own chain COMPLETE but no adapter_final found in {own_dir}")
    adapter_dirs = [own_hits[0]]

    for shard, (username, ref, _) in sorted(TEAMMATES.items()):
        d = download_dataset(ref, f"/tmp/merge_chain/shard{shard}")
        if not os.path.exists(os.path.join(d, "adapter_config.json")):
            raise RuntimeError(f"{ref} downloaded but has no adapter_config.json at top level: "
                               f"{os.listdir(d)}")
        adapter_dirs.append(d)

    log(f"merging {len(adapter_dirs)} adapters: {adapter_dirs}")
    merged_dir = "/tmp/merge_chain/merged"
    shutil.rmtree(merged_dir, ignore_errors=True)
    sys.path.insert(0, "src")
    from merge_adapters import merge
    merge(adapter_dirs, merged_dir)

    log("publishing merged adapter as a Kaggle dataset")
    if os.path.exists(MERGE_STAGE):
        shutil.rmtree(MERGE_STAGE)
    shutil.copytree(merged_dir, MERGE_STAGE)
    with open(f"{MERGE_STAGE}/dataset-metadata.json", "w") as f:
        json.dump({"title": "nascenia-merged-adapter", "id": MERGED_DATASET,
                   "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, f)

    exists = "ready" in run(f"kaggle datasets status {MERGED_DATASET}", check=False).lower()
    if exists:
        run(f'kaggle datasets version -p {MERGE_STAGE} -m "merge update" -d -q')
    else:
        run(f"kaggle datasets create -p {MERGE_STAGE} -q")

    for _ in range(40):
        time.sleep(30)
        if dataset_ready(MERGED_DATASET):
            log("merged adapter dataset ready")
            return
    raise RuntimeError("merged adapter dataset did not become ready")


def push_scoring_kernel():
    """Point the existing scoring kernel at every candidate: your own chain
    (via kernel_sources, automatic), every teammate's adapter, and the merge
    (via dataset_sources). It already globs for adapter_config.json anywhere
    under /kaggle/input, ranks all of it by composite score on held-out val,
    and generates the submission with the winner — no kernel code changes
    needed, only which datasets are mounted."""
    meta_path = f"{SUBMIT_DIR}/kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    sources = [f"{USER}/nascenia-processed-data", MERGED_DATASET]
    sources += [ref for _, ref, _ in teammates_ready().values()]
    meta["dataset_sources"] = sources
    meta["kernel_sources"] = [TRAIN_KERNEL]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"scoring kernel will see {len(sources)} dataset sources: {sources}")
    run(f"kaggle kernels push -p {SUBMIT_DIR}")


def step_once(st):
    if st["phase"] == "waiting":
        if not own_chain_done():
            return "own chain not finished yet"
        tm = teammates_ready()
        not_ready = [f"shard{s}({u})" for s, (u, _, ok) in tm.items() if not ok]
        if not_ready:
            return f"waiting on teammates: {', '.join(not_ready)}"
        merge_and_publish()
        st["phase"] = "merged"
        save_state(st)
        return "merged and published — next step pushes the scoring kernel"

    if st["phase"] == "merged":
        push_scoring_kernel()
        st["phase"] = "scoring"
        save_state(st)
        return "pushed scoring kernel"

    if st["phase"] == "scoring":
        out = run(f"kaggle kernels status {SUBMIT_KERNEL}", check=False)
        if "RUNNING" in out:
            return "scoring kernel still running"
        if "COMPLETE" not in out:
            raise RuntimeError(f"scoring kernel ended unexpectedly: {out}")
        out_dir = "/tmp/merge_chain/submit_out"
        shutil.rmtree(out_dir, ignore_errors=True)
        run(f"kaggle kernels output {SUBMIT_KERNEL} -p {out_dir}")
        os.makedirs("submissions", exist_ok=True)
        for name in ("submission.csv", "checkpoint_selection.json"):
            src = os.path.join(out_dir, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join("submissions", name))
        st["phase"] = "done"
        save_state(st)
        return ("DONE — submissions/submission.csv ready. NOT submitted; "
                "that decision is yours.")

    return "already done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if "REPLACE_ME" in str(TEAMMATES.values()):
        raise SystemExit(
            "TEAMMATES dict at the top of this file still has placeholder "
            "usernames — fill in the real Kaggle usernames before running.")

    st = load_state()
    if args.status:
        print(json.dumps(st, indent=2))
        return

    if args.once:
        try:
            log(step_once(st))
        except Exception as e:
            st["phase_error"] = str(e)
            save_state(st)
            log(f"HALTED: {e}")
            sys.exit(1)
        return

    while st["phase"] != "done":
        log(step_once(st))
        if st["phase"] != "done":
            time.sleep(POLL_S)
    log("chain complete")


if __name__ == "__main__":
    main()
