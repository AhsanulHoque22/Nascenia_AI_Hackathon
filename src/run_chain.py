"""
Orchestrator for the sharded full-dataset training chain.

Runs the whole pipeline unattended:

    for each shard:
        push training kernel  ->  poll until COMPLETE
        download adapter      ->  publish it as a Kaggle dataset
        rewire the kernel to resume from it, bump the shard index
    then:
        push the select+submit kernel  ->  poll  ->  download submission.csv

Design notes, all learned the hard way in this project:

  * IDEMPOTENT AND RESUMABLE. State lives in experiments/chain_state.json, so
    if this script dies (or the laptop sleeps) it picks up exactly where it
    left off instead of re-running an 7-hour shard.
  * NEVER AUTO-SUBMITS. It generates submission.csv and stops. Submitting
    consumes one of 5 irreversible Phase-1 slots, which is the user's call.
  * FAILS LOUDLY. A kernel ERROR halts the chain and prints the tail of the
    remote log, rather than pushing the next shard on top of a broken adapter.
  * Kaggle's dataset processing is asynchronous — a freshly-created version is
    not immediately mountable — so uploads are waited on via `datasets status`
    before the next kernel is pushed.

Usage:
    python src/run_chain.py                 # run/resume the chain
    python src/run_chain.py --status        # print state and exit
    python src/run_chain.py --max-shards 3  # stop early (quota control)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

# The script shells out to the `kaggle` CLI, which lives in the venv's bin
# dir. Put that dir on PATH explicitly, derived from the running interpreter,
# so this works whether launched as `python src/run_chain.py` with the venv
# activated, as `venv/bin/python src/run_chain.py` without activating, or
# from a shell (fish, etc.) that never sourced activate at all.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

USER = "ahsanulhoque48cu"
TRAIN_KERNEL = f"{USER}/nascenia-day11-train"
SUBMIT_KERNEL = f"{USER}/nascenia-day11-select-submit"
ADAPTER_DATASET = f"{USER}/nascenia-adapter"

TRAIN_DIR = "kaggle_kernels/day11_train"
SUBMIT_DIR = "kaggle_kernels/day11_select_submit"
ADAPTER_STAGE = "kaggle_adapter"          # local staging dir for the upload
STATE_PATH = "experiments/chain_state.json"

N_SHARDS = 5
POLL_S = 300                              # 5 min; shards run ~7h
DATASET_WAIT_S = 30


LOCK_PATH = "experiments/chain.lock"


def acquire_lock():
    """Exactly one orchestrator at a time.

    Two copies running (say one inside a Claude session and one in a terminal)
    would both push shard N+1 and race on the adapter dataset. A stale lock
    from a killed process is detected by checking whether that PID is alive,
    so a crash does not require manual cleanup.
    """
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)          # signal 0 = liveness probe only
        except (ValueError, ProcessLookupError, PermissionError):
            log("removing stale lock")
            os.remove(LOCK_PATH)
        else:
            raise SystemExit(
                f"another orchestrator is already running (pid {old_pid}).\n"
                f"Stop it first, or remove {LOCK_PATH} if you know it is dead."
            )
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.exists(LOCK_PATH) and os.remove(LOCK_PATH))


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd, check=True, capture=True):
    r = subprocess.run(cmd, shell=True, text=True,
                       capture_output=capture, timeout=3600)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{r.stdout}\n{r.stderr}")
    return (r.stdout or "") + (r.stderr or "")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"shard": 0, "phase": "push_train", "history": []}


def save_state(st):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)


def set_kernel_vars(shard, resume):
    """Rewrite the SESSION CONTROL block in the training kernel."""
    path = f"{TRAIN_DIR}/script.py"
    with open(path) as f:
        s = f.read()
    s = re.sub(r"^SHARD_INDEX = \d+", f"SHARD_INDEX = {shard}", s, count=1, flags=re.M)
    s = re.sub(r"^RESUME = (True|False)", f"RESUME = {resume}", s, count=1, flags=re.M)
    with open(path, "w") as f:
        f.write(s)
    # The adapter dataset must be mounted for shards 1+, and must NOT be
    # listed for shard 0 (it does not exist yet, and Kaggle rejects the push).
    meta_path = f"{TRAIN_DIR}/kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    sources = [f"{USER}/nascenia-processed-data"]
    if resume:
        sources.append(ADAPTER_DATASET)
    meta["dataset_sources"] = sources
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"kernel set to SHARD_INDEX={shard} RESUME={resume} sources={sources}")


def kernel_status(ref):
    out = run(f"kaggle kernels status {ref}", check=False)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1) if m else out.strip()[:120]


def wait_for_kernel(ref, label):
    log(f"waiting on {label} ({ref})")
    while True:
        st = kernel_status(ref)
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            log(f"{label} -> {st}")
            return st
        time.sleep(POLL_S)


def dump_kernel_log(ref, out_dir):
    """Print the tail of a failed kernel's remote log so the failure is
    diagnosable without a manual round-trip to the website."""
    try:
        run(f"kaggle kernels output {ref} -p {out_dir}", check=False)
        logs = [f for f in os.listdir(out_dir) if f.endswith(".log")]
        if not logs:
            return
        with open(os.path.join(out_dir, logs[0]), encoding="utf-8", errors="replace") as f:
            text = f.read()
        i = text.rfind("Traceback")
        print("---- remote log tail ----")
        print(text[i - 200:i + 2000] if i > 0 else text[-2000:])
        print("-------------------------")
    except Exception as e:
        log(f"could not fetch log: {e}")


def publish_adapter(local_adapter_dir):
    """Upload the adapter as a Kaggle dataset so the next shard can mount it."""
    if os.path.exists(ADAPTER_STAGE):
        shutil.rmtree(ADAPTER_STAGE)
    shutil.copytree(local_adapter_dir, ADAPTER_STAGE)
    meta = {"title": "nascenia-adapter", "id": ADAPTER_DATASET,
            "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}
    with open(f"{ADAPTER_STAGE}/dataset-metadata.json", "w") as f:
        json.dump(meta, f)

    # A dataset that does not exist yet returns "403 Client Error: Forbidden"
    # from `datasets status`, NOT a string containing "not found" — checking
    # for "not found" therefore misreads the common case (first-ever publish)
    # as "already exists" and calls `datasets version` on something that was
    # never created, which fails. Only treat it as existing when the status
    # call actually reports a real dataset state.
    status_out = run(f"kaggle datasets status {ADAPTER_DATASET}", check=False).lower()
    exists = any(s in status_out for s in ("ready", "processing", "error(", "queued"))
    if exists:
        log("publishing new adapter dataset version")
        run(f'kaggle datasets version -p {ADAPTER_STAGE} -m "shard chain" -d')
    else:
        log("creating adapter dataset")
        run(f"kaggle datasets create -p {ADAPTER_STAGE} -d")

    # Kaggle processes uploads asynchronously; mounting too early gives an
    # empty directory and the next shard would silently start from scratch.
    for _ in range(40):
        time.sleep(DATASET_WAIT_S)
        if "ready" in run(f"kaggle datasets status {ADAPTER_DATASET}", check=False).lower():
            log("adapter dataset ready")
            return
    raise RuntimeError("adapter dataset did not become ready")


def do_shard(st, shard):
    resume = shard > 0
    # Adopt a run that is already in flight instead of re-pushing it. Without
    # this, restarting the orchestrator would kill and restart a shard that is
    # already hours in — the single most expensive mistake this script could
    # make. A push is only issued when nothing is currently running.
    current = kernel_status(TRAIN_KERNEL)
    if current == "RUNNING" and st.get("pushed_shard") == shard:
        log(f"shard {shard} already RUNNING — adopting it, not re-pushing")
    elif current == "RUNNING" and st.get("pushed_shard") is None:
        log(f"a run is already in flight and state does not say which shard; "
            f"assuming it is shard {shard} and adopting it")
        st["pushed_shard"] = shard
        save_state(st)
    else:
        set_kernel_vars(shard, resume)
        log(f"pushing shard {shard}/{N_SHARDS - 1}")
        run(f"kaggle kernels push -p {TRAIN_DIR}")
        st["pushed_shard"] = shard
        save_state(st)
        time.sleep(20)

    status = wait_for_kernel(TRAIN_KERNEL, f"shard {shard}")
    out_dir = f"/tmp/chain_shard{shard}"
    shutil.rmtree(out_dir, ignore_errors=True)
    if status != "COMPLETE":
        dump_kernel_log(TRAIN_KERNEL, out_dir)
        raise RuntimeError(f"shard {shard} ended {status} — chain halted")

    log("downloading adapter")
    run(f"kaggle kernels output {TRAIN_KERNEL} -p {out_dir}")
    hits = [os.path.join(r, "adapter_config.json")
            for r, _, fs in os.walk(out_dir) if "adapter_config.json" in fs]
    finals = [h for h in hits if "adapter_final" in h] or sorted(hits)
    if not finals:
        raise RuntimeError(f"no adapter in {out_dir}: {os.listdir(out_dir)}")
    adapter_dir = os.path.dirname(finals[0])

    metrics = {}
    mp = [os.path.join(r, "metrics.json") for r, _, fs in os.walk(out_dir)
          if "metrics.json" in fs]
    if mp:
        with open(mp[0]) as f:
            metrics = json.load(f)
        log(f"shard {shard}: loss {metrics.get('train_loss')}, "
            f"{metrics.get('steps')} steps, {metrics.get('train_time_s', 0)/3600:.2f}h, "
            f"coverage ~{metrics.get('rows_covered_cumulative')}")

    publish_adapter(adapter_dir)
    st["history"].append({
        "shard": shard, "finished": datetime.now().isoformat(timespec="seconds"),
        "train_loss": metrics.get("train_loss"), "steps": metrics.get("steps"),
        "hours": round(metrics.get("train_time_s", 0) / 3600, 2),
    })


def do_submit(st):
    log("pushing select+submit kernel")
    run(f"kaggle kernels push -p {SUBMIT_DIR}")
    time.sleep(20)
    status = wait_for_kernel(SUBMIT_KERNEL, "select+submit")
    out_dir = "/tmp/chain_submit"
    shutil.rmtree(out_dir, ignore_errors=True)
    if status != "COMPLETE":
        dump_kernel_log(SUBMIT_KERNEL, out_dir)
        raise RuntimeError(f"select+submit ended {status}")
    run(f"kaggle kernels output {SUBMIT_KERNEL} -p {out_dir}")
    os.makedirs("submissions", exist_ok=True)
    for name in ("submission.csv", "checkpoint_selection.json"):
        src = os.path.join(out_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join("submissions", name))
            log(f"retrieved {name}")
    sel = os.path.join(out_dir, "checkpoint_selection.json")
    if os.path.exists(sel):
        with open(sel) as f:
            res = json.load(f)
        log("checkpoint ranking:")
        for r in res:
            log(f"  composite {r['composite']:.4f}  len_ratio {r['length_ratio']:.2f}  "
                f"{os.path.basename(r['checkpoint'])}")


def step_once(st, max_shards):
    """One non-blocking pass. Designed for cron (GitHub Actions): look at the
    world, take at most one action, exit. Never waits on a running kernel,
    because a scheduled job cannot sit for the ~7h a shard takes.

    Returns a short status string for the workflow log.
    """
    shard = st["shard"]
    if shard >= min(max_shards, N_SHARDS):
        if st.get("phase") == "done":
            return "chain already complete"
        do_submit(st)
        st["phase"] = "done"
        save_state(st)
        return "submission generated"

    status = kernel_status(TRAIN_KERNEL)

    # Nothing pushed for this shard yet -> push it.
    if st.get("pushed_shard") != shard:
        if status == "RUNNING":
            return f"kernel busy with earlier work; waiting before pushing shard {shard}"
        set_kernel_vars(shard, shard > 0)
        run(f"kaggle kernels push -p {TRAIN_DIR}")
        st["pushed_shard"] = shard
        save_state(st)
        return f"pushed shard {shard}"

    if status == "RUNNING":
        return f"shard {shard} still running"

    if status != "COMPLETE":
        out_dir = f"/tmp/chain_shard{shard}"
        dump_kernel_log(TRAIN_KERNEL, out_dir)
        raise RuntimeError(f"shard {shard} ended {status}")

    # Completed -> harvest the adapter, publish it, advance the counter. The
    # NEXT invocation pushes the next shard, keeping each run short.
    out_dir = f"/tmp/chain_shard{shard}"
    shutil.rmtree(out_dir, ignore_errors=True)
    run(f"kaggle kernels output {TRAIN_KERNEL} -p {out_dir}")
    hits = [os.path.join(r, "adapter_config.json")
            for r, _, fs in os.walk(out_dir) if "adapter_config.json" in fs]
    finals = [h for h in hits if "adapter_final" in h] or sorted(hits)
    if not finals:
        raise RuntimeError(f"shard {shard} COMPLETE but no adapter in {out_dir}")

    metrics = {}
    mp = [os.path.join(r, "metrics.json") for r, _, fs in os.walk(out_dir)
          if "metrics.json" in fs]
    if mp:
        with open(mp[0]) as f:
            metrics = json.load(f)

    publish_adapter(os.path.dirname(finals[0]))
    st["history"].append({
        "shard": shard, "finished": datetime.now().isoformat(timespec="seconds"),
        "train_loss": metrics.get("train_loss"), "steps": metrics.get("steps"),
        "hours": round(metrics.get("train_time_s", 0) / 3600, 2),
        "coverage": metrics.get("rows_covered_cumulative"),
    })
    st["shard"] = shard + 1
    st["pushed_shard"] = None
    save_state(st)
    return (f"shard {shard} COMPLETE (loss {metrics.get('train_loss')}, "
            f"{round(metrics.get('train_time_s', 0)/3600, 2)}h) -> advanced to {shard+1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="single non-blocking step, for cron/CI")
    ap.add_argument("--max-shards", type=int, default=N_SHARDS,
                    help="stop after this many shards (quota control)")
    ap.add_argument("--skip-submit", action="store_true")
    args = ap.parse_args()

    st = load_state()
    if args.status:
        print(json.dumps(st, indent=2))
        print(f"\ntrain kernel: {kernel_status(TRAIN_KERNEL)}")
        return

    if args.once:
        # No lock: CI runs are serialised by the workflow's concurrency group,
        # and a lock file would not survive between fresh containers anyway.
        try:
            log(step_once(st, args.max_shards))
        except Exception as e:
            st["phase"] = f"failed: {e}"
            save_state(st)
            log(f"CHAIN HALTED: {e}")
            sys.exit(1)
        return

    acquire_lock()
    log(f"chain starting at shard {st['shard']} (target {args.max_shards} shards)")
    try:
        while st["shard"] < min(args.max_shards, N_SHARDS):
            do_shard(st, st["shard"])
            st["shard"] += 1
            save_state(st)
            log(f"--- shard done; {st['shard']}/{args.max_shards} complete ---")

        if not args.skip_submit:
            do_submit(st)
            st["phase"] = "done"
            save_state(st)
    except Exception as e:
        st["phase"] = f"failed: {e}"
        save_state(st)
        log(f"CHAIN HALTED: {e}")
        sys.exit(1)

    log("chain complete")
    log("submissions/submission.csv is ready — NOT submitted. "
        "Review it, then submit manually (5 Phase-1 slots, irreversible).")


if __name__ == "__main__":
    main()
