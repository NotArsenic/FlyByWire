"""Controlled-comparison sweep: mask conditions x seeds as concurrent BC subprocesses.

Launches ``python -m flybywire.train.bc`` once per (condition, seed) from ``sweep.conditions`` and
``sweep.seeds``, at most ``sweep.max_parallel`` at a time, each writing to
``<runs_dir>/sweep/<condition>_s<seed>/`` with its stdout/stderr in ``train.log``. When all finish,
collects every ``metrics.json`` into ``<runs_dir>/sweep/summary.json``. The comparison runs in the
BC regime only (docs/DECISIONS.md D7).

CLI::

    python -m flybywire.train.sweep [--dry-run] [--updates N] [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def build_jobs(cfg, cache_dir: str, out_root: Path, extra: list[str], updates: int | None) -> list[dict]:
    jobs = []
    for cond in cfg.sweep.conditions:
        for seed in cfg.sweep.seeds:
            out = out_root / f"{cond}_s{seed}"
            cmd = [sys.executable, "-m", "flybywire.train.bc", "--cache", cache_dir, "--out", str(out)]
            if updates is not None:
                cmd += ["--updates", str(updates)]
            cmd += [f"connectome.mask_type={cond}", f"seed={seed}", *extra]
            jobs.append({"condition": str(cond), "seed": int(seed), "out": out, "cmd": cmd})
    return jobs


def run_jobs(jobs: list[dict], max_parallel: int, poll_s: float = 2.0) -> list[dict]:
    pending = list(jobs)
    running: list[tuple[dict, subprocess.Popen, object]] = []
    done: list[dict] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            job["out"].mkdir(parents=True, exist_ok=True)
            log = open(job["out"] / "train.log", "w")
            print(f"[sweep] start {job['condition']} seed {job['seed']}")
            running.append((job, subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT), log))
        still = []
        for job, proc, log in running:
            if proc.poll() is None:
                still.append((job, proc, log))
                continue
            log.close()
            job["returncode"] = proc.returncode
            print(f"[sweep] end   {job['condition']} seed {job['seed']} rc={proc.returncode}")
            done.append(job)
        running = still
        if running:
            time.sleep(poll_s)
    return done


def summarize(jobs: list[dict], out_root: Path) -> dict:
    rows = []
    for job in jobs:
        m = job["out"] / "metrics.json"
        rows.append({"condition": job["condition"], "seed": job["seed"], "returncode": job.get("returncode"),
                     **(json.loads(m.read_text()) if m.exists() else {})})
    summary = {"runs": rows}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    from flybywire.config import data_dir, load_config, runs_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None, help="default: <runs_dir>/sweep")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cache = args.cache or str(data_dir(cfg) / "cache")
    out_root = Path(args.out) if args.out else runs_dir(cfg) / "sweep"
    jobs = build_jobs(cfg, cache, out_root, list(args.overrides), args.updates)
    if args.dry_run:
        for j in jobs:
            print(" ".join(j["cmd"]))
        return 0
    out_root.mkdir(parents=True, exist_ok=True)
    done = run_jobs(jobs, int(cfg.sweep.max_parallel))
    summarize(done, out_root)
    return 0 if all(j.get("returncode") == 0 for j in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
