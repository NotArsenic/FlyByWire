"""Controlled-comparison sweep: mask conditions x heading decoders x seeds as concurrent BC subprocesses.

Launches ``python -m flybywire.train.bc`` once per (condition, decoder, seed) from
``sweep.conditions``, ``sweep.decoders`` and ``sweep.seeds``, at most ``sweep.max_parallel`` at a
time, each writing to ``<out>/<condition>_<decoder>_s<seed>/`` (default out: ``<runs_dir>/sweep``)
with its stdout/stderr in ``train.log``. With ``--resume`` a run whose directory already has a
``metrics.json`` is not trained again. When all finish, collects every ``metrics.json`` into
``<out>/summary.json``; rows of runs this call doesn't cover (an earlier sweep into the same
directory) stay. The comparison runs in the BC regime only (docs/DECISIONS.md D7).

CLI::

    python -m flybywire.train.sweep [--dry-run] [--resume] [--updates N] [key=value ...]
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
        for decoder in cfg.sweep.decoders:
            for seed in cfg.sweep.seeds:
                out = out_root / f"{cond}_{decoder}_s{seed}"
                cmd = [sys.executable, "-m", "flybywire.train.bc", "--cache", cache_dir, "--out", str(out)]
                if updates is not None:
                    cmd += ["--updates", str(updates)]
                cmd += [f"connectome.mask_type={cond}", f"heads.decoder={decoder}", f"seed={seed}", *extra]
                jobs.append({"condition": str(cond), "decoder": str(decoder), "seed": int(seed), "out": out,
                             "cmd": cmd})
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
            print(f"[sweep] start {job['condition']} {job['decoder']} seed {job['seed']}")
            running.append((job, subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT), log))
        still = []
        for job, proc, log in running:
            if proc.poll() is None:
                still.append((job, proc, log))
                continue
            log.close()
            job["returncode"] = proc.returncode
            print(f"[sweep] end   {job['condition']} {job['decoder']} seed {job['seed']} rc={proc.returncode}")
            done.append(job)
        running = still
        if running:
            time.sleep(poll_s)
    return done


def summarize(jobs: list[dict], out_root: Path) -> dict:
    rows = []
    for job in jobs:
        m = job["out"] / "metrics.json"
        rows.append({"condition": job["condition"], "decoder": job["decoder"], "seed": job["seed"],
                     "returncode": job.get("returncode"), **(json.loads(m.read_text()) if m.exists() else {})})
    path = out_root / "summary.json"
    if path.exists():
        new = {(r["condition"], r["decoder"], r["seed"]) for r in rows}
        kept = [r for r in json.loads(path.read_text())["runs"] if (r["condition"], r["decoder"], r["seed"]) not in new]
        rows = kept + rows
    summary = {"runs": rows}
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    from flybywire.config import data_dir, load_config, runs_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None, help="default: <runs_dir>/sweep")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip runs that already have a metrics.json")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cache = args.cache or str(data_dir(cfg) / "cache")
    out_root = Path(args.out) if args.out else runs_dir(cfg) / "sweep"
    train_overrides = [o for o in args.overrides if not o.startswith("sweep.")]
    jobs = build_jobs(cfg, cache, out_root, train_overrides, args.updates)
    if args.dry_run:
        for j in jobs:
            print(" ".join(j["cmd"]))
        return 0
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (args.resume and (j["out"] / "metrics.json").exists())]
    run_jobs(todo, int(cfg.sweep.max_parallel))
    summarize(jobs, out_root)
    return 0 if all(j.get("returncode", 0) == 0 for j in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
