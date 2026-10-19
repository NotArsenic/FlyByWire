#!/usr/bin/env python
"""Closed-loop success of checkpoints, and of the scripted expert, at each corridor level of the
dense stage (configs/default.yaml curriculum; "default" is the corridor everything else trained on).

Each level flies the held-out seeds 10000.. with the action mean. A level's overrides are applied to
the checkpoint's own config, so the network is the one that was trained and only the corridor (and,
for the expert, its settings) changes.

Usage:
  python scripts/eval_curriculum.py --ckpts "BC=runs/sweep_dart/real_anatomical_s5/model_best.pt" ...
                                    [--levels default spaced dense slalom] [--n 200] [--expert]
                                    [--out runs/curriculum/eval.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flybywire.config import REPO_ROOT, load_config
from flybywire.eval.closed_loop import ExpertDriver, PolicyDriver, fly, load_checkpoint
from flybywire.eval.metrics import wilson_interval


def level_cfg(cfg, level: str):
    if level == "default":
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(load_config().curriculum[level])))


def score(episodes) -> dict:
    ok = sum(e.success for e in episodes)
    return {"success": ok / len(episodes), "wilson95": list(wilson_interval(ok, len(episodes))),
            "x_progress_mean_m": float(np.mean([e.x_progress for e in episodes]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="*", default=[], help="label=checkpoint")
    ap.add_argument("--levels", nargs="+", default=["default", "spaced", "dense", "slalom"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--expert", action="store_true", help="also fly the scripted expert")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/curriculum/eval.json"))
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seed, args.seed + args.n))
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}
    if args.expert:
        base = load_checkpoint(args.ckpts[0].split("=", 1)[1])[1] if args.ckpts else load_config()
        report["expert"] = {}
        for level in args.levels:
            cfg = level_cfg(base, level)
            report["expert"][level] = score(fly(cfg, seeds, ExpertDriver(cfg, args.envs), n_envs=args.envs))
            print(f"expert {level}: {report['expert'][level]['success']:.3f}", flush=True)
    for item in args.ckpts:
        label, path = item.split("=", 1)
        model, cfg = load_checkpoint(path, device)
        report[label] = {"ckpt": path}
        for level in args.levels:
            c = level_cfg(cfg, level)
            report[label][level] = score(fly(c, seeds, PolicyDriver(model, c, args.envs), n_envs=args.envs))
            print(f"{label} {level}: {report[label][level]['success']:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
