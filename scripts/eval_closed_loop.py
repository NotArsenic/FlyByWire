#!/usr/bin/env python
"""Fly the BC policy in the corridor and compare with the scripted expert on the same seeds.

The env and expert are built from the config stored in the checkpoint, not configs/default.yaml,
so later config changes do not alter what an old checkpoint is tested on. Success means no
collision and no wall contact over the whole episode. Seeds default to 10000.., clear of the
expert seeds 0-299 that the policy trained on.

--record DIR also saves the policy's episodes as Trajectory files (hexals the policy saw, scripted
expert labels for the states it visited) for DAgger. See flybywire/eval/closed_loop.py for how the
labels are made.

Usage:
  python scripts/eval_closed_loop.py [--ckpt runs/bc/real_s0/model.pt] [--out DIR] [--n 50] [--seed 10000]
                                     [--n-envs 12] [--record DIR] [--skip-expert]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import REPO_ROOT
from flybywire.eval.closed_loop import (
    ExpertDriver,
    PolicyDriver,
    episode_rows,
    fly,
    load_checkpoint,
    summarize,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "runs" / "bc" / "real_s0" / "model.pt"))
    ap.add_argument("--out", default=None, help="directory for closed_loop.json (default: the checkpoint's)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10000, help="first episode seed")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--record", default=None, help="save policy episodes here in the Trajectory format")
    ap.add_argument("--label-wander-deg", type=float, default=0.0, help="wander of the labelling expert")
    ap.add_argument("--skip-expert", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    out_dir = Path(args.out) if args.out else ckpt.parent
    model, cfg = load_checkpoint(ckpt, device)
    clocks = clocks_from_cfg(cfg)
    seeds = list(range(args.seed, args.seed + args.n))
    print(f"{args.n} episodes (seeds {seeds[0]}..{seeds[-1]}), {args.n_envs} envs, checkpoint {ckpt}")

    t0 = time.time()
    driver = PolicyDriver(model, cfg, args.n_envs, label_wander_deg=args.label_wander_deg)
    meta = {"source": "policy", "checkpoint": str(ckpt), "label_wander_deg": args.label_wander_deg}
    policy = fly(cfg, seeds, driver, args.n_envs, record_dir=args.record, meta=meta, progress=True)
    t_policy = time.time() - t0
    report = {
        "checkpoint": str(ckpt),
        "seeds": [seeds[0], seeds[-1]],
        "n_envs": args.n_envs,
        "episode_seconds": float(cfg.env.episode_seconds),
        "policy": summarize(policy, clocks),
        "policy_episodes": episode_rows(policy),
        "seconds": {"policy": round(t_policy, 1)},
    }
    if args.record:
        report["recorded_to"] = str(args.record)
        report["label_wander_deg"] = args.label_wander_deg

    if not args.skip_expert:
        t0 = time.time()
        expert = fly(cfg, seeds, ExpertDriver(cfg, args.n_envs), args.n_envs, progress=True)
        report["expert"] = summarize(expert, clocks)
        report["expert_episodes"] = episode_rows(expert)
        report["seconds"]["expert"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "closed_loop.json").write_text(json.dumps(report, indent=2))
    for name in ("policy", "expert"):
        if name in report:
            s = report[name]
            print(f"{name:7s} success {s['success_rate']:.2f}  collisions {s['collisions']}  "
                  f"decisions {s['decisions_mean']:.1f}  x_progress {s['x_progress_mean_m']:.2f} m"
                  + (f"  heading err {s['heading_err_deg']['mean']:.1f} deg" if "heading_err_deg" in s else ""))
    print(f"wrote {out_dir / 'closed_loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
