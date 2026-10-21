#!/usr/bin/env python
"""Build a pause bank for a T4/T5 cache by rerunning frozen flyvis (see flybywire/data/pause_bank.py).

Usage:
  python scripts/build_pause_bank.py --cache data/cache_turns --out data/pause_bank_turns
                                     [--per-episode 4] [--hold 48 100] [--recovery 60] [--seed 0]

Each entry holds the frame at a random decision boundary still for a multiple of 4 steps in
--hold, then continues on the real frames for --recovery steps. The cost is about one cache build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import data_dir, load_config
from flybywire.data.cache import T4T5Cache
from flybywire.data.pause_bank import PauseBank, build_bank
from flybywire.vision.flyvis_stepper import FlyvisStepper


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", default=None, help="default: <data_dir>/cache")
    ap.add_argument("--out", default=None, help="default: <data_dir>/pause_bank")
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument(
        "--hold",
        type=int,
        nargs=2,
        default=(48, 100),
        help="shortest and longest pause, core steps",
    )
    ap.add_argument(
        "--recovery",
        type=int,
        default=60,
        help="core steps on real frames after a pause",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="episodes per flyvis batch")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.overrides)
    clocks = clocks_from_cfg(cfg)
    base = data_dir(cfg)
    cache = T4T5Cache(args.cache or base / "cache")
    cache.check_compatible(cfg.vision.flyvis_model, clocks.dt_sensory)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stepper = FlyvisStepper(
        model=cfg.vision.flyvis_model,
        device=device,
        dt=clocks.dt_sensory,
        cell_types=tuple(cfg.vision.cell_types),
        steady_state_seconds=float(cfg.vision.steady_state_seconds),
    )
    extra = {
        "flyvis_model": cfg.vision.flyvis_model,
        "dt": clocks.dt_sensory,
        "stepper_mode": stepper.mode,
        "cache_config_hash": cache.manifest.get("config_hash"),
    }
    out = Path(args.out or base / "pause_bank")
    print(
        f"building pause bank for {len(cache)} episodes of {cache.dir} -> {out} (device {device})"
    )
    build_bank(
        cache,
        stepper,
        out,
        clocks,
        args.per_episode,
        tuple(args.hold),
        args.recovery,
        args.seed,
        args.batch,
        manifest_extra=extra,
    )
    bank = PauseBank(out)
    rel = bank.manifest["recovery_vs_cache_rel"]
    print(
        f"done: {len(bank)} entries, {bank.data.shape[0]} rows, {bank.data.nbytes / 1e9:.2f} GB, "
        f"{bank.manifest['build_seconds']} s"
    )
    print(
        f"rerun vs cache at the branch points: max abs {bank.manifest['rerun_vs_cache_max_abs']:.2e}"
    )
    print(
        "recovery vs cache, mean abs / cache sd after "
        + ", ".join(
            f"{n} steps {rel[n - 1]:.4f}"
            for n in (1, 10, 20, 30, 40, 50, 60)
            if n <= len(rel)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
