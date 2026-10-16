#!/usr/bin/env python
"""Build the T4/T5 cache from stored trajectories (hexals) by running frozen flyvis.

Usage:
  python scripts/build_cache.py [--traj-dir DIR] [--out DIR] [--batch 16] [key=value ...]

Defaults: --traj-dir <data_dir>/expert, --out <data_dir>/cache. See flybywire/data/cache.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import config_hash, data_dir, load_config
from flybywire.data.cache import T4T5Cache, build_cache
from flybywire.data.trajectories import list_trajectories
from flybywire.vision.flyvis_stepper import FlyvisStepper

VISION_KEYS = ["clocks.sensory_hz", "vision.flyvis_model", "vision.boxeye", "vision.cell_types",
               "vision.steady_state_seconds", "env.frame_size", "env.camera_fov_deg"]


def run(cfg, traj_dir: Path, out_dir: Path, batch: int, device: str | None = None) -> Path:
    paths = list_trajectories(traj_dir)
    if not paths:
        raise FileNotFoundError(f"no traj_*.npz in {traj_dir}")
    clocks = clocks_from_cfg(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        "cell_types": list(cfg.vision.cell_types),
        "steady_state_seconds": float(cfg.vision.steady_state_seconds),
        "stepper_mode": stepper.mode,
        "config_hash": config_hash(cfg, VISION_KEYS),
    }
    print(f"building cache from {len(paths)} trajectories in {traj_dir} -> {out_dir} (device {device})")
    build_cache(paths, out_dir, stepper, manifest_extra=extra, batch_size=batch)
    cache = T4T5Cache(out_dir)
    print(f"done: {len(cache)} episodes, total_T={cache.data.shape[0]}, shape={cache.data.shape}, "
          f"{cache.data.nbytes / 1e9:.2f} GB")
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=16, help="trajectories per flyvis batch (10-20)")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(overrides=args.overrides)
    base = Path(data_dir(cfg))
    run(cfg, Path(args.traj_dir or base / "expert"), Path(args.out or base / "cache"), args.batch, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
