#!/usr/bin/env python
"""Measure the preferred image direction of each T4/T5 subtype under this BoxEye setup.

Presents drifting square-wave gratings moving +x, -x, +y, -y (image coordinates: x right, y
down) through the configured BoxEye into frozen flyvis, records the mean rectified response of
each subtype, and writes each subtype's preferred unit vector (plus the grey baseline used by
LPTCPool) to ``data/flyvis/direction_map.json``.

Usage: python scripts/calibrate_t4t5_directions.py [--out PATH] [key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import load_config
from flybywire.vision.boxeye import BoxEye
from flybywire.vision.directions import default_direction_map_path, measure_directions
from flybywire.vision.flyvis_stepper import FlyvisStepper


def calibrate(cfg, device: str | None = None) -> dict:
    clocks = clocks_from_cfg(cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device=device)
    stepper = FlyvisStepper(
        model=cfg.vision.flyvis_model,
        device=device,
        dt=clocks.dt_sensory,
        cell_types=tuple(cfg.vision.cell_types),
        steady_state_seconds=float(cfg.vision.steady_state_seconds),
    )
    dm = measure_directions(stepper, eye, tuple(cfg.vision.cell_types), frame_size=int(cfg.env.frame_size),
                            frame_hz=float(clocks.sensory_hz))
    dm["boxeye"] = {"extent": int(cfg.vision.boxeye.extent), "kernel_size": int(cfg.vision.boxeye.kernel_size)}
    return dm


def print_table(dm: dict) -> None:
    print(f"\n{'subtype':<8} {'dx':>7} {'dy':>7} {'deg':>8} {'DSI':>6}   R(+x)   R(-x)   R(+y)   R(-y)")
    for t, s in dm["subtypes"].items():
        r = s["responses"]
        print(f"{t:<8} {s['direction'][0]:>7.3f} {s['direction'][1]:>7.3f} {s['angle_deg']:>8.1f} "
              f"{s['dsi']:>6.3f}  {r['+x']:.4f}  {r['-x']:.4f}  {r['+y']:.4f}  {r['-y']:.4f}")
    print("(image coordinates: x right, y down; angle = atan2(dy, dx))")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: $FLYVIS_ROOT_DIR/direction_map.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(overrides=args.overrides)
    dm = calibrate(cfg, args.device)
    out = Path(args.out) if args.out else default_direction_map_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dm, indent=2))
    print_table(dm)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
