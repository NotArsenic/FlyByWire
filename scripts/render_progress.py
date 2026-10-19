#!/usr/bin/env python
"""Side-by-side GIF of checkpoints from successive training stages flying the same corridor.

Every checkpoint flies the same held-out seed with the action mean, in lockstep, and its chase
camera fills one tile. A drone whose episode ends keeps its last frame with the outcome written
on it, so the tiles show how far each stage gets. "untrained" as a path builds a network with
random weights from the config of the next checkpoint in the list.

Usage:
  python scripts/render_progress.py --out docs/media/progress.gif --seed 10003 [--seconds 12] \\
      [key=value ...] -- "untrained=untrained" "BC=runs/sweep_dart/real_anatomical_s5/model_best.pt" ...

key=value overrides set the corridor for every checkpoint (e.g. env.obstacle_spacing_m=2.0
env.n_obstacles=12 for the dense level).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch


def untrained_checkpoint(like: str, path: Path) -> str:
    from omegaconf import OmegaConf

    from flybywire.core.model import build_model

    cfg = OmegaConf.create(torch.load(like, map_location="cpu")["config"])
    model = build_model(cfg, verbose=False)
    torch.save({"model": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, path)
    return str(path)


def label_tile(img: np.ndarray, text: str, sub: str) -> np.ndarray:
    import cv2

    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0) if "crash" in sub else (40, 40, 40),
                1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    import imageio.v2 as imageio

    from flybywire.eval.live import Flight

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help=".gif (an .mp4 with the same name is written too)")
    ap.add_argument("--seed", type=int, default=10003)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--tile", type=int, nargs=2, default=(360, 270), help="width height of one tile")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--stages", nargs="+", required=True, help="label=checkpoint, in training order")
    args = ap.parse_args(argv)

    stages = [s.split("=", 1) for s in args.stages]
    tmp = Path(tempfile.mkdtemp())
    for i, (_, path) in enumerate(stages):
        if path == "untrained":
            like = next(p for _, p in stages[i + 1 :] if p != "untrained")
            stages[i][1] = untrained_checkpoint(like, tmp / "untrained.pt")

    overrides = [f"env.episode_seconds={args.seconds}", *args.overrides]
    flights = [Flight(path, args.seed, overrides=overrides) for _, path in stages]
    w, h = args.tile
    last = [None] * len(flights)
    frames = []
    while not all(f.done for f in flights):
        tiles = []
        for i, ((label, _), f) in enumerate(zip(stages, flights, strict=True)):
            if not f.done:
                f.step()
                outcome = f"t = {f.time:4.1f} s"
                if f.done:
                    outcome = (f"crashed at {f.time:.1f} s" if f.reason == "collision"
                               else f"made it: {f.time:.1f} s, {f.trail[-1][0] - f.trail[0][0]:.1f} m")
                last[i] = label_tile(f.chase_image(w, h), label, outcome)
            tiles.append(last[i])
        frames.append(np.concatenate(tiles, axis=1))
    for f in flights:
        f.close()
    frames += [frames[-1]] * 25  # hold the end for a second

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out.with_suffix(".mp4"), frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
    imageio.mimwrite(out, [f[::2, ::2] for f in frames[::2]], duration=80, loop=0)
    print(f"wrote {out} and {out.with_suffix('.mp4')}: {len(frames)} frames")
    for (label, _), f in zip(stages, flights, strict=True):
        print(f"  {label}: {f.reason} at {f.time:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
