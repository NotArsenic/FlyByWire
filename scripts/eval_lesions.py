#!/usr/bin/env python
"""Closed-loop flights with a cell type silenced for the whole episode, and what the EPG bump does.

For each lesion (none, EPG, PEN, D7 by default) the checkpoint flies the same held-out seeds
(10000.. as in failure_breakdown.py). Per lesion: success with its Wilson interval, decoded heading
error, and the EPG ring on the anatomical layout after a 50-step washout:

  rate        mean EPG rate
  size        length of the ring's population vector (how big the bump is)
  share       first-harmonic share of the ring profile (Gate 3's bump share; low = smeared)
  alignment   how well the bump angle follows the true heading up to a fixed offset and sign
  gain        slope of the bump's turn against the drone's turn over 0.5 s windows in which the
              drone turned more than 20 deg, sign from the intact bump: ~1 the bump follows,
              ~0 it stays put while the drone turns (frozen)

Usage:
  python scripts/eval_lesions.py [--ckpt runs/sweep_dart/real_anatomical_s5/model_best.pt]
                                 [--lesions none EPG PEN D7] [--n 200] [--seed 10000] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import REPO_ROOT
from flybywire.eval.closed_loop import PolicyDriver, fly, load_checkpoint, summarize
from flybywire.eval.metrics import wilson_interval
from flybywire.eval.ring import bump, model_layout, ring_activity, ring_tracking

WASHOUT = 50
WINDOW = 50
MIN_TURN = np.radians(20)


class RingDriver(PolicyDriver):
    """PolicyDriver that also keeps the EPG rates of every step, per episode seed."""

    def __init__(self, model, cfg, n_envs: int):
        super().__init__(model, cfg, n_envs)
        self.epg = torch.as_tensor(
            model.groups[model.group_roles["heading"]], device=self.device
        )
        self.seeds = [None] * n_envs
        self.buf = [[] for _ in range(n_envs)]
        self.rates = {}

    def start(self, j, seed, info):
        super().start(j, seed, info)
        self.seeds[j], self.buf[j] = seed, []

    def stop(self, j):
        super().stop(j)
        self.rates[self.seeds[j]] = np.concatenate(self.buf[j])

    @torch.no_grad()
    def act(self, frames, infos):
        n, w = frames.shape[:2]
        x = torch.from_numpy(frames).to(self.device)
        hexals = (
            self.eye(x.reshape(n * w, *x.shape[2:])).half().float().reshape(n, w, -1)
        )
        heading, epg = [], []
        for t in range(w):
            self.v, r = self.model.core_step(self.v, self.stepper.step(hexals[:, t]))
            heading.append(self.model.heading(r))
            epg.append(r[:, self.epg])
        epg = torch.stack(epg, dim=1).cpu().numpy()
        for j, e in enumerate(self.experts):
            if e is not None:
                self.buf[j].append(epg[j])
        label = np.stack(
            [
                e.act(i) if e is not None else np.zeros(4, np.float32)
                for e, i in zip(self.experts, infos, strict=True)
            ]
        )
        return {
            "action": self.model.actor(r).cpu().numpy(),
            "label": label,
            "heading": torch.stack(heading, dim=1).cpu().numpy(),
        }


def turn_gain(bumps, headings, sign: int) -> float:
    db, dh = [], []
    for b, h in zip(bumps, headings, strict=True):
        a = np.unwrap(b["angle"][WASHOUT:])
        y = np.unwrap(h[WASHOUT:])
        d_y = y[WINDOW:] - y[:-WINDOW]
        keep = np.abs(d_y) > MIN_TURN
        db.append((a[WINDOW:] - a[:-WINDOW])[keep])
        dh.append(d_y[keep])
    db, dh = np.concatenate(db), np.concatenate(dh)
    return float(sign * (db @ dh) / (dh @ dh)) if len(dh) else float("nan")


def ring_summary(
    rates: list[np.ndarray], headings: list[np.ndarray], layout, sign: int | None
) -> dict:
    bumps = [bump(ring_activity(r, layout)) for r in rates]
    track = ring_tracking(bumps, headings, washout=WASHOUT)
    sign = track["sign"] if sign is None else sign
    return {
        "rate": float(np.mean([r[WASHOUT:].mean() for r in rates])),
        "size": float(np.mean([b["size"][WASHOUT:].mean() for b in bumps])),
        "share": track["fraction_mean"],
        "alignment": track["alignment"],
        "alignment_null": track["alignment_null"],
        "gain": turn_gain(bumps, headings, sign),
        "sign": sign,
        "offset_deg": track["offset_deg"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "runs/sweep_dart/real_anatomical_s5/model_best.pt"),
    )
    ap.add_argument("--lesions", nargs="+", default=["none", "EPG", "PEN", "D7"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--out", default=None, help="default: runs/lesions/<run name>")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.ckpt, device)
    clocks = clocks_from_cfg(cfg)
    layout = model_layout(cfg, model, "anatomical")
    out = Path(args.out or REPO_ROOT / "runs/lesions" / Path(args.ckpt).parent.name)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.n))

    report, sign = {
        "ckpt": args.ckpt,
        "seeds": [seeds[0], seeds[-1]],
        "layout": layout.label,
    }, None
    for name in args.lesions:
        model.restore()
        if name != "none":
            model.lesion([name])
        driver = RingDriver(model, cfg, args.envs)
        episodes = fly(cfg, seeds, driver, n_envs=args.envs, progress=True)
        s = summarize(episodes, clocks)
        n_ok = sum(e.success for e in episodes)
        rates = [driver.rates[e.seed] for e in episodes]
        headings = [e.state["rel_yaw"] for e in episodes]
        ring = ring_summary(rates, headings, layout, sign)
        if name == "none":
            sign = ring["sign"]
        report[name] = {
            "success": n_ok / len(episodes),
            "wilson95": list(wilson_interval(n_ok, len(episodes))),
            "terminations": s["terminations"],
            "x_progress_mean_m": s["x_progress_mean_m"],
            "heading_err_deg": s["heading_err_deg"],
            "ring": ring,
        }
        print(
            f"{name:5s} success {report[name]['success']:.3f} heading err {s['heading_err_deg']['mean']:.1f} deg "
            + " ".join(f"{k} {v:.3f}" for k, v in ring.items() if isinstance(v, float))
        )
    model.restore()
    (out / "lesions.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'lesions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
