#!/usr/bin/env python
"""Heading persistence when the input stops carrying motion (after Seelig and Jayaraman 2015).

For each cached test episode and each t0, the checkpoint runs on the real T4/T5 up to t0. Then the
input is replaced for --dark-seconds by the flyvis grey steady-state T4/T5 ("no motion") and the
core keeps stepping. The LPTC channels under that input are reported next to their values at
the start of the real episodes (hovering over a static scene) and over the real frames used.
Measured against step 0 (the last real step): the drift of the decoded heading and of the ring
bump angle, and the bump size on the decoder and on the ring (retention). Two references: the same
branch with the recurrent weights zeroed (a network that only decays), and the network started
from rest on the same input (what a bump looks like when no heading was ever there).

The network never saw blank input in training. The input is a static grey scene rather than
darkness, so this tests the response to a lack of motion, not to lights-off.

Drift is only informative away from heading 0, where a collapsed network decodes the bias
heading, so besides all samples there is a subset that is nearly still (mean |yaw rate| over the
last 0.5 s below --still-rate) and more than 45 degrees off axis.

Usage:
  python scripts/eval_darkness.py [--ckpt runs/bc/turns_s0/model_best.pt] [--cache data/cache_turns_test]
                                  [--t0 1.5 1.75 ... 8] [--dark-seconds 2] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import REPO_ROOT
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.darkness import (
    dark_branch,
    dark_series,
    dark_stats,
    grey_t4t5,
    snapshots,
)
from flybywire.eval.ring import model_layout
from flybywire.train.bc import BCData


def plot(path: Path, taus, stats: dict, layout_label: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"recurrent": "#4c78a8", "no_recurrence": "#e45756", "rest": "#888888"}
    names = {
        "recurrent": "trained network",
        "no_recurrence": "recurrent weights zeroed",
        "rest": "from rest",
    }
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    sub = stats["still_off_axis"]
    panels = (
        (
            ax[0, 0],
            "heading_drift_deg",
            "median |drift of decoded heading| (deg)",
            ("recurrent", "no_recurrence"),
        ),
        (
            ax[0, 1],
            "ring_drift_deg",
            "median |drift of ring bump angle| (deg)",
            ("recurrent", "no_recurrence"),
        ),
    )
    for a, key, ylabel, conds in panels:
        for c in conds:
            a.plot(taus, sub[c][f"{key}_median"], color=colors[c], label=names[c])
            a.fill_between(
                taus,
                sub[c][f"{key}_q25"],
                sub[c][f"{key}_q75"],
                color=colors[c],
                alpha=0.15,
                lw=0,
            )
        a.set_ylabel(ylabel)
    for c in ("recurrent", "no_recurrence", "rest"):
        ax[1, 0].plot(
            taus, sub[c]["ring_retention"], color=colors[c], label=names[c] + ", ring"
        )
        ax[1, 0].plot(
            taus,
            sub[c]["decoder_retention"],
            color=colors[c],
            ls="--",
            label=names[c] + ", decoder",
        )
        ax[1, 1].plot(taus, sub[c]["ring_fraction"], color=colors[c], label=names[c])
    ax[1, 0].set_ylabel("bump size / size at t0")
    ax[1, 1].set_ylabel("first-harmonic share of ring profile")
    ax[1, 1].axhline(1 / 8, color="#bbbbbb", lw=0.8)
    for a in ax.flat:
        a.set_xlabel("seconds after input stops")
        a.grid(color="#dddddd", lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[1, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title}; still, >45 deg off axis; {layout_label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt", default=str(REPO_ROOT / "runs" / "bc" / "turns_s0" / "model_best.pt")
    )
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "cache_turns_test"))
    ap.add_argument(
        "--t0",
        type=float,
        nargs="+",
        default=np.arange(1.5, 8.01, 0.25).tolist(),
        help="seconds",
    )
    ap.add_argument("--dark-seconds", type=float, default=2.0)
    ap.add_argument("--still-rate", type=float, default=0.3, help="rad/s")
    ap.add_argument("--layout", default="anatomical")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument(
        "--out",
        default=None,
        help="directory for darkness.json (default: the checkpoint's)",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, cfg = load_checkpoint(ckpt, device)
    clocks = clocks_from_cfg(cfg)
    hz = clocks.sensory_hz
    data = BCData(T4T5Cache(args.cache), clocks)
    layout = model_layout(cfg, model, args.layout)
    t0s = [int(round(t * hz)) for t in args.t0]
    n_dark = int(round(args.dark_seconds * hz))
    blank = grey_t4t5(cfg, device)
    with torch.no_grad():
        s_blank = model.sensory(blank[None])[0].cpu().numpy()

    series = {"recurrent": [], "no_recurrence": [], "rest": []}
    still, off_axis, s_real, s_hover = [], [], [], []
    for start in range(0, len(data), args.chunk):
        idx = list(range(start, min(start + args.chunk, len(data))))
        batch = data.batch(idx, device)
        x = batch["x16"].float()
        yaw_rate = np.stack(
            [
                Trajectory.load(data.cache.trajectories[i], load_hexals=False).yaw_rate
                for i in idx
            ]
        )
        rel_yaw = (
            torch.atan2(batch["heading"][..., 0], batch["heading"][..., 1])
            .cpu()
            .numpy()
        )
        with torch.no_grad():
            s_real.append(model.sensory(x[:, : max(t0s)]).reshape(-1, 8).cpu().numpy())
            s_hover.append(model.sensory(x[:, :4]).mean(1).cpu().numpy())
            v0 = model.init_state(len(idx))
            r0 = model.core.rates(v0)
        snaps = snapshots(model, x, t0s)
        for t0, (v, r) in snaps.items():
            for cond, recurrent in (("recurrent", True), ("no_recurrence", False)):
                series[cond].append(
                    dark_series(
                        model,
                        dark_branch(model, v, r, blank, n_dark, recurrent),
                        layout,
                    )
                )
            series["rest"].append(
                dark_series(model, dark_branch(model, v0, r0, blank, n_dark), layout)
            )
            still.append(np.abs(yaw_rate[:, t0 - 50 : t0]).mean(1) < args.still_rate)
            off_axis.append(np.abs(rel_yaw[:, t0 - 1]) > np.deg2rad(45))
        del x, snaps
    series = {
        c: {k: np.concatenate([s[k] for s in parts]) for k in parts[0]}
        for c, parts in series.items()
    }
    still_off_axis = np.concatenate(still) & np.concatenate(off_axis)
    subsets = {"all": np.ones_like(still_off_axis), "still_off_axis": still_off_axis}

    steps = list(range(0, n_dark + 1, 5))
    taus = [k / hz for k in steps]
    at = [round(t * hz) for t in (0.5, 1.0, 2.0) if t <= args.dark_seconds]
    stats = {}
    for name, mask in subsets.items():
        ref = {k: v[mask] for k, v in series["recurrent"].items()}
        stats[name] = {
            c: dark_stats({k: v[mask] for k, v in s.items()}, steps + at, ref)
            for c, s in series.items()
        }
    n_steps = len(steps)
    curves = {
        name: {c: {k: v[:n_steps] for k, v in d.items()} for c, d in per.items()}
        for name, per in stats.items()
    }
    table = {
        name: {
            c: {
                k: dict(zip([f"{a / hz:g}s" for a in at], v[n_steps:], strict=True))
                for k, v in d.items()
            }
            for c, d in per.items()
        }
        for name, per in stats.items()
    }

    s_real, s_hover = np.concatenate(s_real), np.concatenate(s_hover).mean(0)
    report = {
        "checkpoint": str(ckpt),
        "cache": args.cache,
        "layout": layout.label,
        "t0_seconds": args.t0,
        "dark_seconds": args.dark_seconds,
        "n_samples": {name: int(m.sum()) for name, m in subsets.items()},
        "still_rate_rad_s": args.still_rate,
        "lptc_blank": {
            "channels": s_blank.round(3).tolist(),
            "hover_at_start": s_hover.round(3).tolist(),
            "real_mean": s_real.mean(0).round(3).tolist(),
            "real_sd": s_real.std(0).round(3).tolist(),
            "max_abs_blank_minus_hover_over_median_real_sd": float(
                np.abs(s_blank - s_hover).max() / np.median(s_real.std(0))
            ),
        },
        "at": table,
        "curves": {"tau_s": taus, **curves},
    }
    out_dir = Path(args.out) if args.out else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "darkness.json").write_text(json.dumps(report, indent=2))
    plot(out_dir / "darkness.png", taus, curves, layout.label, ckpt.parent.name)

    sub = table["still_off_axis"]
    print(f"{layout.label}; samples {report['n_samples']}")
    print("LPTC channels, blank / hover at episode start / real mean / real sd:")
    for k in ("channels", "hover_at_start", "real_mean", "real_sd"):
        print("  ", report["lptc_blank"][k])
    for c in ("recurrent", "no_recurrence", "rest"):
        for t in sub[c]["ring_retention"]:
            print(
                f"{c:14s} {t:>5s}  heading drift {sub[c]['heading_drift_deg_median'][t]:6.1f} deg  "
                f"ring drift {sub[c]['ring_drift_deg_median'][t]:6.1f} deg  "
                f"ring size {sub[c]['ring_retention'][t]:.2f}  decoder size {sub[c]['decoder_retention'][t]:.2f}  "
                f"fraction {sub[c]['ring_fraction'][t]:.2f}"
            )
    print(f"wrote {out_dir / 'darkness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
