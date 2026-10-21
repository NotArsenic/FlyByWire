#!/usr/bin/env python
"""What the Gate 3 bump share measures: synthetic bumps of known width, and the floor of real runs.

The share (``flybywire.eval.ring.bump``, key ``fraction``) is, per frame, the 16 wedge values (mean
rate of the reliable EPG neurons whose EB angle falls in the wedge), P = rfft of that profile, and
|P_1|^2 / sum_{k=1..8} |P_k|^2. The mean, k = 0, is not in the denominator, so adding the same
rate to every wedge cannot change it; only a floor that differs between wedges can. Gate 3 scores
the mean over all frames after a 50-step washout.

Synthetic part: Gaussian and von Mises bumps of FWHM 10-360 deg, centre swept over the ring, read
either at the 16 wedge centres ("ideal") or through the real layout (47 neurons at their EB angles,
2-4 per wedge). Gives share -> FWHM and shows the effect of a uniform floor.

Real part, per run: the EPG rates on the held-out cache give each neuron's resting rate (5th
percentile) and a typical bump height; a perfect bump of that height on that floor is scored, to
see whether 0.5 can be reached with the real floor.

Usage:
  python scripts/bump_sharpness_check.py [--runs DIR ...] [--cache data/cache_turns_test] [--out runs/bump_check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate3_heading_decode import WASHOUT, run_episodes  # noqa: E402

from flybywire.clocks import clocks_from_cfg  # noqa: E402
from flybywire.config import REPO_ROOT, load_config  # noqa: E402
from flybywire.connectome.masks import (
    load_base_connectome,
    load_eb_angles,
)  # noqa: E402
from flybywire.data.cache import T4T5Cache  # noqa: E402
from flybywire.eval.closed_loop import load_checkpoint  # noqa: E402
from flybywire.eval.ring import (  # noqa: E402
    RingLayout,
    anatomical_wedges,
    bump,
    model_layout,
    rate_floor,
    ring_activity,
)
from flybywire.train.bc import BCData  # noqa: E402

N_WEDGES = 16
CENTRES = (np.arange(N_WEDGES) + 0.5) * 2 * np.pi / N_WEDGES
POS = np.linspace(0, 2 * np.pi, 96, endpoint=False)  # bump centres; 6 per wedge
WIDTHS = np.arange(10, 361, 1.0)
SHOWN = (10, 20, 30, 40, 53, 60, 80, 100, 120, 150, 180)
TARGETS = (0.13, 0.15, 0.19, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)
SOFTPLUS_ZERO = float(np.log(2))
DEFAULT_RUNS = [
    f"runs/sweep_turns/{m}_anatomical_s{s}"
    for m in ("real", "rewire")
    for s in range(3)
]


def gaussian(x, fwhm_deg):
    sigma = np.radians(fwhm_deg) / (2 * np.sqrt(2 * np.log(2)))
    return sum(np.exp(-0.5 * ((x + 2 * np.pi * k) / sigma) ** 2) for k in range(-2, 3))


def von_mises(x, fwhm_deg):
    kappa = np.log(2) / (1 - np.cos(np.radians(fwhm_deg) / 2))
    return np.exp(kappa * (np.cos(x) - 1))


SHAPES = {"gaussian": gaussian, "von_mises": von_mises}


def ideal_share(shape, fwhm, floor=0.0):
    return bump(floor + shape(CENTRES[None] - POS[:, None], fwhm))["fraction"]


def layout_share(shape, fwhm, layout, angle, floor=0.0, amp=1.0):
    """Neuron i fires floor_i + amp * shape(angle_i - centre); the ring reads wedge means of that."""
    return bump(
        ring_activity(floor + amp * shape(angle[None] - POS[:, None], fwhm), layout)
    )["fraction"]


def curve(share_at):
    s = np.stack([share_at(w) for w in WIDTHS])
    return {"mean": s.mean(1), "min": s.min(1), "max": s.max(1)}


def width_for(share, c):
    """FWHM whose mean share is ``share``, and every FWHM whose min-max band over bump centres covers it."""
    inside = WIDTHS[(c["min"] <= share) & (share <= c["max"])]
    mean = np.maximum.accumulate(c["mean"])
    return {
        "fwhm": (
            float(np.interp(share, mean, WIDTHS)) if share >= mean[0] else None
        ),  # None: narrower than 10
        "range": [float(inside.min()), float(inside.max())] if len(inside) else None,
    }


def noise_share(n: int = 20000) -> float:
    return float(
        bump(np.random.default_rng(0).normal(size=(n, N_WEDGES)))["fraction"].mean()
    )


def real_run(run: Path, cache: str, device: str, angle) -> dict:
    model, cfg = load_checkpoint(run / "model_best.pt", device)
    rates, _, _ = run_episodes(
        model, BCData(T4T5Cache(cache), clocks_from_cfg(cfg)), device
    )
    layout = model_layout(cfg, model, "anatomical")
    floor = rate_floor(rates, WASHOUT)
    profile = np.concatenate([ring_activity(r[WASHOUT:], layout) for r in rates])
    resting = np.concatenate(
        [ring_activity(r[WASHOUT:] - floor, layout) for r in rates]
    )
    amp = float((profile.max(1) - np.median(profile, 1)).mean())
    out = {
        "share": float(bump(profile)["fraction"].mean()),
        "share_floor_subtracted": float(bump(resting)["fraction"].mean()),
        "share_wedges_shuffled": float(
            bump(np.random.default_rng(0).permuted(profile, axis=1))["fraction"].mean()
        ),
        "floor_mean": float(floor.mean()),
        "floor_max": float(floor.max()),
        "bump_height": amp,
    }
    for name, f in (("no_floor", 0.0), ("real_floor", floor)):
        c = curve(lambda w, f=f: layout_share(gaussian, w, layout, angle, f, amp))
        out[f"perfect_bump_{name}"] = {
            "max_share": float(c["mean"].max()),
            **width_for(0.5, c),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "cache_turns_test"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "bump_check"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config()
    angle, reliable = load_eb_angles(cfg, load_base_connectome(cfg))
    layout = RingLayout(
        anatomical_wedges(angle, reliable, N_WEDGES), "anatomical", N_WEDGES
    )

    ideal = {s: curve(lambda w, f=f: ideal_share(f, w)) for s, f in SHAPES.items()}
    real = {
        s: curve(lambda w, f=f: layout_share(f, w, layout, angle))
        for s, f in SHAPES.items()
    }
    flat = [
        float(np.abs(ideal_share(gaussian, 53, f) - ideal_share(gaussian, 53)).max())
        for f in (SOFTPLUS_ZERO, 5.0)
    ]
    table = {
        s: {
            int(w): {
                k: [float(c[k][int(w) - 10]) for k in ("mean", "min", "max")]
                for k, c in (("ideal", ideal[s]), ("layout", real[s]))
            }
            for w in SHOWN
        }
        for s in SHAPES
    }
    lookup = [
        {"share": t, **{s: width_for(t, real[s]) for s in SHAPES}} for t in TARGETS
    ]
    report = {
        "definition": (
            "share = |P_1|^2 / sum_{k=1..8} |P_k|^2, P = rfft of the 16 wedge means, per frame; wedge = "
            "mean rate of the reliable EPG neurons with EB angle in it; mean (k = 0) not in the "
            f"denominator; mean over frames after a {WASHOUT}-step washout"
        ),
        "noise_share": noise_share(),
        "uniform_floor_max_abs_change": {"softplus(0)=0.693": flat[0], "5.0": flat[1]},
        "table_fwhm_deg": table,
        "lookup_layout": lookup,
        "runs": {
            Path(r).name: real_run(Path(r), args.cache, device, angle)
            for r in args.runs
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(
        f"white-noise share {report['noise_share']:.3f}; uniform floor changes the 53 deg share by "
        f"{max(flat):.1e} at most"
    )
    print("FWHM  gaussian ideal / layout (min-max)        von Mises ideal / layout")
    for w in SHOWN:
        g, v = table["gaussian"][w], table["von_mises"][w]
        print(
            f"{w:4d}  {g['ideal'][0]:.3f} / {g['layout'][0]:.3f} ({g['layout'][1]:.2f}-{g['layout'][2]:.2f})"
            f"        {v['ideal'][0]:.3f} / {v['layout'][0]:.3f}"
        )
    print(
        "share -> FWHM (deg), real layout; range = widths whose min-max over bump centres covers the share"
    )
    for row in lookup:
        g, v = row["gaussian"], row["von_mises"]
        width = lambda w: (
            "<10" if w["fwhm"] is None else f"{w['fwhm']:.0f}"
        )  # noqa: E731
        print(
            f"  {row['share']:.2f}  gaussian {width(g):>3s} {g['range']}   von Mises {width(v):>3s} {v['range']}"
        )
    print(
        "runs: share, floor-subtracted, shuffled-wedge null, floor mean, bump height, "
        "FWHM reaching 0.5 (no floor / real floor)"
    )
    for name, r in report["runs"].items():
        print(
            f"  {name:22s} {r['share']:.3f} {r['share_floor_subtracted']:.3f} {r['share_wedges_shuffled']:.3f}  "
            f"{r['floor_mean']:.3f} {r['bump_height']:.2f}  {r['perfect_bump_no_floor']['fwhm']:.1f} / "
            f"{r['perfect_bump_real_floor']['fwhm']:.1f}"
        )
    print(f"wrote {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
