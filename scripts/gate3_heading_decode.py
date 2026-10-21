#!/usr/bin/env python
"""Gate 3: does the trained core carry a heading representation in EPG, as a bump that rotates with yaw?

A BC checkpoint runs open loop over held-out episodes (default data/cache_turns_test, never trained
on). Three conditions, all required, fixed before any result was looked at:

  1. Heading decode from EPG: R^2 > 0.8 at lag 0, joint on (sin, cos), all frames (the decoder's output).
  2. Single bump: mean first-harmonic share of the 16-wedge ring profile > 0.5 per frame
     (pure cosine 1, noise about 0.125), after a 50-step washout.
  3. The bump rotates with yaw: alignment of the bump angle (population vector of the profile)
     with the true heading > 0.8, invariant to the ring's zero and handedness. 1 means the bump sits
     at s * heading + c for fixed c and s = +/-1. The report also gives a null, the alignment when
     each episode's bump is paired with another episode's heading.

The anatomical ring entry also carries ``fraction_mean_floor_subtracted``: the same share after each
neuron's resting rate (its 5th percentile over the scored frames) is taken off. It is a check on
the threshold's reachability (runs/bump_check), not a second gate.

Conditions 2 and 3 are scored on the anatomical layout (EPG neurons binned by their angle around
the ellipsoid body). The tuning layout (neurons binned by their decoder weights) is reported as a
labelled reference only: the decoder is trained to read heading from those neurons, so it shows a
bump by construction.

Writes <out>/report.json and <out>/ring_vs_heading.png (default <out>: runs/gate3/<checkpoint's run>).

Usage:
  python scripts/gate3_heading_decode.py [--ckpt runs/bc/turns_s0/model_best.pt]
                                         [--cache data/cache_turns_test] [--out DIR]
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
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.metrics import angular_error_deg, heading_r2
from flybywire.eval.ring import (
    bump,
    model_layout,
    rate_floor,
    ring_activity,
    ring_tracking,
)
from flybywire.train.bc import BCData

MIN_R2 = 0.8
MIN_FRACTION = 0.5
MIN_ALIGNMENT = 0.8
WASHOUT = 50


@torch.no_grad()
def run_episodes(model, data: BCData, device, chunk: int = 20):
    """Per episode: EPG rates (T, n_epg), decoded (sin, cos) (T, 2) and true heading (T,) in rad."""
    rates, decoded, heading = [], [], []
    for start in range(0, len(data), chunk):
        idx = list(range(start, min(start + chunk, len(data))))
        batch = data.batch(idx, device)
        out = model.run(batch["x16"].float(), model.init_state(len(idx)), t0=0)
        epg = out["r"].index_select(-1, model.heading.source_index).cpu().numpy()
        true = (
            torch.atan2(batch["heading"][..., 0], batch["heading"][..., 1])
            .cpu()
            .numpy()
        )
        for j, i in enumerate(idx):
            n = data.cache.length(i)
            rates.append(epg[j, :n])
            decoded.append(out["heading"][j, :n].cpu().numpy())
            heading.append(true[j, :n])
    return rates, decoded, heading


def plot(path: Path, panels: list[dict], heading, example: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(panels), 3, figsize=(14, 4.2 * len(panels)))
    for row, p in zip(ax, panels, strict=True):
        label, prof, bumps, rep = (
            p["label"],
            p["profile"][example],
            p["bumps"],
            p["report"],
        )
        t = np.arange(len(prof)) / 100
        n = prof.shape[1]
        row[0].imshow(
            prof.T,
            origin="lower",
            aspect="auto",
            extent=(0, t[-1], 0, n),
            cmap="viridis",
        )
        pos = (
            (
                (rep["sign"] * heading[example] + np.radians(rep["offset_deg"]))
                % (2 * np.pi)
            )
            / (2 * np.pi)
            * n
        )
        row[0].plot(
            t,
            np.where(np.abs(np.diff(pos, prepend=pos[0])) > n / 2, np.nan, pos),
            color="w",
            lw=1.2,
        )
        row[0].set_xlabel("time (s)")
        row[0].set_ylabel("wedge")
        row[0].set_title(
            f"{label}\nepisode {example}; white: true heading on the fitted ring",
            fontsize=9,
        )
        h = np.concatenate([x[WASHOUT:] for x in heading])
        a = np.concatenate([b["angle"][WASHOUT:] for b in bumps])
        row[1].hist2d(np.degrees(h), np.degrees(a), bins=72, cmap="Blues", norm="log")
        row[1].set_xlabel("true heading (deg)")
        row[1].set_ylabel("bump angle (deg)")
        row[1].set_title(
            f"alignment {rep['alignment']:.2f} (null {rep['alignment_null']:.2f})",
            fontsize=9,
        )
        f = np.concatenate([b["fraction"][WASHOUT:] for b in bumps])
        row[2].hist(f, bins=50, range=(0, 1), color="#4c78a8")
        row[2].axvline(MIN_FRACTION, color="k", lw=1, ls="--")
        row[2].axvline(1 / 8, color="#999999", lw=1)
        row[2].set_xlabel("first-harmonic share of the ring profile")
        row[2].set_title(f"mean {rep['fraction_mean']:.2f}", fontsize=9)
        for a_ in row[1:]:
            for side in ("top", "right"):
                a_.spines[side].set_visible(False)
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, cfg = load_checkpoint(ckpt, device)
    data = BCData(T4T5Cache(args.cache), clocks_from_cfg(cfg))
    rates, decoded, heading = run_episodes(model, data, device)

    h_all = np.concatenate(heading)
    pred, true = np.concatenate(decoded), np.stack([np.sin(h_all), np.cos(h_all)], 1)
    decode = {"r2": heading_r2(pred, true), "err_deg": angular_error_deg(pred, true)}

    panels = []
    for mode, role in (
        ("anatomical", "gate"),
        ("tuning", "reference only, bump by construction"),
    ):
        layout = model_layout(cfg, model, mode)
        profile = [ring_activity(r, layout) for r in rates]
        bumps = [bump(p) for p in profile]
        panels.append(
            {
                "label": f"{layout.label} ({role})",
                "profile": profile,
                "bumps": bumps,
                "mode": mode,
                "layout": layout,
                "report": ring_tracking(bumps, heading, WASHOUT),
            }
        )
    anat = panels[0]["report"]
    floor = rate_floor(rates, WASHOUT)
    resting = [
        bump(ring_activity(r - floor, panels[0]["layout"]))["fraction"][WASHOUT:]
        for r in rates
    ]
    anat["fraction_mean_floor_subtracted"] = float(np.concatenate(resting).mean())
    checks = {
        "heading_r2": {
            "value": decode["r2"],
            "threshold": MIN_R2,
            "pass": decode["r2"] > MIN_R2,
        },
        "single_bump": {
            "value": anat["fraction_mean"],
            "threshold": MIN_FRACTION,
            "pass": anat["fraction_mean"] > MIN_FRACTION,
        },
        "bump_tracks_heading": {
            "value": anat["alignment"],
            "threshold": MIN_ALIGNMENT,
            "pass": anat["alignment"] > MIN_ALIGNMENT,
        },
    }
    report = {
        "checkpoint": str(ckpt),
        "cache": args.cache,
        "n_episodes": len(data),
        "washout_core_steps": WASHOUT,
        "pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "decode": decode,
        "ring": {p["mode"]: {"layout": p["label"], **p["report"]} for p in panels},
    }
    out_dir = (
        Path(args.out) if args.out else REPO_ROOT / "runs" / "gate3" / ckpt.parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    span = [np.ptp(np.unwrap(h)) for h in heading]
    plot(out_dir / "ring_vs_heading.png", panels, heading, int(np.argmax(span)))

    for k, c in checks.items():
        print(
            f"{k:22s} {c['value']:.3f}  (> {c['threshold']})  {'pass' if c['pass'] else 'FAIL'}"
        )
    print(
        f"share with resting rates subtracted: {anat['fraction_mean_floor_subtracted']:.3f}"
    )
    ref = report["ring"]["tuning"]
    print(
        f"reference, tuning layout: fraction {ref['fraction_mean']:.3f}, alignment {ref['alignment']:.3f}"
    )
    print(
        f"Gate 3 {'PASS' if report['pass'] else 'FAIL'}; wrote {out_dir / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
