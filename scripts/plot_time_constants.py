#!/usr/bin/env python
"""Learned single-neuron time constants per cell type, and what they allow in the darkness test.

tau = dt / alpha with alpha = sigmoid(a) and dt = 10 ms (the exact decay time of the discrete
update, -dt / ln(1 - alpha), is shorter for fast neurons). If EPG time constants were long compared
with the 2 s dark window, a bump could persist in single neurons. With the recurrent weights zeroed
(`recurrence_off`) every neuron only relaxes to its bias with its own alpha, so that ablation is
the model "no coupling, own tau"; its ring-size curve is read from each run's darkness.json.

Usage:
  python scripts/plot_time_constants.py [--runs "runs/sweep_turns/*_s[012]/model_best.pt"] [--out runs/darkness_diag]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from flybywire.config import REPO_ROOT
from flybywire.eval.closed_loop import load_checkpoint

TYPES = ("ER", "EPG", "PEN", "PEG", "D7", "PFL3")
BLUE, ORANGE = "#2a78d6", "#eb6834"
DT_MS = 10.0
PERCENTILES = (0, 10, 50, 90, 100)
LONGER_THAN_MS = (100, 200, 500, 2000)


def tau_ms(model, group) -> np.ndarray:
    alpha = model.core.alpha().detach().numpy()[model.groups[group]]
    return DT_MS / alpha


def decoder_weighted_median(tau, weight) -> float:
    order = np.argsort(tau)
    cum = np.cumsum(weight[order]) / weight.sum()
    return float(tau[order][np.searchsorted(cum, 0.5)])


def half_time(t, y):
    """First time y drops below half its starting value, None if it never does."""
    below = np.flatnonzero(np.asarray(y) < 0.5 * y[0])
    return float(t[below[0]]) if below.size else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            str(REPO_ROOT / "runs" / "sweep_turns" / "*_s[012]" / "model_best.pt")
        ],
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "darkness_diag"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for ckpt in sorted(p for pat in args.runs for p in glob.glob(pat)):
        model, cfg = load_checkpoint(ckpt, "cpu")
        name = Path(ckpt).parent.name
        taus = {g: tau_ms(model, g) for g in TYPES}
        w = model.heading.weight.detach().abs().sum(0).numpy()
        dark = json.loads((Path(ckpt).parent / "darkness.json").read_text())["curves"]
        curves = dark["still_off_axis"]
        t = np.array(dark["tau_s"])
        runs[name] = {
            "tau": taus,
            "decoder_weight": w,
            "t": t,
            "ring_recurrent": np.array(curves["recurrent"]["ring_retention"]),
            "ring_off": np.array(curves["no_recurrence"]["ring_retention"]),
        }

    report = {
        "dt_ms": DT_MS,
        "alpha_init": 0.2,
        "tau_at_init_ms": DT_MS / 0.2,
        "runs": {},
        "by_condition": {},
    }
    for name, r in runs.items():
        epg = r["tau"]["EPG"]
        report["runs"][name] = {
            "tau_ms": {
                g: dict(
                    zip(
                        ("min", "p10", "median", "p90", "max"),
                        np.percentile(r["tau"][g], PERCENTILES).round(1).tolist(),
                        strict=True,
                    )
                )
                | {
                    f"frac_over_{c}ms": float((r["tau"][g] > c).mean())
                    for c in LONGER_THAN_MS
                }
                for g in TYPES
            },
            "epg_decoder_weighted_median_tau_ms": round(
                decoder_weighted_median(epg, r["decoder_weight"]), 1
            ),
            "ring_size_half_time_s": {
                "recurrent": half_time(r["t"], r["ring_recurrent"]),
                "recurrence_off": half_time(r["t"], r["ring_off"]),
            },
            "ring_size_at_0.5s": {
                "recurrent": float(np.interp(0.5, r["t"], r["ring_recurrent"])),
                "recurrence_off": float(np.interp(0.5, r["t"], r["ring_off"])),
            },
            "ring_size_at_2s": {
                "recurrent": float(np.interp(2.0, r["t"], r["ring_recurrent"])),
                "recurrence_off": float(np.interp(2.0, r["t"], r["ring_off"])),
            },
        }
    for g in TYPES:
        med = {
            c: np.array(
                [
                    report["runs"][n]["tau_ms"][g]["median"]
                    for n in runs
                    if n.startswith(c)
                ]
            )
            for c in ("real", "rewire")
        }
        report["by_condition"][g] = {
            "median_tau_ms_per_run": {c: v.tolist() for c, v in med.items()},
            "mean_of_run_medians_ms": {
                c: round(float(v.mean()), 1) for c, v in med.items()
            },
            "mannwhitney_p_run_medians": float(
                mannwhitneyu(med["real"], med["rewire"]).pvalue
            ),
        }
    (out_dir / "time_constants.json").write_text(json.dumps(report, indent=2))
    plot(out_dir / "time_constants.png", runs)
    for n, r in report["runs"].items():
        e = r["tau_ms"]["EPG"]
        print(
            f"{n:22s} EPG tau ms: min {e['min']:6.1f} median {e['median']:6.1f} p90 {e['p90']:6.1f} "
            f"max {e['max']:6.1f}  >500ms {e['frac_over_500ms']:.3f}  ring size at 0.5 s: "
            f"recurrent {r['ring_size_at_0.5s']['recurrent']:.2f} off {r['ring_size_at_0.5s']['recurrence_off']:.2f}"
        )
    print(f"wrote {out_dir / 'time_constants.json'}")
    return 0


def plot(path: Path, runs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for a, g in zip(ax.flat, TYPES, strict=False):
        for name, r in runs.items():
            tau = np.sort(r["tau"][g])
            a.step(
                tau,
                np.arange(1, len(tau) + 1) / len(tau),
                where="post",
                lw=1.2,
                alpha=0.8,
                color=BLUE if name.startswith("real") else ORANGE,
                ls="--" if "anatomical" in name else "-",
            )
        a.set_xscale("log")
        a.set_xlim(3, 10000)
        a.set_title(f"{g} (n = {len(r['tau'][g])})", fontsize=9)
        a.set_xlabel("tau = dt / alpha (ms)")
        a.set_ylabel("fraction of neurons")
        for sec in (0.5, 2.0):
            a.axvline(sec * 1000, color="#999999", lw=0.8)
        a.axvline(DT_MS / 0.2, color="#999999", lw=0.8, ls=":")
    ax[0, 1].text(520, 0.05, "0.5 s", fontsize=7, color="#52514e")
    ax[0, 1].text(2100, 0.05, "2 s", fontsize=7, color="#52514e")
    ax[0, 1].text(52, 0.05, "init", fontsize=7, color="#52514e")

    a = ax[1, 2]
    for name, r in runs.items():
        c = BLUE if name.startswith("real") else ORANGE
        a.plot(r["t"], r["ring_recurrent"], color=c, lw=1.2, alpha=0.8)
        a.plot(r["t"], r["ring_off"], color=c, lw=1.2, alpha=0.8, ls=":")
    a.set_xlabel("seconds after input stops (grey)")
    a.set_ylabel("ring bump size / size at t0")
    a.set_title("solid: trained; dotted: recurrent weights zeroed", fontsize=9)

    leg = ax[1, 3]
    leg.axis("off")
    for label, c, ls in (
        ("real mask, learned decoder", BLUE, "-"),
        ("real mask, anatomical decoder", BLUE, "--"),
        ("rewired mask, learned decoder", ORANGE, "-"),
        ("rewired mask, anatomical decoder", ORANGE, "--"),
    ):
        leg.plot([], [], color=c, ls=ls, label=label)
    leg.legend(loc="upper left", frameon=False, fontsize=9, title="3 seeds each")
    for a in ax.flat[:7]:
        a.grid(color="#dddddd", lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
