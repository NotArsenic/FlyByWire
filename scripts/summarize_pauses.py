#!/usr/bin/env python
"""Side-by-side table for the pause pilot: runs trained with and without pauses, same seed and mask.

Reads what the eval scripts wrote under <pilot>/ (default runs/step4_pilot):
  eval/<run>/open_loop.json         scripts/eval_open_loop.py --split all on data/cache_turns_test
  eval/<run>/gate3/report.json      scripts/gate3_heading_decode.py
  eval/<run>/pause_eval.json        scripts/eval_pauses.py
  darkness_diag/diagnose_darkness.json  scripts/diagnose_darkness.py (held last frame = "static", and grey)

Field flatness: the median drift of the decoded heading over the next second, per 20 degree bin of the
heading it starts from, over the bins with enough samples (see eval.darkness.velocity_field). A
continuous attractor would give 0 in every bin; ``rms`` is the root mean square over bins (deg per
second), ``stable`` the headings where the drift crosses zero from positive to negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SECONDS = ("0.5s", "1s", "2s")


def flatness(field: dict) -> dict:
    v = np.array(field["median_drift_1s_deg"])
    return {"n_bins": len(v), "rms": float(np.sqrt((v**2).mean())), "max_abs": float(np.abs(v).max()),
            "stable": field["stable_deg"]}


def load(path: Path):
    return json.loads(path.read_text())


def plot(path: Path, pilot: Path, runs: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_diag = load(pilot / "darkness_diag" / "diagnose_darkness.json")["runs"]
    diag = {r["run"]: r["velocity_field"]["static"] for r in runs_diag}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name in runs:
        fields = (diag[name], load(pilot / "eval" / name / "pause_eval.json")["velocity_field_1s"])
        for a, f in zip(ax, fields, strict=True):
            a.plot(f["centre_deg"], f["median_drift_1s_deg"], color="#2a78d6" if name.startswith("real") else "#eb6834",
                   ls="-" if "pause" in name else "--", marker="o", ms=3, label=name)
    for a, title in zip(ax, ("last frame held still, switched on at once", "pause with real flyvis transient"),
                        strict=True):
        a.axhline(0, color="#999999", lw=0.8)
        a.set_xlabel("decoded heading at the start (deg)")
        a.set_title(title, fontsize=9)
        a.grid(color="#dddddd", lw=0.5)
    ax[0].set_ylabel("median drift over the next 1 s (deg)")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(pilot: Path, runs: list[str]) -> dict:
    diag = {r["run"]: r for r in load(pilot / "darkness_diag" / "diagnose_darkness.json")["runs"]}
    out = {}
    for name in runs:
        ev = pilot / "eval" / name
        ol, g3, pe = load(ev / "open_loop.json"), load(ev / "gate3" / "report.json"), load(ev / "pause_eval.json")
        d = diag[name]
        row = {
            "test_r2": ol["heading_r2"],
            "test_err_deg": ol["heading_err_deg"],
            "action_mse": ol["action_mse"],
            "bump_share": g3["checks"]["single_bump"]["value"],
            "ring_alignment": g3["checks"]["bump_tracks_heading"]["value"],
        }
        for branch in ("static", "grey"):
            for sub in ("all", "still_off_axis"):
                for sec in SECONDS:
                    s = d["drift"][branch][sub][sec]
                    row[f"{branch}_{sub}_{sec}"] = {"mean": s["mean"], "mean_abs": s["mean_abs"]}
            row[f"{branch}_field"] = flatness(d["velocity_field"][branch])
        for sec, s in pe["drift_deg"]["all"].items():
            row[f"pause_all_{sec}"] = {"mean": s["mean"], "mean_abs": s["mean_abs"], "median_abs": s["median_abs"]}
        row["pause_field"] = flatness(pe["velocity_field_1s"])
        row["spliced_test"] = pe["spliced_test"]
        out[name] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default="runs/step4_pilot")
    ap.add_argument("--runs", nargs="+", default=["real_anatomical_s0", "real_pause_anatomical_s0",
                                                  "rewire_anatomical_s0", "rewire_pause_anatomical_s0"])
    args = ap.parse_args(argv)
    pilot = Path(args.pilot)
    table = summarize(pilot, args.runs)
    (pilot / "summary.json").write_text(json.dumps(table, indent=2))
    plot(pilot / "velocity_fields.png", pilot, args.runs)

    def line(label, f):
        print(f"{label:34s}" + "".join(f"{f(table[n]):>{max(len(n), 8) + 2}}" for n in args.runs))

    print(f"{'':34s}" + "".join(f"{n:>{max(len(n), 8) + 2}}" for n in args.runs))
    line("test R^2", lambda r: f"{r['test_r2']:.3f}")
    line("test heading error (deg)", lambda r: f"{r['test_err_deg']:.2f}")
    line("test action MSE", lambda r: f"{r['action_mse']:.4f}")
    line("Gate 3 bump share", lambda r: f"{r['bump_share']:.3f}")
    line("Gate 3 ring alignment", lambda r: f"{r['ring_alignment']:.3f}")
    for branch in ("static", "grey"):
        for sec in SECONDS:
            line(f"{branch} {sec}: mean / mean|.| (deg)",
                 lambda r, b=branch, s=sec: f"{r[f'{b}_all_{s}']['mean']:.1f} / {r[f'{b}_all_{s}']['mean_abs']:.1f}")
        line(f"{branch} field rms / max (deg/s)", lambda r, b=branch: f"{r[f'{b}_field']['rms']:.1f} / "
             f"{r[f'{b}_field']['max_abs']:.1f}")
        line(f"{branch} field stable points", lambda r, b=branch: str(len(r[f"{b}_field"]["stable"])))
    for sec in table[args.runs[0]]:
        if sec.startswith("pause_all_"):
            line(f"pause {sec[10:]}: mean / mean|.| (deg)",
                 lambda r, s=sec: f"{r[s]['mean']:.1f} / {r[s]['mean_abs']:.1f}")
    line("pause field rms / max (deg/s)",
         lambda r: f"{r['pause_field']['rms']:.1f} / {r['pause_field']['max_abs']:.1f}")
    line("pause field stable points", lambda r: str(len(r["pause_field"]["stable"])))
    line("spliced test R^2", lambda r: f"{r['spliced_test']['heading_r2']:.3f}")
    line("err in / out of pauses (deg)", lambda r: f"{r['spliced_test']['err_at_pause_decisions_deg']:.1f} / "
         f"{r['spliced_test']['err_at_other_decisions_deg']:.1f}")
    print(f"wrote {pilot / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
