#!/usr/bin/env python
"""Collect the per-run evaluations of a sweep and summarise them per condition.

Each run directory <root>/<mask>_<decoder>_s<seed>/ needs the outputs of, on model_best.pt:

  eval_open_loop.py --cache data/cache_turns_test --split all --out <run>/open_loop_test.json
  eval_topography.py            (topography.json)
  gate3_heading_decode.py       (runs/gate3/<run name>/report.json)
  eval_darkness.py              (darkness.json)
  eval_closed_loop.py --n 50 --skip-expert   (closed_loop.json)

Prints one row per run and mean +/- sd (n - 1) per condition, and says which condition pairs are
separated, meaning their per-seed ranges do not overlap. Writes <root>/results.json.

Usage:
  python scripts/summarize_sweep.py [--root runs/sweep_turns] [--gate3 runs/gate3]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from flybywire.config import REPO_ROOT


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run_metrics(run: Path, gate3: Path) -> dict:
    ol = read(run / "open_loop_test.json")
    topo = read(run / "topography.json")["reliable_anatomy"]
    g3 = read(gate3 / run.name / "report.json")
    dark = read(run / "darkness.json")["at"]["still_off_axis"]["recurrent"]
    cl = read(run / "closed_loop.json")["policy"]
    return {
        "test R2": ol["heading_r2"],
        "test err deg": ol["heading_err_deg"],
        "action MSE": ol["action_mse"],
        "topo align (activity)": topo["activity_phi"]["alignment"]["value"],
        "topo p (activity)": topo["activity_phi"]["alignment"]["p"],
        "topo align (decoder)": topo["decoder_phi"]["alignment"]["value"],
        "topo p (decoder)": topo["decoder_phi"]["alignment"]["p"],
        "gate3 bump share": g3["ring"]["anatomical"]["fraction_mean"],
        "gate3 alignment": g3["ring"]["anatomical"]["alignment"],
        "gate3 pass": float(g3["pass"]),
        "ref tuning share": g3["ring"]["tuning"]["fraction_mean"],
        "ref tuning alignment": g3["ring"]["tuning"]["alignment"],
        "dark drift 0.5s deg": dark["heading_drift_deg_median"]["0.5s"],
        "dark drift 1s deg": dark["heading_drift_deg_median"]["1s"],
        "dark drift 2s deg": dark["heading_drift_deg_median"]["2s"],
        "dark decoder size 2s": dark["decoder_retention"]["2s"],
        "dark ring size 2s": dark["ring_retention"]["2s"],
        "closed-loop success": cl["success_rate"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT / "runs" / "sweep_turns"))
    ap.add_argument("--gate3", default=str(REPO_ROOT / "runs" / "gate3"))
    args = ap.parse_args(argv)
    root, gate3 = Path(args.root), Path(args.gate3)

    runs = {}
    for d in sorted(root.glob("*_s[0-9]*")):
        m = re.fullmatch(r"(\w+?)_(learned|anatomical)_s(\d+)", d.name)
        if m and (d / "closed_loop.json").exists():
            runs[d.name] = {"condition": f"{m[1]}/{m[2]}", "seed": int(m[3]), **run_metrics(d, gate3)}
    conds = sorted({r["condition"] for r in runs.values()})
    names = [k for k in next(iter(runs.values())) if k not in ("condition", "seed")]

    print("per run")
    for name, r in runs.items():
        print(f"  {name:22s} " + "  ".join(f"{k} {r[k]:.3g}" for k in names))
    table = {}
    for c in conds:
        rows = [r for r in runs.values() if r["condition"] == c]
        table[c] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                        "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else float("nan"),
                        "min": float(min(r[k] for r in rows)), "max": float(max(r[k] for r in rows)),
                        "n": len(rows)} for k in names}
    print("\nmean +/- sd over seeds")
    print(f"{'':24s}" + "".join(f"{c:>26s}" for c in conds))
    for k in names:
        print(f"{k:24s}" + "".join(f"{table[c][k]['mean']:>14.3g} +/- {table[c][k]['sd']:<7.2g}" for c in conds))
    print("n seeds" + " " * 17 + "".join(f"{table[c][names[0]]['n']:>26d}" for c in conds))

    pairs = [(a, b) for a in conds for b in conds if a < b]
    separated = {f"{a} vs {b}": [k for k in names if table[a][k]["min"] > table[b][k]["max"]
                                 or table[b][k]["min"] > table[a][k]["max"]] for a, b in pairs}
    print("\nper-seed ranges do not overlap")
    for pair, ks in separated.items():
        print(f"  {pair}: {', '.join(ks) if ks else 'none'}")
    (root / "results.json").write_text(json.dumps({"runs": runs, "per_condition": table, "separated": separated},
                                                  indent=2))
    print(f"\nwrote {root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
