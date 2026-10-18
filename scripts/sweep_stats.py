#!/usr/bin/env python
"""Real vs rewired over seeds: the per-run table and the exact permutation tests of PROTOCOL.md.

Reads, for each run <root>/<mask>_<decoder>_s<seed>/, its topography.json and
<gate3>/<run>/report.json (scripts/eval_topography.py and scripts/gate3_heading_decode.py on
model_best.pt). Prints every per-seed value, mean / sd / min / max per mask, and for each metric
the difference of means (real - rewire) with its exact two-sided permutation p over every
relabeling of the runs, once for all seeds and once for seeds from --fresh-from on (the seeds
the hypothesis was not picked on). Writes <root>/stats.json (stats_<decoder>.json for a decoder
other than anatomical).

Usage:
  python scripts/sweep_stats.py [--decoder anatomical|learned] [--root runs/sweep_turns]
                                [--gate3 runs/gate3] [--fresh-from 3]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from flybywire.config import REPO_ROOT

SHORT = {"bump share": "share", "bump share, resting rate subtracted": "share-rest", "heading R2": "R2",
         "heading err deg": "err deg", "ring alignment": "ring align", "topography alignment (activity)": "topo act",
         "topography alignment (decoder)": "topo dec", "circ corr, signed (activity)": "cc act",
         "circ corr, abs (activity)": "|cc| act", "circ corr, signed (decoder)": "cc dec",
         "circ corr, abs (decoder)": "|cc| dec"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run_values(run: Path, gate3: Path, decoder: str) -> tuple[dict, dict]:
    """Tested metrics and the permutation p values of the topography scores."""
    g3 = read(gate3 / run.name / "report.json")
    ring = g3["ring"]["anatomical"]
    topo = read(run / "topography.json")["reliable_anatomy"]
    values = {
        "bump share": ring["fraction_mean"],
        "bump share, resting rate subtracted": ring["fraction_mean_floor_subtracted"],
        "heading R2": g3["decode"]["r2"],
        "heading err deg": g3["decode"]["err_deg"],
        "ring alignment": ring["alignment"],
    }
    p = {}
    # the anatomical readout's weights are the EB angles, so only the activity phi says anything
    for which in ("activity",) if decoder == "anatomical" else ("activity", "decoder"):
        t = topo[f"{which}_phi"]
        values[f"topography alignment ({which})"] = t["alignment"]["value"]
        p[f"topography alignment ({which})"] = t["alignment"]["p"]
        if decoder != "anatomical":
            values[f"circ corr, signed ({which})"] = t["circ_corr_signed"]
            values[f"circ corr, abs ({which})"] = t["circ_corr_abs"]["value"]
            p[f"circ corr, abs ({which})"] = t["circ_corr_abs"]["p"]
    return values, p


def exact_permutation(real, rewire) -> dict:
    """Two-sided exact test of mean(real) - mean(rewire) over every way to relabel the runs."""
    x = np.concatenate([real, rewire]).astype(np.float64)
    n, k = len(x), len(real)
    chosen = np.array(list(itertools.combinations(range(n), k)))
    picked = x[chosen].sum(1)
    diff = picked / k - (x.sum() - picked) / (n - k)
    observed = float(np.mean(real) - np.mean(rewire))
    return {"difference": observed, "p": float((np.abs(diff) >= abs(observed) - 1e-12).mean()),
            "n_relabelings": len(chosen)}


def spread(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", default="anatomical", choices=("anatomical", "learned"))
    ap.add_argument("--root", default=str(REPO_ROOT / "runs" / "sweep_turns"))
    ap.add_argument("--gate3", default=str(REPO_ROOT / "runs" / "gate3"))
    ap.add_argument("--fresh-from", type=int, default=3)
    args = ap.parse_args(argv)
    root, gate3 = Path(args.root), Path(args.gate3)

    found = []
    for d in root.glob(f"*_{args.decoder}_s[0-9]*"):
        m = re.fullmatch(rf"(real|rewire)_{args.decoder}_s(\d+)", d.name)
        if m and (d / "topography.json").exists() and (gate3 / d.name / "report.json").exists():
            found.append((m[1], int(m[2]), d))
    runs = {}
    for mask, seed, d in sorted(found, key=lambda f: f[:2]):
        values, p = run_values(d, gate3, args.decoder)
        runs[d.name] = {"mask": mask, "seed": seed, "values": values, "p": p}
    names = list(next(iter(runs.values()))["values"])

    def group(mask, seeds, metric):
        return [r["values"][metric] for r in runs.values() if r["mask"] == mask and r["seed"] in seeds]

    all_seeds = {r["seed"] for r in runs.values()}
    fresh = {s for s in all_seeds if s >= args.fresh_from}
    print("per run")
    print(f"  {'run':22s}" + "".join(f"{SHORT[k]:>12s}" for k in names))
    for name, r in runs.items():
        print(f"  {name:22s}" + "".join(f"{r['values'][k]:>12.4f}" for k in names))
    print("\npermutation p of the topography scores (label shuffle of the EB angles, one-sided)")
    for name, r in runs.items():
        print(f"  {name:22s}" + "".join(f"  {SHORT[k]} {v:.4f}" for k, v in r["p"].items()))
    stats = {}
    for metric in names:
        stats[metric] = {"per_mask": {m: spread(group(m, all_seeds, metric)) for m in ("real", "rewire")},
                         "test_all_seeds": exact_permutation(group("real", all_seeds, metric),
                                                             group("rewire", all_seeds, metric))}
        if len(fresh) > 1:
            stats[metric][f"test_seeds_{min(fresh)}_{max(fresh)}"] = exact_permutation(
                group("real", fresh, metric), group("rewire", fresh, metric))
    print(f"\nmean +/- sd (n - 1) [min, max]; difference real - rewire, exact permutation p "
          f"(all {len(all_seeds)} seeds | seeds {min(fresh) if fresh else '-'}+)")
    for metric, s in stats.items():
        cells = "   ".join(f"{m} {c['mean']:.4f} +/- {c['sd']:.4f} [{c['min']:.3f}, {c['max']:.3f}]"
                           for m, c in s["per_mask"].items())
        tests = [v for k, v in s.items() if k.startswith("test_")]
        print(f"  {metric:36s} {cells}   diff {tests[0]['difference']:+.4f}   p "
              + " | ".join(f"{t['p']:.4f}" for t in tests))
    out = root / ("stats.json" if args.decoder == "anatomical" else f"stats_{args.decoder}.json")
    out.write_text(json.dumps({"runs": runs, "stats": stats}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
