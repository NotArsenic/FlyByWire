#!/usr/bin/env python
"""The tests of docs/protocols/sweep_uniform_pauses.md on runs/sweep_uniform.

Per run (<arm>_<mask>_s<seed>): pause drift (mean |drift| 2 s into a held frame), velocity-field rms
and stable points (pause_eval.json), bump share (gate3/<run>/report.json), closed-loop success
(failure_flights.npz, uniform arm only). Every test is an exact permutation test over relabelings
of runs, two-sided on |difference of means|:

  H1  field rms, uniform vs control, arm labels shuffled within each mask; plus how many runs
      have a stable point within 20 deg of +85 or -85 (and of 0, descriptive)
  H2  real vs rewired in the uniform arm on drift, field rms and bump share, Holm over the three
  H3  closed-loop success, real vs rewired in the uniform arm

Usage:
  python scripts/sweep_uniform_stats.py [--root runs/sweep_uniform]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from flybywire.config import REPO_ROOT

NEAR_DEG = 20.0


def run_values(run: Path, root: Path) -> dict:
    pe = json.loads((run / "pause_eval.json").read_text())
    field = pe["velocity_field_1s"]
    v = np.array(field["median_drift_1s_deg"])
    stable = np.array(field["stable_deg"])
    g3 = json.loads((root / "gate3" / run.name / "report.json").read_text())
    out = {
        "drift_2s": pe["drift_deg"]["all"]["2s"]["mean_abs"],
        "field_rms": float(np.sqrt((v**2).mean())),
        "stable_deg": stable.tolist(),
        "stable_near_85": bool((np.abs(np.abs(stable) - 85) < NEAR_DEG).any()),
        "stable_near_0": bool((np.abs(stable) < NEAR_DEG).any()),
        "bump_share": g3["ring"]["anatomical"]["fraction_mean"],
        "heading_r2": g3["decode"]["r2"],
        "spliced_r2": pe["spliced_test"]["heading_r2"],
    }
    flights = run / "failure_flights.npz"
    if flights.exists():
        reasons = np.load(flights)["reasons"]
        out["success"] = float(np.isin(reasons, ["time", "end"]).mean())
        out["n_flights"] = int(len(reasons))
    return out


def perm_p(a, b, strata_a=None, strata_b=None) -> tuple[float, float]:
    """Difference of means a - b and its exact two-sided p; with strata, labels move only within a stratum."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    if strata_a is None:
        x = np.concatenate([a, b])
        diffs = []
        for pick in itertools.combinations(range(len(x)), len(a)):
            m = np.zeros(len(x), bool)
            m[list(pick)] = True
            diffs.append(x[m].mean() - x[~m].mean())
    else:
        groups = sorted(set(strata_a) | set(strata_b))
        per = []
        for g in groups:
            xa = [v for v, s in zip(a, strata_a, strict=True) if s == g]
            xb = [v for v, s in zip(b, strata_b, strict=True) if s == g]
            x = np.array(xa + xb)
            per.append([(x[list(p)].sum(), x.sum() - x[list(p)].sum())
                        for p in itertools.combinations(range(len(x)), len(xa))])
        diffs = [sum(s[0] for s in combo) / len(a) - sum(s[1] for s in combo) / len(b)
                 for combo in itertools.product(*per)]
    diffs = np.array(diffs)
    return float(obs), float((np.abs(diffs) >= abs(obs) - 1e-12).mean())


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = running
    return out


def describe(xs) -> dict:
    x = np.asarray(xs, float)
    return {"values": x.round(4).tolist(), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT / "runs/sweep_uniform"))
    args = ap.parse_args(argv)
    root = Path(args.root)

    runs = {}
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(uniform|control)_(real|rewire)_s(\d+)", d.name)
        if m and (d / "pause_eval.json").exists():
            runs[d.name] = {"arm": m[1], "mask": m[2], "seed": int(m[3]), **run_values(d, root)}

    def cell(arm, mask, key):
        return [r[key] for r in runs.values() if r["arm"] == arm and r["mask"] == mask and key in r]

    def rows(arm, key):
        return [(r[key], r["mask"]) for r in runs.values() if r["arm"] == arm and key in r]

    report = {"runs": runs, "cells": {}}
    keys = ("drift_2s", "field_rms", "bump_share", "heading_r2", "spliced_r2", "success")
    for arm, mask in itertools.product(("uniform", "control"), ("real", "rewire")):
        report["cells"][f"{arm}_{mask}"] = {k: describe(cell(arm, mask, k)) for k in keys if cell(arm, mask, k)}

    u, c = rows("uniform", "field_rms"), rows("control", "field_rms")
    h1 = {"n_uniform": len(u), "n_control": len(c)}
    if u and c:
        (x_u, m_u), (x_c, m_c) = zip(*u, strict=True), zip(*c, strict=True)
        h1["field_rms_diff"], h1["p"] = perm_p(x_u, x_c, m_u, m_c)
        for arm in ("uniform", "control"):
            sel = [r for r in runs.values() if r["arm"] == arm]
            h1[f"{arm}_runs_stable_near_85"] = sum(r["stable_near_85"] for r in sel)
            h1[f"{arm}_runs_stable_near_0"] = sum(r["stable_near_0"] for r in sel)
        h1["supported"] = bool(h1["p"] < 0.05 and h1["field_rms_diff"] < 0
                               and h1["uniform_runs_stable_near_85"] < len(u) / 2)
    report["H1"] = h1

    h2 = {}
    for key, better in (("drift_2s", -1), ("field_rms", -1), ("bump_share", 1)):
        real, rew = cell("uniform", "real", key), cell("uniform", "rewire", key)
        if real and rew:
            diff, p = perm_p(real, rew)
            h2[key] = {"real_minus_rewire": diff, "p": p, "real_better": bool(np.sign(diff) == better)}
    if h2:
        adj = holm({k: v["p"] for k, v in h2.items()})
        for k in h2:
            h2[k]["p_holm"] = adj[k]
        h2["supported"] = bool(any(v["p_holm"] < 0.05 and v["real_better"] for v in h2.values())
                               and not any(v["p_holm"] < 0.05 and not v["real_better"] for v in h2.values()))
    report["H2"] = h2

    real, rew = cell("uniform", "real", "success"), cell("uniform", "rewire", "success")
    h3 = {"n_real": len(real), "n_rewire": len(rew)}
    if real and rew:
        h3["real_minus_rewire"], h3["p"] = perm_p(real, rew)
        h3["replicates"] = bool(h3["p"] < 0.05 and h3["real_minus_rewire"] < 0)
    report["H3"] = h3

    (root / "stats.json").write_text(json.dumps(report, indent=2))
    print(f"{'run':24s} {'drift2s':>8s} {'rms':>6s} {'share':>6s} {'R2':>6s} {'succ':>6s}  stable")
    for name, r in runs.items():
        print(f"{name:24s} {r['drift_2s']:8.2f} {r['field_rms']:6.2f} {r['bump_share']:6.3f} {r['heading_r2']:6.3f} "
              f"{r.get('success', float('nan')):6.3f}  {r['stable_deg']}")
    for h in ("H1", "H2", "H3"):
        print(h, json.dumps(report[h]))
    print(f"wrote {root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
