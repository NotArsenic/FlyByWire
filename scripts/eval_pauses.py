#!/usr/bin/env python
"""Heading persistence through a time-freeze pause on held-out episodes (D39).

The checkpoint runs on the real T4/T5 up to the start t0 of a pause-bank entry. From there the
input is the bank's: flyvis' response to the frame at t0 held still, entry transient included.
After the bank's rows the last row is held, so the pause can run longer than the bank (default
10 s) on the steady response. Drift is the change of the decoded heading since t0, signed (positive
= turning left), per entry; the bank's t0 are spread uniformly over the episodes, so they include
mid-maneuver starts. The sign test and bootstrap work on episodes, not entries.

Also scores the checkpoint on test episodes with pauses spliced in as in training (R^2 and
error over everything, and at the decisions inside the pauses against outside them).

Build the banks first, from the cache the episodes come from. --bank is the usual kind (entries far
enough apart to be spliced together); --drift-bank can be a dense one with a short recovery, many
more entries per episode, so the velocity field has samples at every heading. A single hold length
keeps the entries comparable.

  python scripts/build_pause_bank.py --cache data/cache_turns_test --out data/pause_bank_turns_test \\
      --hold 100 100
  python scripts/build_pause_bank.py --cache data/cache_turns_test --out data/pause_bank_turns_test_dense \\
      --hold 100 100 --per-episode 27 --recovery 8

Usage:
  python scripts/eval_pauses.py --ckpt runs/step4_pilot/real_pause_anatomical_s0/model_best.pt
                                [--cache data/cache_turns_test] [--bank data/pause_bank_turns_test]
                                [--drift-bank data/pause_bank_turns_test_dense]
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
from flybywire.data.pause_bank import PauseBank, PauseSampler
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.darkness import pause_branch, signed_stats, snapshots, velocity_field
from flybywire.eval.metrics import angular_error_deg, heading_r2
from flybywire.eval.topography import wrap
from flybywire.train.bc import BCData

SECONDS = (0.5, 1.0, 2.0, 5.0, 10.0)


def entry_inputs(model, data, bank, device, chunk):
    """Per bank entry: core state at t0, LPTC channels of its held-frame rows, true heading at t0."""
    v, r, s, true = [], [], [], []
    for start in range(0, len(data), chunk):
        idx = list(range(start, min(start + chunk, len(data))))
        es = np.concatenate([bank.entries(i) for i in idx])
        batch = data.batch(idx, device)
        snaps = snapshots(model, batch["x16"].float(), sorted({int(bank.t0[e]) + 1 for e in es}))
        for e in es:
            b = int(bank.episode[e]) - start
            sv, sr = snaps[int(bank.t0[e]) + 1]
            v.append(sv[b])
            r.append(sr[b])
            rows = torch.from_numpy(np.array(bank.segments(e)[0])).to(device).float()
            with torch.no_grad():
                s.append(model.sensory(rows))
            h = batch["heading"][b, int(bank.t0[e])]
            true.append(torch.atan2(h[0], h[1]).item())
        del batch, snaps
    length = max(len(x) for x in s)
    s = torch.stack([torch.cat([x, x[-1:].expand(length - len(x), -1)]) for x in s])
    return torch.stack(v), torch.stack(r), s, np.array(true)


def spliced_scores(model, cache, bank, clocks, device, per_episode, seed, chunk):
    """Heading R^2 and error on episodes with pauses spliced in as in training."""
    data = BCData(cache, clocks, PauseSampler(bank, per_episode, 1.0))
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in ("pred", "true", "mask", "pred_dec", "true_dec", "real_dec", "pause_dec")}
    for start in range(0, len(data), chunk):
        idx = list(range(start, min(start + chunk, len(data))))
        batch = data.batch(idx, device, rng)
        with torch.no_grad():
            out = model.run(batch["x16"].float(), model.init_state(len(idx)))
        dt = out["decision_t"]
        k = dt // clocks.sensory_per_decision
        valid = batch["mask"][:, dt]
        parts["pred"].append(out["heading"].reshape(-1, 2).cpu())
        parts["true"].append(batch["heading"].reshape(-1, 2).cpu())
        parts["mask"].append(batch["mask"].reshape(-1).cpu())
        parts["pred_dec"].append(out["heading"][:, dt].reshape(-1, 2).cpu())
        parts["true_dec"].append(batch["heading"][:, dt].reshape(-1, 2).cpu())
        parts["real_dec"].append((valid & batch["amask"][:, k]).reshape(-1).cpu())
        parts["pause_dec"].append((valid & ~batch["amask"][:, k]).reshape(-1).cpu())
    p = {k: torch.cat(v) for k, v in parts.items()}
    return {
        "heading_r2": heading_r2(p["pred"], p["true"], p["mask"]),
        "heading_err_deg": angular_error_deg(p["pred"], p["true"], p["mask"]),
        "err_at_pause_decisions_deg": angular_error_deg(p["pred_dec"], p["true_dec"], p["pause_dec"]),
        "err_at_other_decisions_deg": angular_error_deg(p["pred_dec"], p["true_dec"], p["real_dec"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "cache_turns_test"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "data" / "pause_bank_turns_test"))
    ap.add_argument("--drift-bank", default=None, help="entries for the drift curves (default: --bank)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--per-episode", type=int, default=2, help="pauses per episode in the spliced scores")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out", default=None, help="directory for pause_eval.json (default: the checkpoint's)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, cfg = load_checkpoint(ckpt, device)
    clocks = clocks_from_cfg(cfg)
    hz = clocks.sensory_hz
    cache, bank = T4T5Cache(args.cache), PauseBank(args.bank)
    drift_bank = PauseBank(args.drift_bank) if args.drift_bank else bank
    for b in (bank, drift_bank):
        b.check_cache(cache)
    data = BCData(cache, clocks)

    v, r, s, true = entry_inputs(model, data, drift_bank, device, args.chunk)
    n_steps = int(round(args.seconds * hz))
    sel = np.arange(0, n_steps + 1, 5)
    drift, start = [], []
    for a in range(0, len(v), 100):
        rs = pause_branch(model, v[a : a + 100], r[a : a + 100], s[a : a + 100], n_steps)
        with torch.no_grad():
            head = model.heading(rs[:, sel]).cpu().numpy()
        ang = np.arctan2(head[..., 0], head[..., 1])
        drift.append(np.degrees(wrap(ang - ang[:, :1])))
        start.append(ang[:, 0])
    drift, start = np.concatenate(drift), np.concatenate(start)
    episode = drift_bank.episode[np.concatenate([drift_bank.entries(i) for i in range(len(data))])]
    subsets = {"all": np.ones(len(drift), bool), "off_axis": np.abs(true) > np.deg2rad(45)}
    at = {f"{sec:g}s": int(round(sec * hz)) // 5 for sec in SECONDS if sec <= args.seconds}
    stats = {}
    for name, keep in subsets.items():
        stats[name] = {}
        for label, k in at.items():
            st = signed_stats(drift[keep, k], episode[keep])
            st["median_abs"] = float(np.median(np.abs(drift[keep, k])))
            stats[name][label] = st
    report = {
        "checkpoint": str(ckpt),
        "cache": str(cache.dir),
        "bank": str(bank.dir),
        "drift_bank": str(drift_bank.dir),
        "hold_steps": sorted(set(drift_bank.hold.tolist())),
        "n_entries": {k: int(m.sum()) for k, m in subsets.items()},
        "drift_deg": stats,
        "velocity_field_1s": velocity_field(drift[:, at["1s"]], start),
        "spliced_test": spliced_scores(model, cache, bank, clocks, device, args.per_episode, 0, args.chunk),
    }
    out_dir = Path(args.out) if args.out else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pause_eval.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / "pause_eval_drift.npz", t_s=sel / hz, drift_deg=drift.astype(np.float32),
                        start_decoded=start, start_true=true, episode=episode)

    for label in at:
        a = stats["all"][label]
        print(f"{label:>4s}  mean {a['mean']:6.1f} deg [{a['ci95_low']:.1f}, {a['ci95_high']:.1f}]  "
              f"mean |.| {a['mean_abs']:5.1f}  median |.| {a['median_abs']:5.1f}  "
              f"off-axis mean |.| {stats['off_axis'][label]['mean_abs']:5.1f}")
    vf = report["velocity_field_1s"]
    print(f"velocity field: stable {vf['stable_deg']}, unstable {vf['unstable_deg']}")
    print("spliced test:", {k: round(x, 3) for k, x in report["spliced_test"].items()})
    print(f"wrote {out_dir / 'pause_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
