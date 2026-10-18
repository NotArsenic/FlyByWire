#!/usr/bin/env python
"""Open-loop check of a BC checkpoint on cached episodes (no flyvis).

By default the episodes are the validation split train_bc held out of the checkpoint's own cache;
--split all scores every episode, for a cache the model never trained on. Reports heading R^2
(joint sin/cos) and angular error, the error by second of the episode, action MSE per dimension,
and heading R^2 of the decoded heading at t against the true heading at t - L. The training target
stays the heading at t; a best L above 0 means the decoder reads the heading late.

Usage:
  python scripts/eval_open_loop.py [--ckpt runs/bc/real_s0/model.pt] [--cache DIR] [--split val|all]
                                   [--max-lag 20] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import ACTION_KEYS, REPO_ROOT
from flybywire.data.cache import T4T5Cache
from flybywire.eval.closed_loop import checkpoint_cache, heading_error_deg, load_checkpoint
from flybywire.eval.metrics import angular_error_deg, heading_r2, lagged_heading_r2
from flybywire.eval.topography import val_split
from flybywire.train.bc import BCData


@torch.no_grad()
def run_episodes(model, data: BCData, idx, device, batch_episodes: int = 32):
    """Per episode: decoded (sin, cos), true (sin, cos), actor mean and expert action at decisions."""
    spd = data.clocks.sensory_per_decision
    pred_h, true_h, pred_a, true_a = [], [], [], []
    for start in range(0, len(idx), batch_episodes):
        sub = list(idx[start : start + batch_episodes])
        batch = data.batch(sub, device)
        out = model.run(batch["x16"].float(), model.init_state(len(sub)), t0=0)
        for j, i in enumerate(sub):
            n = data.cache.length(i)
            pred_h.append(out["heading"][j, :n].cpu().numpy())
            true_h.append(batch["heading"][j, :n].cpu().numpy())
            pred_a.append(out["action_mean"][j, : n // spd].cpu().numpy())
            true_a.append(batch["actions"][j, : n // spd].cpu().numpy())
    return pred_h, true_h, pred_a, true_a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "runs" / "bc" / "real_s0" / "model.pt"))
    ap.add_argument("--cache", default=None, help="default: the cache the checkpoint was trained on")
    ap.add_argument("--split", choices=["val", "all"], default="val")
    ap.add_argument("--max-lag", type=int, default=20, help="largest lag in core steps (10 ms each)")
    ap.add_argument("--out", default=None, help="default: open_loop_<checkpoint name>.json next to the checkpoint")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, cfg = load_checkpoint(ckpt, device)
    clocks = clocks_from_cfg(cfg)
    cache = T4T5Cache(args.cache or checkpoint_cache(ckpt, cfg))
    data = BCData(cache, clocks)
    idx = val_split(len(data), int(cfg.seed)) if args.split == "val" else np.arange(len(data))

    pred_h, true_h, pred_a, true_a = run_episodes(model, data, idx, device, int(cfg.bc.batch_episodes))
    ph, th = np.concatenate(pred_h), np.concatenate(true_h)
    pa, ta = np.concatenate(pred_a), np.concatenate(true_a)

    sec = clocks.sensory_hz
    err = [heading_error_deg(p, np.arctan2(t[:, 0], t[:, 1])) for p, t in zip(pred_h, true_h, strict=True)]
    by_second = [float(np.concatenate([e[s * sec : (s + 1) * sec] for e in err]).mean())
                 for s in range(max(len(e) for e in err) // sec)]
    se = (pa - ta) ** 2
    lags = list(range(args.max_lag + 1))
    r2_lag = lagged_heading_r2(pred_h, true_h, lags)
    best = int(np.argmax(r2_lag))

    report = {
        "checkpoint": str(ckpt),
        "cache": str(cache.dir),
        "split": args.split,
        "n_episodes": len(idx),
        "heading_r2": heading_r2(ph, th),
        "heading_err_deg": angular_error_deg(ph, th),
        "heading_err_deg_by_second": by_second,
        "action_mse": float(se.mean()),
        "action_mse_per_dim": dict(zip(ACTION_KEYS, se.mean(0).round(5).tolist(), strict=True)),
        "action_var_per_dim": dict(zip(ACTION_KEYS, ta.var(0).round(5).tolist(), strict=True)),
        "lag": {"lags": lags, "heading_r2": r2_lag, "r2_at_0": r2_lag[0], "best_lag": lags[best],
                "r2_at_best": r2_lag[best]},
    }
    out = Path(args.out) if args.out else ckpt.parent / f"open_loop_{ckpt.stem}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{len(idx)} episodes ({args.split}) from {cache.dir}")
    print(f"heading R2 {report['heading_r2']:.3f}  err {report['heading_err_deg']:.2f} deg  "
          f"action MSE {report['action_mse']:.4f}  {report['action_mse_per_dim']}")
    print(f"R2 at lag 0 {r2_lag[0]:.3f}, best {r2_lag[best]:.3f} at lag {lags[best]} ({lags[best] * 10} ms)")
    print(f"error by second (deg): {np.round(by_second, 1).tolist()}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
