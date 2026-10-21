#!/usr/bin/env python
"""Effective trainable parameter counts, by component.

``W`` counts only its on-mask entries (the nonzeros of ``W * M``); everything else counts all
trainable elements. Reports the ratio of connectome parameters (on-mask W) to LPTC parameters for
the configured LPTC mode and for all three modes.

Usage: python scripts/count_params.py [key=value ...]   (e.g. connectome.mask_type=rewire)
"""

from __future__ import annotations

import sys

import numpy as np

from flybywire.config import load_config
from flybywire.connectome.masks import build_connectome
from flybywire.core.model import build_model
from flybywire.vision.lptc import MODES, LPTCPool


def count(model) -> dict[str, int]:
    core = model.core
    rows = {
        "core.W (on-mask, = nnz of M)": int(core.M.sum()),
        "core.a + core.b (per-neuron)": core.a.numel() + core.b.numel(),
    }
    if core.input_dependent_alpha:
        rows["core.c (input-dependent alpha)"] = core.c.numel()
    rows["input projection (ER only)"] = sum(
        p.numel() for p in model.inproj.parameters()
    )
    rows["heading decoder (EPG)"] = sum(p.numel() for p in model.heading.parameters())
    rows["actor (PFL3) incl. log_std"] = sum(
        p.numel() for p in model.actor.parameters()
    )
    rows["critic MLP (PPO only)"] = sum(p.numel() for p in model.critic.parameters())
    rows["LPTC"] = model.lptc.n_trainable() if model.lptc is not None else 0
    return rows


def main(argv=None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    conn = build_connectome(cfg)
    k, d, p = len(cfg.lptc.channels), len(cfg.vision.cell_types), 721
    dummy = np.zeros((k, d, p), dtype=np.float32)
    model = build_model(
        cfg, conn=conn, lptc=LPTCPool(dummy, mode=cfg.lptc.mode), verbose=False
    )
    rows = count(model)
    w = max(len(r) for r in rows)
    print(
        f"\nsource={cfg.connectome.source} mask={cfg.connectome.mask_type} N={conn.n} "
        f"edges={conn.n_edges} lptc.mode={cfg.lptc.mode}\n"
    )
    for name, n in rows.items():
        print(f"  {name:<{w}}  {n:>10,d}")
    total = sum(rows.values())
    bc_total = total - rows["critic MLP (PPO only)"]
    print(f"  {'-' * w}  {'-' * 10}")
    print(f"  {'total (all heads)':<{w}}  {total:>10,d}")
    print(f"  {'total used by BC (no critic)':<{w}}  {bc_total:>10,d}")
    conn_params = rows["core.W (on-mask, = nnz of M)"]
    print("\nconnectome (on-mask W) : LPTC parameter ratio")
    for mode in MODES:
        n_lptc = LPTCPool(dummy, mode=mode).n_trainable()
        ratio = f"{conn_params / n_lptc:.2f}" if n_lptc else "inf (LPTC frozen)"
        mark = "  <- configured" if mode == cfg.lptc.mode else ""
        print(f"  {mode:<7} LPTC={n_lptc:>7,d}  ratio={ratio}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
