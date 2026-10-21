#!/usr/bin/env python
"""Gate 2: is yaw rate linearly decodable from T4/T5 activity in the corridor?

Pass criterion: ridge regression from T4/T5 to yaw rate on held-out trajectories, R^2 > 0.5.

Features are the cached T4/T5 voltages at frame t (8 x 721), standardised on the training set;
the target is body yaw rate at frame t. Trajectories are split 70/10/20 into train, validation
(picks the penalty) and test; the model is refit on train + validation and scored once on test.
The first --washout frames of each episode are dropped because flyvis starts from the grey steady
state. A version pooled over a coarse grid of hexals is reported as well, but the gate uses the
full one. Only X'X, X'y and friends are accumulated, so the cache is read once.

If it fails, try env.wall_texture=checker or noise first; switching to vision.backend=farneback
is a separate decision. Writes runs/gate2/report.json.

Usage:
  python scripts/gate2_decode_yaw_rate.py [--cache DIR] [--out DIR] [--washout 50] [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybywire.config import data_dir, load_config, runs_dir
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.vision.flowfields import hexal_normalized_coords

THRESHOLD = 0.5
ALPHAS = np.logspace(-4, 4, 17)  # per-sample penalty on standardised features
SPLIT = (0.7, 0.1, 0.2)


class Stats:
    """Sums for least squares with an intercept, kept in float64."""

    def __init__(self, dim: int, device):
        self.n = 0
        self.sx = torch.zeros(dim, dtype=torch.float64, device=device)
        self.sy = 0.0
        self.syy = 0.0
        self.xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.xy = torch.zeros(dim, dtype=torch.float64, device=device)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.n += x.shape[0]
        self.sx += x.sum(0, dtype=torch.float64)
        self.sy += float(y.sum(dtype=torch.float64))
        self.syy += float((y.double() ** 2).sum())
        self.xx += (x.T @ x).double()
        self.xy += (x.T @ y).double()

    def __add__(self, other: Stats) -> Stats:
        s = Stats(0, self.sx.device)
        s.n, s.sy, s.syy = self.n + other.n, self.sy + other.sy, self.syy + other.syy
        s.sx, s.xx, s.xy = self.sx + other.sx, self.xx + other.xx, self.xy + other.xy
        return s


class Ridge:
    """Ridge on standardised features; one eigendecomposition serves every penalty."""

    def __init__(self, st: Stats):
        n = st.n
        self.mu = (st.sx / n).cpu().numpy()
        self.ybar = st.sy / n
        cov = (st.xx / n).cpu().numpy() - np.outer(self.mu, self.mu)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        self.sd = np.where(sd > 1e-8, sd, 1.0)
        zz = cov / np.outer(self.sd, self.sd)
        zy = ((st.xy.cpu().numpy() / n) - self.mu * self.ybar) / self.sd
        self.evals, self.evecs = np.linalg.eigh(zz)
        self.proj = self.evecs.T @ zy

    def coef(self, alpha: float) -> tuple[np.ndarray, float]:
        w = self.evecs @ (self.proj / (self.evals + alpha))
        beta = w / self.sd
        return beta, self.ybar - float(self.mu @ beta)


def r2(st: Stats, beta: np.ndarray, b: float) -> float:
    """R^2 of x @ beta + b on the rows summarised in st."""
    xx, xy, sx = (t.cpu().numpy() for t in (st.xx, st.xy, st.sx))
    sse = (
        st.syy
        - 2 * (beta @ xy + b * st.sy)
        + beta @ xx @ beta
        + 2 * b * (beta @ sx)
        + st.n * b * b
    )
    sst = st.syy - st.sy**2 / st.n
    return float(1 - sse / sst)


def pooling_matrix(
    n_hexals: int, extent: int, kernel_size: int, grid: int
) -> np.ndarray:
    xy = hexal_normalized_coords(extent, kernel_size)
    bins = np.clip(((xy + 1) / 2 * grid).astype(int), 0, grid - 1)
    cell = bins[:, 1] * grid + bins[:, 0]
    used = np.unique(cell)
    pool = np.zeros((n_hexals, len(used)), np.float32)
    for j, c in enumerate(used):
        m = cell == c
        pool[m, j] = 1.0 / m.sum()
    return pool


def split(n: int, seed: int) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, round(SPLIT[2] * n))
    n_val = max(1, round(SPLIT[1] * n))
    return {
        "test": perm[:n_test],
        "val": perm[n_test : n_test + n_val],
        "train": perm[n_test + n_val :],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--washout",
        type=int,
        default=50,
        help="frames dropped at the start of each episode",
    )
    ap.add_argument(
        "--grid",
        type=int,
        default=4,
        help="pooled variant: grid x grid bins per subtype",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(overrides=args.overrides)
    cache = T4T5Cache(args.cache or data_dir(cfg) / "cache")
    out = Path(args.out) if args.out else runs_dir(cfg) / "gate2"
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_types, n_hexals = cache.data.shape[1:]
    pool = torch.from_numpy(
        pooling_matrix(
            n_hexals,
            int(cfg.vision.boxeye.extent),
            int(cfg.vision.boxeye.kernel_size),
            args.grid,
        )
    ).to(device)

    sets = split(len(cache), args.seed)
    which = {int(i): name for name, idx in sets.items() for i in idx}
    full = {k: Stats(n_types * n_hexals, device) for k in sets}
    pooled = {k: Stats(n_types * pool.shape[1], device) for k in sets}
    shift = None
    yaw_rates, metas = [], []
    t0 = time.time()
    for i, path in enumerate(cache.trajectories):
        tr = Trajectory.load(path, load_hexals=False)
        metas.append(tr.meta)
        y = torch.from_numpy(tr.yaw_rate[args.washout :]).to(device)
        x = (
            torch.from_numpy(np.array(cache.episode(i)[args.washout :]))
            .to(device)
            .float()
        )
        if (
            shift is None
        ):  # any constant shift works; it just keeps the float32 sums accurate
            shift = x.mean(0)
        x = x - shift
        full[which[i]].add(x.reshape(len(x), -1), y)
        pooled[which[i]].add((x @ pool).reshape(len(x), -1), y)
        yaw_rates.append(tr.yaw_rate[args.washout :])
    read_s = time.time() - t0

    def fit(stats):
        val_r2 = {}
        model = Ridge(stats["train"])
        for a in ALPHAS:
            val_r2[float(a)] = r2(stats["val"], *model.coef(a))
        best = max(val_r2, key=val_r2.get)
        final = Ridge(stats["train"] + stats["val"])
        beta, b = final.coef(best)
        return {
            "alpha": best,
            "val_r2": val_r2,
            "test_r2": r2(stats["test"], beta, b),
            "train_val_r2": r2(stats["train"] + stats["val"], beta, b),
        }

    res_full = fit(full)
    res_pooled = fit(pooled)
    yr = np.concatenate(yaw_rates)
    passed = res_full["test_r2"] > THRESHOLD
    report = {
        "gate": 2,
        "criterion": f"ridge T4/T5 -> yaw rate on held-out trajectories, test R^2 > {THRESHOLD}",
        "passed": bool(passed),
        "test_r2": res_full["test_r2"],
        "full": {"n_features": n_types * n_hexals, **res_full},
        "pooled": {
            "n_features": n_types * int(pool.shape[1]),
            "grid": args.grid,
            **res_pooled,
        },
        "n_trajectories": {k: int(len(v)) for k, v in sets.items()},
        "test_trajectories": [
            str(cache.trajectories[i].name) for i in sorted(sets["test"])
        ],
        "n_frames": {k: full[k].n for k in sets},
        "washout_frames": args.washout,
        "yaw_rate": {
            "mean": float(yr.mean()),
            "sd": float(yr.std()),
            "abs_max": float(np.abs(yr).max()),
        },
        "wall_texture": sorted({m.get("wall_texture", "?") for m in metas}),
        "cache": {
            k: cache.manifest.get(k)
            for k in ("flyvis_model", "dt", "config_hash", "total_T")
        },
        "seconds": {"read": round(read_s, 1), "total": round(time.time() - t0, 1)},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"Gate 2 {'PASS' if passed else 'FAIL'}: test R^2 = {res_full['test_r2']:.3f} "
        f"(full, alpha {res_full['alpha']:.0e}); pooled {args.grid}x{args.grid}: {res_pooled['test_r2']:.3f}; "
        f"train+val R^2 {res_full['train_val_r2']:.3f}"
    )
    print(f"report: {out / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
