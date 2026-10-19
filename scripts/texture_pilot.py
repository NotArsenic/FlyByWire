#!/usr/bin/env python
"""Does a texture make the obstacles easier to read from T4/T5? A Gate-2-style ridge on pilot data.

Physics does not depend on rendering, so the same seeds under different textures give identical
trajectories and labels; only the frames differ. For each cache (one per texture, built from the
same seeds with collect_expert.py and build_cache.py) this fits one ridge from the T4/T5 voltages
at each decision frame (8 x 721, standardised) to

  yaw_rate                       body yaw rate at that frame, rad/s
  a_forward, a_lateral, a_yaw    the expert's normalised action for the decision
  surface                        distance to the nearest obstacle surface, clipped at 2 m
  y, wall_gap                    lateral position, and distance to the nearer side wall (m)
  sin_yaw, cos_yaw               heading relative to the corridor axis (an absolute cue, not integrated)

on held-out episodes (70/10/20 split by episode, penalty picked per target on validation, the same
split for every cache). The expert's forward and lateral commands depend on where the obstacles
are, so R^2 on those says how much of that the network can get from the optic lobe. Intervals are
bootstrap percentiles over test episodes. --probe mlp swaps the ridge for one hidden layer (512 units),
for textures that only a nonlinear readout can use.

Usage:
  python scripts/texture_pilot.py NAME=CACHE_DIR [NAME=CACHE_DIR ...] [--out runs/texture_pilot] [--probe ridge|mlp]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.utils import seeding

from flybywire.clocks import clocks_from_cfg
from flybywire.config import load_config
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.envs.corridor import CorridorEnv
from flybywire.envs.expert import surface_distance

TARGETS = ("yaw_rate", "a_forward", "a_lateral", "a_yaw", "surface", "y", "wall_gap", "sin_yaw", "cos_yaw")
ALPHAS = np.logspace(-3, 3, 13)
SURFACE_CLIP_M = 2.0
WASHOUT_DECISIONS = 13  # flyvis starts from grey; Gate 2 drops 50 frames


def layout(seed: int, cfg) -> np.ndarray:
    """The obstacles ``CorridorEnv.reset(seed=seed)`` draws."""
    stub = SimpleNamespace(np_random=seeding.np_random(seed)[0], n_obstacles=int(cfg.env.n_obstacles),
                           length=float(cfg.env.corridor.length_m), spacing=None)
    return CorridorEnv._sample_layout(stub)


def episode_targets(tr: Trajectory, cfg, spd: int) -> np.ndarray:
    t = np.arange(tr.n_decisions) * spd + spd - 1
    obstacles = layout(int(tr.meta["seed"]), cfg)
    half_width = float(cfg.env.corridor.width_m) / 2
    surface = [min(surface_distance(p[0], p[1], ob)[0] for ob in obstacles) for p in tr.state["pos"][t]]
    y = tr.state["pos"][t, 1]
    return np.column_stack([tr.yaw_rate[t], tr.actions[:, 0], tr.actions[:, 1], tr.actions[:, 3],
                            np.clip(surface, 0.0, SURFACE_CLIP_M), y, half_width - np.abs(y),
                            np.sin(tr.rel_yaw[t]), np.cos(tr.rel_yaw[t])])[WASHOUT_DECISIONS:]


def split(n: int) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(0).permutation(n)
    n_test, n_val = round(0.2 * n), round(0.1 * n)
    return {"test": perm[:n_test], "val": perm[n_test : n_test + n_val], "train": perm[n_test + n_val :]}


def load(cache_dir: str, cfg) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache = T4T5Cache(cache_dir)
    spd = clocks_from_cfg(cfg).sensory_per_decision
    xs, ys = [], []
    for i, path in enumerate(cache.trajectories):
        tr = Trajectory.load(path, load_hexals=False)
        x = np.asarray(cache.episode(i)[spd - 1 :: spd], dtype=np.float32)
        xs.append(x.reshape(len(x), -1)[WASHOUT_DECISIONS:])
        ys.append(episode_targets(tr, cfg, spd).astype(np.float32))
    return xs, ys


def r2_by_target(pred: np.ndarray, y: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return 1 - ((pred - y) ** 2).sum(0) / ((y - mean) ** 2).sum(0)


def ridge_predict(xs, ys, parts, device: str) -> tuple[np.ndarray, dict]:
    def cat(idx, a):
        return np.concatenate([a[i] for i in idx])

    xt = torch.from_numpy(cat(parts["train"], xs)).to(device)
    yt = torch.from_numpy(cat(parts["train"], ys)).to(device)
    mu, sd = xt.mean(0), xt.std(0).clamp_min(1e-6)
    zt, ybar = (xt - mu) / sd, yt.mean(0)
    evals, evecs = torch.linalg.eigh((zt.T @ zt).double() / len(zt))
    proj = evecs.T @ ((zt.T @ (yt - ybar)).double() / len(zt))
    del zt, xt

    def z(idx):
        return (torch.from_numpy(cat(idx, xs)).to(device) - mu) / sd

    def weights(a):
        return (evecs @ (proj / (evals + a)[:, None])).float()

    zv, yv = z(parts["val"]), cat(parts["val"], ys)
    val = np.stack([r2_by_target((zv @ weights(a) + ybar).cpu().numpy(), yv, yv.mean(0)) for a in ALPHAS])
    best = ALPHAS[val.argmax(0)]
    zs = z(parts["test"])
    pred = np.stack([(zs @ weights(a) + ybar).cpu().numpy()[:, j] for j, a in enumerate(best)], axis=1)
    return pred, {"alpha": dict(zip(TARGETS, best.tolist(), strict=True))}


def mlp_predict(xs, ys, parts, device: str, epochs: int = 30, hidden: int = 512) -> tuple[np.ndarray, dict]:
    """One hidden layer, all targets at once; the epoch with the best mean validation R^2 is kept."""
    def cat(idx, a):
        return torch.from_numpy(np.concatenate([a[i] for i in idx])).to(device)

    x = {k: cat(parts[k], xs) for k in parts}
    y = {k: cat(parts[k], ys) for k in parts}
    mu, sd = x["train"].mean(0), x["train"].std(0).clamp_min(1e-6)
    ymu, ysd = y["train"].mean(0), y["train"].std(0)
    z = {k: (v - mu) / sd for k, v in x.items()}
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(x["train"].shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(hidden, y["train"].shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, n = -np.inf, None, len(z["train"])
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(n, device=device).split(256):
            loss = ((net(z["train"][idx]) - (y["train"][idx] - ymu) / ysd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = r2_by_target((net(z["val"]) * ysd + ymu).cpu().numpy(), y["val"].cpu().numpy(),
                               y["val"].mean(0).cpu().numpy()).mean()
        if val > best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        return (net(z["test"]) * ysd + ymu).cpu().numpy(), {"val_r2_mean": float(best)}


def run(xs, ys, device: str, n_boot: int = 500, probe: str = "ridge") -> dict:
    parts = split(len(xs))
    pred, extra = (ridge_predict if probe == "ridge" else mlp_predict)(xs, ys, parts, device)
    y = np.concatenate([ys[i] for i in parts["test"]])
    lengths = [len(ys[i]) for i in parts["test"]]
    owner = np.repeat(np.arange(len(lengths)), lengths)
    mean = y.mean(0)
    sse = np.stack([((pred - y) ** 2)[owner == e].sum(0) for e in range(len(lengths))])
    sst = np.stack([((y - mean) ** 2)[owner == e].sum(0) for e in range(len(lengths))])
    rng = np.random.default_rng(0)
    boot = np.stack([1 - sse[b].sum(0) / sst[b].sum(0)
                     for b in (rng.integers(0, len(lengths), len(lengths)) for _ in range(n_boot))])
    return {
        "n_episodes": {k: len(v) for k, v in parts.items()},
        **extra,
        "test_r2": dict(zip(TARGETS, (1 - sse.sum(0) / sst.sum(0)).tolist(), strict=True)),
        "test_r2_ci95": {t: np.percentile(boot[:, j], [2.5, 97.5]).tolist() for j, t in enumerate(TARGETS)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", help="NAME=CACHE_DIR")
    ap.add_argument("--out", default="runs/texture_pilot")
    ap.add_argument("--probe", choices=("ridge", "mlp"), default="ridge")
    args = ap.parse_args(argv)
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for item in args.caches:
        name, cache_dir = item.split("=", 1)
        report[name] = run(*load(cache_dir, cfg), device, probe=args.probe)
        r2 = report[name]["test_r2"]
        print(f"{name:16s} " + "  ".join(f"{t} {r2[t]:.3f}" for t in TARGETS))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
