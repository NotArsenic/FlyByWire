#!/usr/bin/env python
"""Can the expert's dodge around the next pillar be read from what the network sees?

Decisions of cached expert episodes where the nearest obstacle row is 1.0-2.5 m ahead and the drone
faces within 30 deg of the corridor axis. Targets: the dodge side (sign of the expert's world-frame
lateral velocity command, where that is above 0.1 m/s) and the normalised v_lateral and yaw-rate
commands. Feature sets:

  lptc8     the checkpoint's own 8 LPTC channels (6 self-motion + loom_left / loom_right)
  sectors6  the 6 self-motion channels + forward-translation loom in 6 azimuth sectors (edges
            -36, -18, 0, 18, 36 deg, so the central 18 deg on each side are separate), frozen gains
  sectors8  the same with 8 equal sectors across the 110 deg field
  t4t5      the full activity at that frame, 8 x 721
  geometry  privileged: offsets of the row's pillars, their half widths, the drone's y and yaw

and ``+hist``: the channel values at lags 0, 30, ..., 150 ms. Each with a linear model (ridge /
L2 logistic) and a 2-layer MLP, fitted on 75% of the episodes and scored on the rest. "dead ahead":
a pillar of the row within 0.35 m of the drone's y.

Usage:
  python scripts/probe_pillar_dodge.py --ckpt runs/curriculum/dense_s5/model.pt \\
      --sets default=runs/probe/cache_default forest=runs/probe/cache_forest [--out runs/probe/report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybywire.clocks import clocks_from_cfg
from flybywire.config import load_config
from flybywire.data.cache import T4T5Cache
from flybywire.envs.corridor import CorridorEnv, denormalize_action
from flybywire.envs.expert import ROW_TOL, surface_distance
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.vision.directions import baselines, direction_vectors, load_direction_map
from flybywire.vision.flowfields import (
    channel_flow,
    hexal_normalized_coords,
    normalized_to_focal,
)
from flybywire.vision.lptc import LPTCPool, build_templates

EGO = ("yaw", "pitch", "roll", "forward", "lateral", "vertical")
LAGS = (0, 3, 6, 9, 12, 15)  # sensory frames, 10 ms each
AHEAD = (1.0, 2.5)
DEAD_AHEAD = 0.35
MIN_VY = 0.1


def sector_pool(cfg, edges) -> LPTCPool:
    cell_types = tuple(cfg.vision.cell_types)
    dm = load_direction_map(cfg.lptc.get("direction_map"), cell_types=cell_types)
    xy = normalized_to_focal(
        hexal_normalized_coords(
            int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size)
        ),
        float(cfg.env.camera_fov_deg),
    )
    tilt = float(cfg.env.camera_tilt_deg)
    az = np.degrees(np.arctan(xy[:, 0]))
    forward = channel_flow(xy, "forward", tilt)
    flows = [channel_flow(xy, c, tilt) for c in EGO]
    flows += [
        forward * ((az >= lo) & (az < hi))[:, None]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return LPTCPool(
        build_templates(np.stack(flows), direction_vectors(dm, cell_types)),
        baseline=baselines(dm, cell_types),
        mode="frozen",
    )


def level_cfg(level: str):
    base = load_config()
    return (
        base
        if level == "default"
        else load_config(overrides=list(base.curriculum[level]))
    )


def episode_rows(
    level: str, cache: T4T5Cache, pools: dict, limits, decision_steps, device
) -> dict:
    """Features and targets of one level's selected decisions."""
    env = CorridorEnv(level_cfg(level))
    out = {
        k: []
        for k in (
            "ep",
            "side",
            "vy",
            "vlat",
            "yaw_rate",
            "dead",
            "geometry",
            "t4t5",
            "inside",
            *pools,
            *(f"{k}+hist" for k in pools),
        )
    }
    for i, path in enumerate(cache.trajectories):
        z = np.load(path)
        seed = json.loads(str(z["meta"]))["seed"]
        obstacles = np.asarray(env.reset(seed=seed)[1]["obstacles"], dtype=float)
        actions, pos, yaw, rel_yaw = z["actions"], z["pos"], z["yaw"], z["rel_yaw"]
        raw = np.asarray(cache.episode(i))
        t4t5 = torch.from_numpy(raw).to(device).float()
        with torch.no_grad():
            channels = {k: pool(t4t5).cpu().numpy() for k, pool in pools.items()}
        steps = decision_steps(len(actions))
        for k, t in enumerate(steps):
            x, y = pos[t, :2]
            ahead = obstacles[obstacles[:, 1] > x]
            if not len(ahead) or abs(np.degrees(rel_yaw[t])) > 30:
                continue
            ox = ahead[:, 1].min()
            if not AHEAD[0] <= ox - x <= AHEAD[1]:
                continue
            row = obstacles[np.abs(obstacles[:, 1] - ox) < ROW_TOL]
            row = row[np.argsort(row[:, 2])]
            vf, vl = denormalize_action(actions[k], limits)[:2]
            vy = np.sin(yaw[t]) * vf + np.cos(yaw[t]) * vl
            pillars = np.zeros(6)
            for j, ob in enumerate(row[:2]):
                pillars[3 * j : 3 * j + 3] = ob[2] - y, ob[4], 1.0
            lagged = np.maximum(t - np.asarray(LAGS), 0)
            out["ep"].append(i)
            out["side"].append(float(vy > 0))
            out["vy"].append(vy)
            out["vlat"].append(actions[k, 1])
            out["yaw_rate"].append(actions[k, 3])
            out["dead"].append(bool(np.any(np.abs(row[:, 2] - y) <= DEAD_AHEAD)))
            out["geometry"].append(np.r_[ox - x, y, rel_yaw[t], pillars])
            out["t4t5"].append(raw[t].astype(np.float16).ravel())
            out["inside"].append(
                min(surface_distance(x, y, ob)[0] for ob in obstacles) < 0
            )
            for name, c in channels.items():
                out[name].append(c[t])
                out[f"{name}+hist"].append(c[lagged].ravel())
    env.close()
    return {k: np.asarray(v) for k, v in out.items()}


def standardise(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd[sd < 1e-6] = (
        1.0  # T4/T5 cells that never leave baseline in the training episodes
    )
    return [(a - mu) / sd for a in (train, *others)]


def fit_ridge(xtr, ytr, xva, yva):
    best = None
    xtr1 = torch.cat([xtr, torch.ones(len(xtr), 1, device=xtr.device)], 1)
    xva1 = torch.cat([xva, torch.ones(len(xva), 1, device=xva.device)], 1)
    gram, rhs = xtr1.T @ xtr1, xtr1.T @ ytr
    for lam in (1e-2, 1e-1, 1, 10, 100, 1e3, 1e4):
        w = torch.linalg.solve(
            gram + lam * torch.eye(len(gram), device=gram.device), rhs
        )
        err = ((xva1 @ w - yva) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, w)
    w = best[1]
    return lambda x: torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1) @ w


def fit_net(xtr, ytr, xva, yva, hidden: int | None, classify: bool):
    """Linear (hidden=None, L2 picked on validation) or 2-layer MLP, early-stopped on validation."""
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()
    best = None
    decays = (1e-4,) if hidden else (1e-5, 1e-4, 1e-3, 1e-2)
    for wd in decays:
        torch.manual_seed(0)
        d, o = xtr.shape[1], ytr.shape[1]
        net = (
            nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, o),
            )
            if hidden
            else nn.Linear(d, o)
        ).to(xtr.device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        state, best_val, wait = None, np.inf, 0
        for _ in range(200):
            net.train()
            for idx in torch.randperm(len(xtr), device=xtr.device).split(1024):
                opt.zero_grad()
                loss_fn(net(xtr[idx]), ytr[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(xva), yva))
            if val < best_val - 1e-5:
                best_val, wait = val, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait == 8:
                    break
        net.load_state_dict(state)
        if best is None or best_val < best[0]:
            best = (best_val, net)
    net = best[1].eval()
    return lambda x: net(x).detach()


def r2(pred, y) -> float:
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(data: dict, features: str, model: str, device) -> dict:
    ep = data["ep_global"]
    test_np = ep % 4 == 0
    val_np = ~test_np & (ep % 5 == 1)
    train_np = ~test_np & ~val_np
    train, val, test = (
        torch.from_numpy(m).to(device) for m in (train_np, val_np, test_np)
    )
    x = torch.from_numpy(np.asarray(data[features], dtype=np.float32)).to(device)
    xtr, xva, xte = standardise(x[train], x[val], x[test])
    hidden = 256 if model == "mlp" else None
    report = {}

    has_side = np.abs(data["vy"]) > MIN_VY
    side = torch.from_numpy(data["side"][:, None].astype(np.float32)).to(device)
    m_tr, m_va, m_te = (
        torch.from_numpy(has_side[s]).to(device) for s in (train_np, val_np, test_np)
    )
    f = fit_net(
        xtr[m_tr], side[train][m_tr], xva[m_va], side[val][m_va], hidden, classify=True
    )
    with torch.no_grad():
        correct = ((f(xte[m_te]) > 0) == (side[test][m_te] > 0.5)).cpu().numpy()[:, 0]
    sel = has_side[test_np]
    dead, level = data["dead"][test_np][sel], data["level"][test_np][sel]
    report["side_acc"] = float(correct.mean())
    report["side_acc_dead"] = float(correct[dead].mean())
    for lv in np.unique(data["level"]):
        report[f"side_acc_{lv}"] = float(correct[level == lv].mean())
        report[f"side_acc_dead_{lv}"] = float(correct[dead & (level == lv)].mean())

    y = torch.from_numpy(
        np.stack([data["vlat"], data["yaw_rate"]], 1).astype(np.float32)
    ).to(device)
    if model == "linear":
        f = fit_ridge(xtr, y[train], xva, y[val])
    else:
        f = fit_net(xtr, y[train], xva, y[val], hidden, classify=False)
    with torch.no_grad():
        pred = f(xte).cpu().numpy()
    yte = y[test].cpu().numpy()
    dead = data["dead"][test_np]
    for j, name in enumerate(("vlat", "yaw_rate")):
        report[f"r2_{name}"] = r2(pred[:, j], yte[:, j])
        report[f"r2_{name}_dead"] = r2(pred[dead, j], yte[dead, j])
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default="runs/curriculum/dense_s5/model.pt")
    ap.add_argument("--sets", nargs="+", required=True, help="level=cache_dir")
    ap.add_argument("--out", default="runs/probe/report.json")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.ckpt, device)
    half = float(cfg.env.camera_fov_deg) / 2
    pools = {
        "lptc8": model.lptc.eval(),
        "sectors6": sector_pool(cfg, [-90, -36, -18, 0, 18, 36, 90]).to(device),
        "sectors8": sector_pool(cfg, [-90, *np.linspace(-half, half, 9)[1:-1], 90]).to(
            device
        ),
    }
    clocks = clocks_from_cfg(cfg)
    parts = []
    for item in args.sets:
        level, cache_dir = item.split("=", 1)
        d = episode_rows(
            level,
            T4T5Cache(cache_dir),
            pools,
            cfg.action.limits,
            clocks.decision_core_steps,
            device,
        )
        d["level"] = np.full(len(d["ep"]), level)
        d["ep_global"] = d["ep"] + 100000 * len(parts)
        print(
            f"{level}: {len(d['ep'])} decisions from {len(np.unique(d['ep']))} episodes, "
            f"{d['dead'].mean():.1%} dead ahead, {d['inside'].mean():.2%} inside an obstacle",
            flush=True,
        )
        parts.append(d)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    sets = [
        "geometry",
        "lptc8",
        "lptc8+hist",
        "sectors6",
        "sectors6+hist",
        "sectors8",
        "sectors8+hist",
        "t4t5",
    ]
    report = {
        "n": int(len(data["ep"])),
        "dead_fraction": float(data["dead"].mean()),
        "side_majority": float(max(data["side"].mean(), 1 - data["side"].mean())),
        "probes": {},
    }
    print(
        f"{'features':16s} {'model':6s} {'side':>6s} {'dead':>6s} {'r2 vlat':>8s} {'dead':>6s} "
        f"{'r2 yaw':>7s} {'dead':>6s}"
    )
    for features in sets:
        for kind in ("linear", "mlp"):
            r = probe(data, features, kind, device)
            report["probes"][f"{features}/{kind}"] = r
            print(
                f"{features:16s} {kind:6s} {r['side_acc']:6.3f} {r['side_acc_dead']:6.3f} {r['r2_vlat']:8.3f} "
                f"{r['r2_vlat_dead']:6.3f} {r['r2_yaw_rate']:7.3f} {r['r2_yaw_rate_dead']:6.3f}",
                flush=True,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
