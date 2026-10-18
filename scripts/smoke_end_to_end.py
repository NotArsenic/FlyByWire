#!/usr/bin/env python
"""End-to-end smoke run that checks the plumbing. Not a result; don't report its numbers.

Renders a synthetic textured scene drifting horizontally under a random-walk yaw rate (8 fake
trajectories of 300 core steps = 75 decisions, saved to data/smoke/expert/), builds the T4/T5
cache with frozen flyvis (data/smoke/cache/), then runs BC on the FlyWire connectome for 200 updates
with target action = yaw rate proportional to -rel_yaw. Asserts every loss is finite and the total
loss drops by >= 30%; prints the heading-decode R^2.

Uses data/flyvis/direction_map.json if it exists; otherwise measures a provisional map into
data/smoke/direction_map.json (same routine as scripts/calibrate_t4t5_directions.py).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cache as build_cache_script  # noqa: E402

from flybywire.clocks import clocks_from_cfg  # noqa: E402
from flybywire.config import data_dir, load_config, runs_dir  # noqa: E402
from flybywire.data.trajectories import Trajectory, relative_yaw, trajectory_path  # noqa: E402
from flybywire.train.bc import train_bc  # noqa: E402
from flybywire.vision.boxeye import BoxEye  # noqa: E402
from flybywire.vision.directions import default_direction_map_path, measure_directions  # noqa: E402
from flybywire.vision.flyvis_stepper import FlyvisStepper  # noqa: E402
from flybywire.vision.stimuli import panorama_texture, view_from_panorama  # noqa: E402

N_TRAJ, T_CORE, N_UPDATES, MIN_DROP = 8, 300, 200, 0.30


def make_trajectories(cfg, eye: BoxEye, out_dir: Path, seed: int = 0) -> list[Path]:
    clocks = clocks_from_cfg(cfg)
    size, fov = int(cfg.env.frame_size), float(cfg.env.camera_fov_deg)
    k = T_CORE // clocks.sensory_per_decision
    dec_t = clocks.decision_core_steps(k)
    paths = []
    for i in range(N_TRAJ):
        rng = np.random.default_rng(seed + i)
        pano = panorama_texture(size, fov, seed=seed + 100 + i)
        omega, yaw = 0.0, 0.0
        yaws, rates = np.zeros(T_CORE), np.zeros(T_CORE)
        for t in range(T_CORE):
            omega = float(np.clip(0.98 * omega + rng.normal(0.0, 0.11), -1.5, 1.5))
            yaw += omega * clocks.dt_sensory
            yaws[t], rates[t] = yaw, omega
        frames = np.stack([view_from_panorama(pano, y, size, fov) for y in yaws])
        hexals = eye(torch.from_numpy(frames).to(eye.device)).cpu().numpy()
        rel = relative_yaw(yaws, 0.0)
        actions = np.zeros((k, 4), dtype=np.float32)
        actions[:, 3] = np.clip(-rel[dec_t] / (np.pi / 2), -1.0, 1.0)  # steer back toward rel_yaw = 0
        p = trajectory_path(out_dir, i)
        Trajectory(hexals=hexals, actions=actions, rel_yaw=rel, yaw_rate=rates, yaw_0=0.0,
                   state={"yaw": yaws}, meta={"synthetic": True, "seed": seed + i}).save(p, clocks)
        paths.append(p)
    return paths


def main() -> int:
    t0 = time.time()
    base_cfg = load_config()
    smoke_dir = data_dir(base_cfg) / "smoke"
    traj_dir, cache_dir = smoke_dir / "expert", smoke_dir / "cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clocks = clocks_from_cfg(base_cfg)

    dm_path = default_direction_map_path()
    eye = BoxEye(int(base_cfg.vision.boxeye.extent), int(base_cfg.vision.boxeye.kernel_size), device=device)
    if not dm_path.exists():
        print(f"[smoke] {dm_path} missing -> measuring a provisional direction map")
        stepper = FlyvisStepper(base_cfg.vision.flyvis_model, device=device, dt=clocks.dt_sensory)
        dm = measure_directions(stepper, eye, frame_size=int(base_cfg.env.frame_size), seconds=1.0)
        dm_path = smoke_dir / "direction_map.json"
        dm_path.parent.mkdir(parents=True, exist_ok=True)
        dm_path.write_text(json.dumps(dm, indent=2))
        del stepper
    cfg = load_config(overrides=[
        f"lptc.direction_map={dm_path}",
        f"bc.batch_episodes={N_TRAJ}",
        f"paths.runs_dir={runs_dir(base_cfg)}",
    ])

    print("[smoke] 1/4 synthetic trajectories")
    make_trajectories(cfg, eye, traj_dir)
    print("[smoke] 2/4 build cache")
    build_cache_script.run(cfg, traj_dir, cache_dir, batch=N_TRAJ, device=device)
    print("[smoke] 3/4 BC (FlyWire connectome)")
    res = train_bc(cfg, cache_dir, runs_dir(cfg) / "smoke" / "bc", n_updates=N_UPDATES, device=device,
                   log_every=25)
    hist = res["history"]

    print("[smoke] 4/4 checks")
    losses = np.array([h["loss"] for h in hist])
    assert len(hist) == N_UPDATES, f"expected {N_UPDATES} updates, got {len(hist)}"
    assert np.isfinite(losses).all(), "non-finite loss"
    assert all(np.isfinite([h["action_mse"], h["heading_mse"]]).all() for h in hist)
    n = 15  # 5 episodes x 3 TBPTT windows: compares like-for-like window positions
    first, last = losses[:n].mean(), losses[-n:].mean()
    drop = 1 - last / first
    print(f"[smoke] total loss first{n}={first:.4f} last{n}={last:.4f} "
          f"drop={100 * drop:.1f}% (need >= {100 * MIN_DROP:.0f}%)")
    print(f"[smoke] heading decode R^2 (train episodes, no threshold): {res['metrics']['heading_r2']:.3f}  "
          f"err {res['metrics']['heading_err_deg']:.1f} deg")
    print(f"[smoke] elapsed {time.time() - t0:.1f} s (plumbing only, not a result)")
    assert drop >= MIN_DROP, f"loss dropped only {100 * drop:.1f}%"
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
