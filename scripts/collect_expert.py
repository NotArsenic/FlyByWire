#!/usr/bin/env python
"""Collect scripted-expert trajectories into <data_dir>/expert/traj_XXXXX.npz.

Episode i uses seed --seed + i for the corridor layout and the expert's wander. Workers run the
env and expert and send back raw frames; the main process converts them to hexals with BoxEye
and writes the format in flybywire/data/trajectories.py.

With expert.exec_noise set, the executed action carries OU noise but the stored label is the
expert's own action (DART); see ExecNoise in flybywire/envs/expert.py.

Window k (the reset pre-roll for k = 0) is the input to decision k. The window returned by the
last step has no decision, so it is dropped and every file has T = 4K frames.

--save-frames also keeps the raw frames of the first 20 episodes in <data_dir>/frames/.

Usage:
  python scripts/collect_expert.py [--n 300] [--workers 12] [--seed 0] [--out DIR] [--save-frames]
                                   [--overwrite] [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from flybywire.clocks import clocks_from_cfg
from flybywire.config import config_hash, data_dir, load_config, obstacle_texture
from flybywire.data.trajectories import STATE_KEYS, Trajectory, list_trajectories, trajectory_path

ENV_KEYS = ["clocks", "env", "action", "expert", "reward"]
MAX_SAVED_FRAMES = 20
_worker = {}


def _init_worker(overrides: list[str]) -> None:
    from flybywire.envs.corridor import CorridorEnv

    cfg = load_config(overrides=overrides)
    _worker["cfg"] = cfg
    _worker["env"] = CorridorEnv(cfg)


def run_episode(seed: int) -> dict:
    from flybywire.envs.expert import ExecNoise, ScriptedExpert

    cfg, env = _worker["cfg"], _worker["env"]
    expert = ScriptedExpert(cfg, seed=seed)
    spec = OmegaConf.select(cfg, "expert.exec_noise", default=None)
    noise = ExecNoise(spec, clocks_from_cfg(cfg).dt_control, seed) if spec else None
    frames, info = env.reset(seed=seed)
    expert.reset(info)
    windows, infos, actions = [frames], [info], []
    while True:
        a = expert.act(info)
        actions.append(a)
        frames, _, terminated, truncated, info = env.step(a if noise is None else noise(a))
        if terminated or truncated:
            break
        windows.append(frames)
        infos.append(info)
    reason = ("collision" if info["collision"] else "end" if info["reached_end"]
              else "out_of_bounds" if info["out_of_bounds"] else "time")
    return {
        "seed": seed,
        "frames": np.concatenate(windows),
        "actions": np.stack(actions).astype(np.float32),
        "state": {k: np.concatenate([i[k] for i in infos]) for k in (*STATE_KEYS, "rel_yaw", "yaw_rate")},
        "yaw_0": infos[0]["yaw_0"],
        "collided": bool(info["collision"]),
        "reason": reason,
        "x_progress": float(info["x_progress"]),
        "renderer": env.renderer,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help="episodes (default bc.n_trajectories)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None, help="first episode seed (default cfg.seed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace existing traj_*.npz in --out")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(overrides=args.overrides)
    clocks = clocks_from_cfg(cfg)
    n = int(args.n or cfg.bc.n_trajectories)
    seed0 = int(cfg.seed if args.seed is None else args.seed)
    out = Path(args.out) if args.out else data_dir(cfg) / "expert"
    frames_dir = data_dir(cfg) / "frames"
    existing = list_trajectories(out)
    if existing and not args.overwrite:
        raise SystemExit(f"{out} already has {len(existing)} trajectories; pass --overwrite or another --out")
    for p in existing:
        p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from flybywire.vision.boxeye import BoxEye

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device=device)
    env_hash = config_hash(cfg, ENV_KEYS)
    exec_noise = OmegaConf.select(cfg, "expert.exec_noise", default=None)
    print(f"collecting {n} episodes (seeds {seed0}..{seed0 + n - 1}) with {args.workers} workers -> {out}")

    t0 = time.time()
    summary = {"time": 0, "collision": 0, "end": 0, "out_of_bounds": 0}
    lengths, progress = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(list(args.overrides),)) as pool:
        from tqdm import tqdm

        for ep in tqdm(pool.imap_unordered(run_episode, range(seed0, seed0 + n)), total=n, unit="ep"):
            i = ep["seed"] - seed0
            frames = ep["frames"]
            hexals = torch.cat([eye(torch.from_numpy(frames[s : s + 256]).to(device)).cpu()
                                for s in range(0, len(frames), 256)]).numpy()
            st = ep["state"]
            Trajectory(
                hexals=hexals,
                actions=ep["actions"],
                rel_yaw=st["rel_yaw"],
                yaw_rate=st["yaw_rate"],
                collided=ep["collided"],
                yaw_0=ep["yaw_0"],
                state={k: st[k] for k in STATE_KEYS},
                meta={
                    "seed": ep["seed"],
                    "wall_texture": str(cfg.env.wall_texture),
                    "obstacle_texture": obstacle_texture(cfg),
                    "renderer": ep["renderer"],
                    "config_hash": env_hash,
                    "termination": ep["reason"],
                    "x_progress": ep["x_progress"],
                },
            ).save(trajectory_path(out, i), clocks)
            if args.save_frames and i < MAX_SAVED_FRAMES:
                frames_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(frames_dir / f"traj_{i:05d}.npz", frames=frames)
            summary[ep["reason"]] += 1
            lengths.append(len(ep["actions"]))
            progress.append(ep["x_progress"])

    size = sum(p.stat().st_size for p in list_trajectories(out))
    report = {
        "n": n,
        "terminations": summary,
        "decisions_mean": float(np.mean(lengths)),
        "x_progress_mean_m": float(np.mean(progress)),
        "exec_noise": OmegaConf.to_container(exec_noise) if exec_noise else None,
        "seconds": round(time.time() - t0, 1),
        "bytes": size,
        "config_hash": env_hash,
    }
    (out / "collect.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
