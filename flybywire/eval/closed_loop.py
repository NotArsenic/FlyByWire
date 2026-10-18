"""Closed-loop flight in the corridor env: the trained network, or the scripted expert as a baseline.

Envs run in worker processes (``make_vec_env``); BoxEye, flyvis and the core run batched here.
Each decision window is 4 frames: every frame goes through BoxEye, the flyvis stepper and one core
step, and the action is the actor mean read after the 4th frame, which is the decision step of
``Clocks`` (the same step ``model.run`` reads).

Every episode gets an explicit seed, so the layout for seed s is the one ``collect_expert.py`` would
use. With SAME_STEP autoreset a finished env already holds the next, unseeded episode; it is
replaced by a seeded reset before any of its frames are used, and the core and flyvis state of that
env are zeroed first.

Recording (``record_dir``) writes ``Trajectory`` files with the hexals the policy saw and, as
actions, a scripted expert's label for each visited state. The labelling expert gets the same
infos as a flying one would, with its random wander switched off (``label_wander_deg=0``): the
wander exists to diversify the states a flying expert visits, but in DAgger the policy's own errors
do that, and wander in a label is noise the observations cannot predict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flybywire.clocks import clocks_from_cfg
from flybywire.config import ACTION_KEYS, config_hash, data_dir
from flybywire.data.trajectories import STATE_KEYS, Trajectory, trajectory_path, wrap_angle

ENV_KEYS = ["clocks", "env", "action", "expert", "reward"]
STATE_FIELDS = (*STATE_KEYS, "rel_yaw", "yaw_rate")
NOT_INFO = ("final_info", "final_obs")


def load_checkpoint(path, device="cpu"):
    """Model (eval mode) and the config it was trained with."""
    from flybywire.core.model import build_model

    ck = torch.load(path, map_location="cpu")
    cfg = OmegaConf.create(ck["config"])
    model = build_model(cfg, verbose=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def checkpoint_cache(path, cfg) -> Path:
    """The cache a checkpoint was trained on, from its metrics.json (runs from before that: data/cache)."""
    metrics = Path(path).parent / "metrics.json"
    if metrics.exists():
        trained_on = json.loads(metrics.read_text()).get("cache")
        if trained_on:
            return Path(trained_on)
    return data_dir(cfg) / "cache"


def heading_error_deg(pred, rel_yaw) -> np.ndarray:
    """Wrapped angle between the decoded (sin, cos) and the true relative yaw, per step."""
    pred = np.asarray(pred)
    return np.degrees(np.abs(wrap_angle(np.arctan2(pred[:, 0], pred[:, 1]) - rel_yaw)))


class PolicyDriver:
    """The network flies; a scripted expert (no wander) labels every state it visits."""

    def __init__(self, model, cfg, n_envs: int, label_wander_deg: float = 0.0, label_maneuvers: bool = True):
        from flybywire.vision.boxeye import BoxEye
        from flybywire.vision.flyvis_stepper import FlyvisStepper

        self.model = model
        self.device = next(model.parameters()).device
        clocks = clocks_from_cfg(cfg)
        self.eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device=self.device)
        self.stepper = FlyvisStepper(
            model=cfg.vision.flyvis_model,
            device=self.device,
            dt=clocks.dt_sensory,
            cell_types=tuple(cfg.vision.cell_types),
            steady_state_seconds=float(cfg.vision.steady_state_seconds),
        )
        self.stepper.reset(n_envs)
        self.v = model.init_state(n_envs)
        self.label_cfg = OmegaConf.merge(cfg, {"expert": {"wander_deg": label_wander_deg}})
        if not label_maneuvers:
            self.label_cfg.expert.maneuvers = None
        self.experts = [None] * n_envs

    def start(self, j: int, seed: int, info: dict) -> None:
        from flybywire.envs.expert import ScriptedExpert

        self.v[j] = 0
        self.stepper.reset(env_ids=[j])
        self.experts[j] = ScriptedExpert(self.label_cfg, seed=seed)
        self.experts[j].reset(info)

    def stop(self, j: int) -> None:
        self.experts[j] = None

    @torch.no_grad()
    def act(self, frames: np.ndarray, infos: list) -> dict:
        n, w = frames.shape[:2]
        x = torch.from_numpy(frames).to(self.device)
        # fp16 round trip: trajectories store hexals as fp16, and the cache was built from those
        hexals = self.eye(x.reshape(n * w, *x.shape[2:])).half().float().reshape(n, w, -1)
        heading = []
        for t in range(w):
            self.v, r = self.model.core_step(self.v, self.stepper.step(hexals[:, t]))
            heading.append(self.model.heading(r))
        label = np.stack([e.act(i) if e is not None else np.zeros(4, np.float32)
                          for e, i in zip(self.experts, infos, strict=True)])
        return {
            "action": self.model.actor(r).cpu().numpy(),
            "label": label,
            "heading": torch.stack(heading, dim=1).cpu().numpy(),
            "hexals": hexals.half().cpu().numpy(),
        }


class ExpertDriver:
    """The scripted expert flies (random wander on, seeded like ``collect_expert.py``)."""

    def __init__(self, cfg, n_envs: int):
        self.cfg = cfg
        self.experts = [None] * n_envs

    def start(self, j: int, seed: int, info: dict) -> None:
        from flybywire.envs.expert import ScriptedExpert

        self.experts[j] = ScriptedExpert(self.cfg, seed=seed)
        self.experts[j].reset(info)

    def stop(self, j: int) -> None:
        self.experts[j] = None

    def act(self, frames: np.ndarray, infos: list) -> dict:
        action = np.stack([e.act(i) if e is not None else np.zeros(4, np.float32)
                           for e, i in zip(self.experts, infos, strict=True)])
        return {"action": action, "label": action}


@dataclass
class Episode:
    index: int
    seed: int
    reason: str  # collision, out_of_bounds, end or time
    x_progress: float
    collided: bool
    action: np.ndarray  # (K, 4) flown
    label: np.ndarray  # (K, 4) scripted expert on the visited states
    heading_err_deg: np.ndarray | None = None  # (T,)
    file: str | None = None

    @property
    def n_decisions(self) -> int:
        return len(self.action)

    @property
    def success(self) -> bool:
        return self.reason in ("end", "time")


@dataclass
class Rollout:
    index: int
    seed: int
    infos: list = field(default_factory=list)  # one per window: what decision k saw
    outs: list = field(default_factory=list)


def env_info(info: dict, j: int) -> dict:
    return {k: v[j] for k, v in info.items() if not k.startswith("_") and k not in NOT_INFO}


def fly(cfg, seeds, driver, n_envs: int = 12, record_dir=None, meta: dict | None = None,
        progress: bool = False) -> list[Episode]:
    """Fly one episode per seed. Returns episodes in seed order.

    ``driver`` is a PolicyDriver or ExpertDriver built for ``n_envs`` envs. With ``record_dir`` every
    episode is also saved as ``traj_XXXXX.npz`` (XXXXX = position in ``seeds``); only a PolicyDriver
    provides hexals to record.
    """
    from tqdm import tqdm

    from flybywire.envs.corridor import normalize_action
    from flybywire.envs.vec import make_vec_env

    clocks = clocks_from_cfg(cfg)
    hover = normalize_action(np.zeros(4), cfg.action.limits).astype(np.float32)
    queue = iter(enumerate(seeds))
    bar = tqdm(total=len(seeds), unit="ep", disable=not progress)
    vec = make_vec_env(cfg, n_envs)
    try:
        first = [next(queue, None) for _ in range(n_envs)]
        frames, info = vec.reset(seed=[None if f is None else f[1] for f in first])
        rolls = [None if f is None else Rollout(*f) for f in first]
        infos = [None if r is None else env_info(info, j) for j, r in enumerate(rolls)]
        for j, r in enumerate(rolls):
            if r is not None:
                driver.start(j, r.seed, infos[j])

        episodes: list[Episode] = []
        while any(r is not None for r in rolls):
            out = driver.act(frames, infos)
            actions = out["action"].astype(np.float32)
            for j, r in enumerate(rolls):
                if r is None:
                    actions[j] = hover
                else:
                    r.infos.append(infos[j])
                    r.outs.append({k: v[j] for k, v in out.items()})
            frames, _, terminated, truncated, info = vec.step(actions)

            free = []
            for j, r in enumerate(rolls):
                if r is None:
                    continue
                if terminated[j] or truncated[j]:
                    episodes.append(_finish(r, env_info(info["final_info"], j), cfg, clocks, record_dir, meta))
                    bar.update()
                    rolls[j], infos[j] = None, None
                    driver.stop(j)
                    free.append(j)
                else:
                    infos[j] = env_info(info, j)

            mask, new_seeds = np.zeros(n_envs, dtype=bool), [None] * n_envs
            for j in free:
                nxt = next(queue, None)
                if nxt is not None:
                    mask[j], new_seeds[j] = True, nxt[1]
                    rolls[j] = Rollout(*nxt)
            if mask.any():
                new_frames, new_info = vec.reset(seed=new_seeds, options={"reset_mask": mask})
                for j in np.flatnonzero(mask):
                    frames[j] = new_frames[j]
                    infos[j] = env_info(new_info, j)
                    driver.start(j, rolls[j].seed, infos[j])
    finally:
        vec.close()
        bar.close()
    return sorted(episodes, key=lambda e: e.index)


def _finish(r: Rollout, final: dict, cfg, clocks, record_dir, meta) -> Episode:
    reason = ("collision" if final["collision"] else "end" if final["reached_end"]
              else "out_of_bounds" if final["out_of_bounds"] else "time")
    state = {k: np.concatenate([i[k] for i in r.infos]) for k in STATE_FIELDS}
    action = np.stack([o["action"] for o in r.outs])
    label = np.stack([o["label"] for o in r.outs])
    heading_err = None
    if "heading" in r.outs[0]:
        heading_err = heading_error_deg(np.concatenate([o["heading"] for o in r.outs]), state["rel_yaw"])
    ep = Episode(r.index, r.seed, reason, float(final["x_progress"]), bool(final["collision"]), action, label,
                 heading_err)
    if record_dir is not None:
        path = trajectory_path(record_dir, r.index)
        Trajectory(
            hexals=np.concatenate([o["hexals"] for o in r.outs]),
            actions=label,
            rel_yaw=state["rel_yaw"],
            yaw_rate=state["yaw_rate"],
            collided=ep.collided,
            yaw_0=float(r.infos[0]["yaw_0"]),
            state={k: state[k] for k in STATE_KEYS},
            meta={"seed": r.seed, "wall_texture": str(cfg.env.wall_texture), "config_hash": config_hash(cfg, ENV_KEYS),
                  "termination": reason, "x_progress": ep.x_progress,
                  "action_mse_policy_vs_label": float(np.mean((action - label) ** 2)), **(meta or {})},
        ).save(path, clocks)
        ep.file = str(path)
    return ep


def summarize(episodes: list[Episode], clocks) -> dict:
    reasons = [e.reason for e in episodes]
    out = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([e.success for e in episodes])),
        "terminations": {k: reasons.count(k) for k in ("collision", "out_of_bounds", "end", "time")},
        "collisions": int(sum(e.collided for e in episodes)),
        "decisions_mean": float(np.mean([e.n_decisions for e in episodes])),
        "x_progress_mean_m": float(np.mean([e.x_progress for e in episodes])),
        "x_progress_sd_m": float(np.std([e.x_progress for e in episodes])),
    }
    gap = np.concatenate([(e.action - e.label) ** 2 for e in episodes])
    out["action_mse_vs_label"] = float(gap.mean())
    out["action_mse_vs_label_per_dim"] = dict(zip(ACTION_KEYS, gap.mean(0).round(5).tolist(), strict=True))
    if episodes[0].heading_err_deg is not None:
        err = [e.heading_err_deg for e in episodes]
        pooled = np.concatenate(err)
        sec = clocks.sensory_hz
        n_sec = max(len(x) for x in err) // sec
        by_second = []
        for s in range(n_sec):
            part = np.concatenate([x[s * sec : (s + 1) * sec] for x in err])
            by_second.append(float(part.mean()))
        out["heading_err_deg"] = {
            "mean": float(pooled.mean()),
            "median": float(np.median(pooled)),
            "p95": float(np.quantile(pooled, 0.95)),
            "mean_after_1s": float(np.concatenate([x[sec:] for x in err]).mean()),
            "by_second": by_second,
        }
    return out


def episode_rows(episodes: list[Episode]) -> list[dict]:
    rows = []
    for e in episodes:
        row = {"index": e.index, "seed": e.seed, "termination": e.reason, "decisions": e.n_decisions,
               "x_progress_m": e.x_progress, "collision": e.collided,
               "action_mse_vs_label": float(np.mean((e.action - e.label) ** 2))}
        if e.heading_err_deg is not None:
            row["heading_err_deg_mean"] = float(e.heading_err_deg.mean())
        if e.file:
            row["file"] = e.file
        rows.append(row)
    return rows
