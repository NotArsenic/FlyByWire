"""Expert trajectory files ``data/expert/traj_XXXXX.npz``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flybywire.clocks import Clocks

REQUIRED = ("hexals", "actions", "rel_yaw", "yaw_rate")
STATE_KEYS = ("pos", "quat", "vel", "ang_vel", "yaw")


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def relative_yaw(yaw, yaw_0):
    """Heading relative to episode start: wrap(yaw_t - yaw_0)."""
    return wrap_angle(np.asarray(yaw) - yaw_0)


def heading_target(rel_yaw):
    """(sin, cos) of relative yaw, shape (..., 2)."""
    rel_yaw = np.asarray(rel_yaw)
    return np.stack([np.sin(rel_yaw), np.cos(rel_yaw)], axis=-1)


@dataclass
class Trajectory:
    hexals: np.ndarray
    actions: np.ndarray
    rel_yaw: np.ndarray
    yaw_rate: np.ndarray
    collided: bool = False
    yaw_0: float = 0.0
    state: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_core_steps(self) -> int:
        return int(self.hexals.shape[0])

    @property
    def n_decisions(self) -> int:
        return int(self.actions.shape[0])

    def validate(self, clocks: Clocks | None = None, n_hexals: int = 721) -> None:
        clocks = clocks or Clocks()
        t = self.n_core_steps
        if self.hexals.ndim != 2 or self.hexals.shape[1] != n_hexals:
            raise ValueError(f"hexals must be (T, {n_hexals}), got {self.hexals.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] != 4:
            raise ValueError(f"actions must be (T_dec, 4), got {self.actions.shape}")
        if clocks.n_core_steps(self.n_decisions) != t:
            raise ValueError(
                f"T={t} core steps but {self.n_decisions} decisions; expected T = "
                f"{clocks.sensory_per_decision} * T_dec (clocks.py alignment)"
            )
        for k in ("rel_yaw", "yaw_rate"):
            if getattr(self, k).shape != (t,):
                raise ValueError(f"{k} must be (T,), got {getattr(self, k).shape}")
        for k, v in self.state.items():
            if v.shape[0] != t:
                raise ValueError(f"state {k} has {v.shape[0]} steps, expected {t}")
        if np.abs(self.actions).max(initial=0) > 1 + 1e-6:
            raise ValueError("actions must be normalised to [-1, 1]")

    def save(self, path: str | Path, clocks: Clocks | None = None) -> None:
        self.validate(clocks)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hexals=self.hexals.astype(np.float16),
            actions=self.actions.astype(np.float32),
            rel_yaw=self.rel_yaw.astype(np.float32),
            yaw_rate=self.yaw_rate.astype(np.float32),
            collided=np.asarray(bool(self.collided)),
            yaw_0=np.asarray(self.yaw_0, dtype=np.float32),
            meta=np.asarray(json.dumps(self.meta)),
            **{k: v.astype(np.float32) for k, v in self.state.items()},
        )

    @classmethod
    def load(cls, path: str | Path, load_hexals: bool = True) -> Trajectory:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in REQUIRED if k not in z.files]
            if missing:
                raise ValueError(f"{path} is missing {missing}")
            return cls(
                hexals=(
                    z["hexals"]
                    if load_hexals
                    else np.empty((len(z["rel_yaw"]), 0), np.float16)
                ),
                actions=z["actions"],
                rel_yaw=z["rel_yaw"],
                yaw_rate=z["yaw_rate"],
                collided=bool(z["collided"]) if "collided" in z.files else False,
                yaw_0=float(z["yaw_0"]) if "yaw_0" in z.files else 0.0,
                state={k: z[k] for k in STATE_KEYS if k in z.files},
                meta=json.loads(str(z["meta"])) if "meta" in z.files else {},
            )


def trajectory_path(directory: str | Path, index: int) -> Path:
    return Path(directory) / f"traj_{index:05d}.npz"


def list_trajectories(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("traj_*.npz"))
