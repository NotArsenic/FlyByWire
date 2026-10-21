"""Clock rates and timing alignment across physics, sensory, and control loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clocks:
    physics_hz: int = 200
    sensory_hz: int = 100
    control_hz: int = 25

    def __post_init__(self) -> None:
        for name in ("physics_hz", "sensory_hz", "control_hz"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"clocks.{name} must be a positive int, got {v!r}")
        if self.physics_hz % self.sensory_hz:
            raise ValueError(
                f"physics_hz={self.physics_hz} not divisible by sensory_hz={self.sensory_hz}"
            )
        if self.sensory_hz % self.control_hz:
            raise ValueError(
                f"sensory_hz={self.sensory_hz} not divisible by control_hz={self.control_hz}"
            )

    @property
    def physics_per_sensory(self) -> int:
        return self.physics_hz // self.sensory_hz

    @property
    def sensory_per_decision(self) -> int:
        return self.sensory_hz // self.control_hz

    @property
    def physics_per_decision(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def dt_physics(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def dt_sensory(self) -> float:
        return 1.0 / self.sensory_hz

    @property
    def dt_control(self) -> float:
        return 1.0 / self.control_hz

    def n_decisions(self, episode_seconds: float) -> int:
        k = episode_seconds * self.control_hz
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                f"episode_seconds={episode_seconds} is not a whole number of decisions"
            )
        return int(round(k))

    def n_core_steps(self, n_decisions: int) -> int:
        return n_decisions * self.sensory_per_decision

    def decision_core_step(self, k: int) -> int:
        return (k + 1) * self.sensory_per_decision - 1

    def decision_core_steps(self, n_decisions: int) -> np.ndarray:
        return np.arange(1, n_decisions + 1) * self.sensory_per_decision - 1

    def is_decision_step(self, t: int) -> bool:
        return (t + 1) % self.sensory_per_decision == 0

    def decision_of_core_step(self, t: int) -> int:
        return t // self.sensory_per_decision

    def action_core_steps(self, k: int) -> range:
        s = self.sensory_per_decision
        return range((k + 1) * s, (k + 2) * s)

    def action_physics_steps(self, k: int) -> range:
        p = self.physics_per_decision
        return range((k + 1) * p, (k + 2) * p)

    def frame_physics_step(self, t: int) -> int:
        return (t + 1) * self.physics_per_sensory - 1

    def frame_time(self, t: int) -> float:
        return (t + 1) / self.sensory_hz

    def windows(self, n_core_steps: int, window: int) -> list[tuple[int, int]]:
        """Split [0, T) into consecutive TBPTT windows of length `window`."""
        if window % self.sensory_per_decision:
            raise ValueError(
                f"TBPTT window {window} must be a multiple of {self.sensory_per_decision}"
            )
        return [
            (s, min(s + window, n_core_steps)) for s in range(0, n_core_steps, window)
        ]


def clocks_from_cfg(cfg) -> Clocks:
    c = cfg.clocks
    return Clocks(int(c.physics_hz), int(c.sensory_hz), int(c.control_hz))
