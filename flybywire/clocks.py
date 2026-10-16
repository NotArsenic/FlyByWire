"""Clock rates and the decision/action timing alignment.

This module is the **only** place that defines how physics steps, sensory (core) steps and
actor decisions line up. Data collection, BC and PPO must all use :class:`Clocks`.

Timeline for one episode with ``K`` decisions (defaults: 200 / 100 / 25 Hz)::

    physics step p :  0 1 | 2 3 | 4 5 | 6 7 || 8 9 | ...
    frame / core t :    0 |   1 |   2 |   3 ||   4 | ...
    decision k     :                      0 ||                 1 ...
    action held    :  a_hover (pre-roll)    ||  a_0 for physics 8..15, ...

* Frame ``t`` is rendered after physics steps ``0 .. 2t+1`` have run (sim time ``(t+1)/sensory_hz``)
  and is consumed by core step ``t``. The sensory path never skips a step.
* Decision ``k`` is read out **after** core step ``4k + 3`` (the last tick of its 4-tick window),
  so every decision sees exactly 4 fresh frames.
* Action ``a_k`` is held over physics steps ``8(k+1) .. 8(k+1)+7`` (core steps ``4(k+1) .. 4(k+1)+3``).
* The first window (physics ``0..7``, frames ``0..3``) is a pre-roll during which the env holds a
  hover command (zero body velocity, zero yaw rate); it produces the frames for decision 0.
* An episode of ``K`` decisions therefore has exactly ``T = 4K`` frames / core steps.
"""

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
            raise ValueError(f"physics_hz={self.physics_hz} not divisible by sensory_hz={self.sensory_hz}")
        if self.sensory_hz % self.control_hz:
            raise ValueError(f"sensory_hz={self.sensory_hz} not divisible by control_hz={self.control_hz}")

    # --- ratios -----------------------------------------------------------------
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
        """flyvis integration step and core step (seconds)."""
        return 1.0 / self.sensory_hz

    @property
    def dt_control(self) -> float:
        return 1.0 / self.control_hz

    # --- alignment --------------------------------------------------------------
    def n_decisions(self, episode_seconds: float) -> int:
        k = episode_seconds * self.control_hz
        if abs(k - round(k)) > 1e-9:
            raise ValueError(f"episode_seconds={episode_seconds} is not a whole number of decisions")
        return int(round(k))

    def n_core_steps(self, n_decisions: int) -> int:
        """Frames / core steps in an episode with ``n_decisions`` decisions (``T = 4K``)."""
        return n_decisions * self.sensory_per_decision

    def decision_core_step(self, k: int) -> int:
        """Core step after which decision ``k`` is read out."""
        return (k + 1) * self.sensory_per_decision - 1

    def decision_core_steps(self, n_decisions: int) -> np.ndarray:
        """Indices ``t`` (into a length-``4K`` core sequence) at which decisions are read out."""
        return np.arange(1, n_decisions + 1) * self.sensory_per_decision - 1

    def is_decision_step(self, t: int) -> bool:
        return (t + 1) % self.sensory_per_decision == 0

    def decision_of_core_step(self, t: int) -> int:
        """Index of the decision whose observation window contains core step ``t``."""
        return t // self.sensory_per_decision

    def action_core_steps(self, k: int) -> range:
        """Core steps whose frames are rendered while action ``a_k`` is held."""
        s = self.sensory_per_decision
        return range((k + 1) * s, (k + 2) * s)

    def action_physics_steps(self, k: int) -> range:
        """Physics steps during which action ``a_k`` is held (pre-roll occupies ``0..P-1``)."""
        p = self.physics_per_decision
        return range((k + 1) * p, (k + 2) * p)

    def frame_physics_step(self, t: int) -> int:
        """Physics step after which frame ``t`` is rendered."""
        return (t + 1) * self.physics_per_sensory - 1

    def frame_time(self, t: int) -> float:
        """Simulation time (s) at which frame ``t`` is rendered."""
        return (t + 1) / self.sensory_hz

    def windows(self, n_core_steps: int, window: int) -> list[tuple[int, int]]:
        """Split ``[0, T)`` into consecutive TBPTT windows of ``window`` core steps.

        ``window`` must be a multiple of ``sensory_per_decision`` so no decision window is split.
        """
        if window % self.sensory_per_decision:
            raise ValueError(f"TBPTT window {window} must be a multiple of {self.sensory_per_decision}")
        return [(s, min(s + window, n_core_steps)) for s in range(0, n_core_steps, window)]


def clocks_from_cfg(cfg) -> Clocks:
    c = cfg.clocks
    return Clocks(int(c.physics_hz), int(c.sensory_hz), int(c.control_hz))
