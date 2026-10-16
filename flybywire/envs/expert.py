"""Scripted expert (STUB — implemented in week 1).

Spec: section "Environment" / "Scripted expert". Uses the privileged state in ``info``:

* potential-field obstacle avoidance (repulsion from obstacles and walls, attraction along the
  corridor axis),
* a heading PID that steers ``yaw`` back to ``yaw_0`` (relative heading 0) after dodging,
* a forward-speed target.

Outputs a normalised ``[-1, 1]^4`` action ``[v_forward, v_lateral, v_vertical, yaw_rate]`` once per
decision (25 Hz), using the same decision alignment as everything else (``flybywire.clocks``).
"""

from __future__ import annotations

import numpy as np


class ScriptedExpert:
    """Potential-field + heading-PID expert (STUB)."""

    def __init__(self, cfg, seed: int | None = None):
        """Read gains, speed target and action limits from ``cfg`` (STUB)."""
        raise NotImplementedError("week 1: scripted expert (spec 'Scripted expert')")

    def reset(self, info: dict) -> None:
        """Reset PID integrators and store ``yaw_0`` / obstacle layout from ``info`` (STUB)."""
        raise NotImplementedError("week 1: scripted expert")

    def act(self, info: dict) -> np.ndarray:
        """Return the normalised action for the current decision from privileged ``info`` (STUB)."""
        raise NotImplementedError("week 1: scripted expert")
