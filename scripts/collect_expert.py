#!/usr/bin/env python
"""Collect scripted-expert trajectories (STUB — week 1).

Spec: "Data and storage" and "Scripted expert". For ``bc.n_trajectories`` (300) seeded episodes:
run :class:`flybywire.envs.expert.ScriptedExpert` in :class:`flybywire.envs.corridor.CorridorEnv`,
BoxEye every frame (store hexals, not frames), and save ``data/expert/traj_XXXXX.npz`` via
:class:`flybywire.data.trajectories.Trajectory` with the decision alignment from
``flybywire.clocks.Clocks`` (T = 4 * T_dec). ``--save-frames`` keeps raw frames for at most 20
episodes in ``data/frames/``. Metadata: seed, wall texture, config hash.

Usage (once implemented): python scripts/collect_expert.py [--n 300] [--save-frames] [key=value ...]
"""

from __future__ import annotations


def main(argv=None) -> int:
    """Collect expert trajectories (STUB)."""
    raise NotImplementedError("week 1: needs the corridor env and scripted expert (spec 'Scripted expert')")


if __name__ == "__main__":
    raise SystemExit(main())
