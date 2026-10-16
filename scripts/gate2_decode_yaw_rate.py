#!/usr/bin/env python
"""Gate 2: is yaw rate linearly decodable from T4/T5 activity in the corridor? (STUB — week 1)

Procedure: from the T4/T5 cache of expert trajectories, fit ridge regression from (optionally
spatially pooled) T4/T5 activity to body yaw rate, using a trajectory-level train/test split
(no episode in both).

PASS CRITERION: ridge regression from T4/T5 to yaw rate on held-out trajectories reaches R² > 0.5.

If it fails: retry with ``env.wall_texture=checker`` / ``noise`` (the textures exist for this
purpose) before considering ``vision.backend=farneback`` (the user's decision).
Writes runs/gate2/report.json.
"""

from __future__ import annotations


def main(argv=None) -> int:
    """Run Gate 2 (STUB)."""
    raise NotImplementedError("week 1: needs expert data + cache (spec 'Gates')")


if __name__ == "__main__":
    raise SystemExit(main())
