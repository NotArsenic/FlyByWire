#!/usr/bin/env python
"""Gate 3: does the trained core carry a heading representation in EPG? (STUB)

Procedure: after BC on the FlyWire connectome (real mask), run held-out episodes, decode heading
relative to episode start from EPG rates with the heading decoder, compute
:func:`flybywire.eval.metrics.heading_r2` (joint on sin/cos) and the angular error, and render the
16-wedge ring (:mod:`flybywire.eval.ring`) against true heading over time.

PASS CRITERION: heading decode from EPG reaches R² > 0.8, AND the ring shows a single bump that
rotates with yaw.
Writes runs/gate3/report.json and a ring-vs-heading figure.
"""

from __future__ import annotations


def main(argv=None) -> int:
    """Run Gate 3 (STUB)."""
    raise NotImplementedError("needs FlyWire connectome + trained BC model (spec 'Gates')")


if __name__ == "__main__":
    raise SystemExit(main())
