#!/usr/bin/env python
"""Gate 3: does the trained core carry a heading representation in EPG? Not implemented yet.

Procedure: after BC on the FlyWire connectome (real mask), run held-out episodes, decode heading
relative to episode start from EPG rates with the heading decoder, compute
:func:`flybywire.eval.metrics.heading_r2` (joint on sin/cos) and the angular error, and render the
16-wedge ring (:mod:`flybywire.eval.ring`) against true heading over time.

Pass criterion: heading decode from EPG reaches R^2 > 0.8, and the ring shows a single bump that
rotates with yaw.
Writes runs/gate3/report.json and a ring-vs-heading figure.
"""

from __future__ import annotations


def main(argv=None) -> int:
    raise NotImplementedError("needs a BC model trained on the FlyWire graph")


if __name__ == "__main__":
    raise SystemExit(main())
