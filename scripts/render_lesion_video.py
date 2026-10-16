#!/usr/bin/env python
"""Render the lesion demo video (STUB).

Spec: "Dashboard and video". Plays a deterministic scripted episode (fixed seed) with a trained
policy, toggles lesions at fixed times (e.g. EPG at 3 s, restore at 6 s; then PEN; then D7) via
``model.lesion([...])`` / ``model.restore()``, and writes a side-by-side mp4: FPV view + drone
trajectory on the left, the 16-wedge EPG ring (layout labelled) on the right, with the active
lesion annotated. The demo video comes from this script, not from the live dashboard.

Usage (once implemented): python scripts/render_lesion_video.py --checkpoint runs/.../model.pt --out demo.mp4
"""

from __future__ import annotations


def main(argv=None) -> int:
    """Render the lesion video (STUB)."""
    raise NotImplementedError("needs the corridor env + trained policy (spec 'Dashboard and video')")


if __name__ == "__main__":
    raise SystemExit(main())
