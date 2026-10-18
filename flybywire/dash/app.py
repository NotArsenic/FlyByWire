"""Streamlit dashboard. Not implemented yet.

Planned layout: FPV view on the left; the 16-wedge EPG ring
(:mod:`flybywire.eval.ring`, layout label shown) plus lesion toggles (``connectome.lesion_groups``)
on the right. Toggling a lesion calls ``model.lesion([...])`` / ``model.restore()`` between steps.
The demo video comes from ``scripts/render_lesion_video.py``, not from this live app.

Run::

    streamlit run flybywire/dash/app.py
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("dashboard is not implemented yet")


if __name__ == "__main__":
    main()
