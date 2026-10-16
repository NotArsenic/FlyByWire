"""Streamlit dashboard (STUB).

Spec: section "Dashboard and video". Layout: FPV view on the left; the 16-wedge EPG ring
(:mod:`flybywire.eval.ring`, layout label shown) plus lesion toggles (``connectome.lesion_groups``)
on the right. Toggling a lesion calls ``model.lesion([...])`` / ``model.restore()`` between steps.
The demo video comes from ``scripts/render_lesion_video.py``, not from this live app.

Run (once implemented)::

    streamlit run flybywire/dash/app.py
"""

from __future__ import annotations


def main() -> None:
    """Build the Streamlit page (STUB)."""
    raise NotImplementedError("dashboard (spec 'Dashboard and video')")


if __name__ == "__main__":
    main()
