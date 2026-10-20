"""Streamlit dashboard: a trained policy flies the corridor and cell types can be silenced mid-flight.

Same views as the lesion video (flybywire/eval/live.py): chase camera, what the eye sees, the
corridor from above and the EPG ring on the anatomical EB layout. Changing a lesion takes effect
from the next decision (``model.lesion`` / ``model.restore``); the flight itself carries on.
The video for the README comes from scripts/render_lesion_video.py, not from here.

Run::

    streamlit run flybywire/dash/app.py -- [--ckpt runs/curriculum/dense_s5/model.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from flybywire.config import REPO_ROOT  # noqa: E402
from flybywire.eval.live import Flight, draw, figure_rgb  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "runs/curriculum/dense_s5/model.pt"


@st.cache_resource
def get_flight(ckpt: str) -> Flight:
    flight = Flight(ckpt, overrides=["env.episode_seconds=20"])
    flight.align_ring(REPO_ROOT / "runs/lesions" / Path(ckpt).parent.name / "lesions.json")
    return flight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    args = ap.parse_args(sys.argv[1:])

    st.set_page_config(page_title="Fly-By-Wire", layout="wide")
    flight = get_flight(args.ckpt)
    with st.sidebar:
        st.markdown(f"**checkpoint**  \n`{Path(args.ckpt).parent.name}/{Path(args.ckpt).name}`")
        seed = int(st.number_input("corridor seed (held-out: 10000 and up)", value=flight.seed, step=1))
        if st.button("new flight") or seed != flight.seed:
            flight.reset(seed)
        lesions = st.multiselect("silence", list(flight.model.lesion_groups), default=list(flight.lesions))
        flying = st.toggle("fly", value=True)
        st.caption("A lesion sets the rates of that cell type to exactly 0 from the next step on.")
    if tuple(lesions) != flight.lesions:
        flight.set_lesions(lesions)

    view = st.empty()
    status = st.empty()
    fig = plt.figure(figsize=(12, 7.2), dpi=90)
    draw(fig, flight)
    view.image(figure_rgb(fig))
    while flying and not flight.done:
        flight.step()
        draw(fig, flight)
        view.image(figure_rgb(fig))
    if flight.done:
        status.info(f"episode ended: {flight.reason} at {flight.time:.1f} s. Press 'new flight' or change the seed.")
    plt.close(fig)


main()
