"""Streamlit dashboard: a trained policy flies the corridor and cell types can be silenced mid-flight.

Same views as the lesion video (flybywire/eval/live.py): chase camera, what the eye sees, the
corridor from above and the EPG ring on the anatomical EB layout. Changing a lesion takes effect
from the next decision (``model.lesion`` / ``model.restore``); the flight itself carries on.
Flights last the checkpoint's own episode length (10 s, as in training and evaluation) on the
corridor level picked in the sidebar.
The video for the README comes from scripts/render_lesion_video.py, not from here.

Run::

    streamlit run flybywire/dash/app.py -- [--ckpt runs/curriculum/dense_s5/model.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from flybywire.config import REPO_ROOT, load_config
from flybywire.eval.live import Flight, draw, figure_rgb

DEFAULT_CKPT = REPO_ROOT / "runs/curriculum/dense_s5/model.pt"
LEVELS = ("default", "spaced", "dense", "slalom", "forest")


def level_overrides(level: str) -> list[str]:
    # the checkpoint's config holds whatever corridor it was trained on, so reset the layout keys to
    # the default corridor first and then apply the level (configs/default.yaml curriculum)
    cfg = load_config()
    env = cfg.env
    spacing = "null" if env.obstacle_spacing_m is None else env.obstacle_spacing_m
    base = [f"env.n_obstacles={env.n_obstacles}", f"env.obstacle_spacing_m={spacing}", f"env.pair_prob={env.pair_prob}"]
    return base + ([] if level == "default" else list(cfg.curriculum[level]))


@st.cache_resource
def get_flight(ckpt: str) -> Flight:
    flight = Flight(ckpt, overrides=level_overrides("default"))
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
        overrides = level_overrides(st.selectbox("corridor level", LEVELS))
        if overrides != flight.overrides:
            flight.set_overrides(overrides)
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
    # not pyplot: streamlit runs each rerun on its own thread and pyplot's global state is not shared safely
    fig = Figure(figsize=(12, 7.2), dpi=90)
    FigureCanvasAgg(fig)
    draw(fig, flight)
    view.image(figure_rgb(fig))
    while flying and not flight.done:
        flight.step()
        draw(fig, flight)
        view.image(figure_rgb(fig))
    if flight.done:
        status.info(f"episode ended: {flight.reason} at {flight.time:.1f} s. Press 'new flight' or change the seed.")


main()
