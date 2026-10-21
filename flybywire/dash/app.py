"""Streamlit dashboard: a trained policy flies the corridor and cell types can be silenced mid-flight.

Run:

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

    cfg = load_config()
    env = cfg.env
    spacing = "null" if env.obstacle_spacing_m is None else env.obstacle_spacing_m
    base = [
        f"env.n_obstacles={env.n_obstacles}",
        f"env.obstacle_spacing_m={spacing}",
        f"env.pair_prob={env.pair_prob}",
    ]
    return base + ([] if level == "default" else list(cfg.curriculum[level]))


@st.cache_resource
def get_flight(ckpt: str) -> Flight:
    flight = Flight(ckpt, overrides=level_overrides("default"))
    flight.align_ring(
        REPO_ROOT / "runs/lesions" / Path(ckpt).parent.name / "lesions.json"
    )
    return flight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    args = ap.parse_args(sys.argv[1:])

    st.set_page_config(page_title="Fly-By-Wire", layout="wide")
    flight = get_flight(args.ckpt)
    with st.sidebar:
        st.markdown(
            f"**checkpoint**  \n`{Path(args.ckpt).parent.name}/{Path(args.ckpt).name}`"
        )
        overrides = level_overrides(st.selectbox("corridor level", LEVELS))
        if overrides != flight.overrides:
            flight.set_overrides(overrides)
        seed = int(
            st.number_input(
                "corridor seed (held-out: 10000 and up)", value=flight.seed, step=1
            )
        )
        if st.button("new flight") or seed != flight.seed:
            flight.reset(seed)
        lesions = st.multiselect(
            "silence", list(flight.model.lesion_groups), default=list(flight.lesions)
        )
        flying = st.toggle("fly", value=True)
        st.caption(
            "A lesion sets the rates of that cell type to exactly 0 from the next step on."
        )
    if tuple(lesions) != flight.lesions:
        flight.set_lesions(lesions)

    view = st.empty()
    status = st.empty()

    fig = Figure(figsize=(12, 7.2), dpi=90)
    FigureCanvasAgg(fig)
    draw(fig, flight)
    view.image(figure_rgb(fig))
    while flying and not flight.done:
        flight.step()
        draw(fig, flight)
        view.image(figure_rgb(fig))
    if flight.done:
        status.info(
            f"episode ended: {flight.reason} at {flight.time:.1f} s. Press 'new flight' or change the seed."
        )


main()
