"""EPG "compass ring": 16 wedges computed from EPG rates.

Layout modes (``heads.ring_layout``):

* ``anatomical``: parse the protocerebral-bridge glomerulus from instance names
  (:func:`~flybywire.connectome.celltypes.parse_pb_glomerulus`) and map it to a wedge
  (:func:`~flybywire.connectome.celltypes.pb_glomerulus_to_wedge`, approximate convention).
* ``tuning``: each EPG neuron gets the angle ``atan2`` of its heading-decoder weights
  ``(w_sin, w_cos)``, binned into 16 wedges.
* ``auto``: anatomical if parsing succeeds for >= 90% of EPG neurons, else tuning.

The figure always states the layout that was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybywire.connectome.celltypes import parse_pb_glomerulus, pb_glomerulus_to_wedge

LAYOUTS = ("anatomical", "tuning", "auto")


@dataclass
class RingLayout:
    wedge_of_neuron: np.ndarray  # (n_epg,) int wedge index, -1 = unassigned
    mode: str  # the mode actually used
    parsed_fraction: float  # fraction of EPG neurons parsed anatomically
    n_wedges: int = 16

    @property
    def label(self) -> str:
        return f"layout: {self.mode} ({100 * self.parsed_fraction:.0f}% names parsed)"


def anatomical_wedges(instances, n_wedges: int = 16) -> np.ndarray:
    out = np.full(len(instances), -1, dtype=int)
    for i, name in enumerate(instances):
        g = parse_pb_glomerulus(name)
        if g is not None:
            w = pb_glomerulus_to_wedge(*g, n_wedges=n_wedges)
            if w is not None:
                out[i] = w
    return out


def tuning_wedges(decoder_weight, n_wedges: int = 16) -> np.ndarray:
    """decoder_weight: (2, n_epg) rows (sin, cos) -> wedge index per EPG neuron."""
    w = np.asarray(decoder_weight.detach().cpu() if hasattr(decoder_weight, "detach") else decoder_weight)
    ang = np.arctan2(w[0], w[1]) % (2 * np.pi)
    return (np.floor(ang / (2 * np.pi) * n_wedges).astype(int)) % n_wedges


def make_layout(mode: str, epg_instances, decoder_weight=None, n_wedges: int = 16, min_parsed: float = 0.9):
    if mode not in LAYOUTS:
        raise ValueError(f"ring layout must be one of {LAYOUTS}")
    anat = anatomical_wedges(epg_instances, n_wedges)
    frac = float((anat >= 0).mean()) if len(anat) else 0.0
    if mode == "anatomical" or (mode == "auto" and frac >= min_parsed):
        return RingLayout(anat, "anatomical", frac, n_wedges)
    if decoder_weight is None:
        raise ValueError("tuning layout needs the heading decoder weights")
    return RingLayout(tuning_wedges(decoder_weight, n_wedges), "tuning", frac, n_wedges)


def ring_activity(r_epg, layout: RingLayout) -> np.ndarray:
    """r_epg: (..., n_epg) rates -> (..., n_wedges) mean rate per wedge (NaN for empty wedges)."""
    r = np.asarray(r_epg.detach().cpu() if hasattr(r_epg, "detach") else r_epg, dtype=np.float64)
    out = np.full((*r.shape[:-1], layout.n_wedges), np.nan)
    for w in range(layout.n_wedges):
        sel = layout.wedge_of_neuron == w
        if sel.any():
            out[..., w] = r[..., sel].mean(axis=-1)
    return out


def plot_ring(ax, wedge_values, layout: RingLayout, heading_rad: float | None = None, vmax: float | None = None):
    """Draw one ring frame on a polar matplotlib axis. Labels the layout in the figure."""
    import matplotlib.pyplot as plt

    n = layout.n_wedges
    vals = np.nan_to_num(np.asarray(wedge_values, dtype=float), nan=0.0)
    vmax = vmax or (vals.max() if vals.max() > 0 else 1.0)
    width = 2 * np.pi / n
    theta = np.arange(n) * width + width / 2
    colors = plt.cm.viridis(np.clip(vals / vmax, 0, 1))
    ax.bar(theta, np.ones(n), width=width * 0.95, bottom=0.6, color=colors, edgecolor="k", linewidth=0.3)
    if heading_rad is not None:
        ax.annotate("", xy=(heading_rad % (2 * np.pi), 0.55), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="r"))
    ax.set_ylim(0, 1.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(layout.label, fontsize=8)
    return ax
