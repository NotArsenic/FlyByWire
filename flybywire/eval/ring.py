"""EPG "compass ring": 16 wedges computed from EPG rates.

Layout modes (``heads.ring_layout``):

* ``anatomical``: each EPG neuron with a reliable angle around the ellipsoid body (EB angles from
  ``scripts/epg_anatomy.py``) goes into the wedge that angle falls in; unreliable neurons get none.
* ``tuning``: each EPG neuron gets the angle ``atan2`` of its heading-decoder weights
  ``(w_sin, w_cos)``. The decoder is trained to read heading out of these neurons, so a bump
  shows up on this ring by construction. Use it as a reference, not as evidence.
* ``auto``: anatomical if the angles file exists, else tuning.

The figure always states the layout that was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybywire.connectome.masks import eb_angles_path, load_base_connectome, load_eb_angles
from flybywire.eval.topography import alignment

LAYOUTS = ("anatomical", "tuning", "auto")


@dataclass
class RingLayout:
    wedge_of_neuron: np.ndarray
    mode: str
    n_wedges: int = 16

    @property
    def label(self) -> str:
        placed = int((self.wedge_of_neuron >= 0).sum())
        return f"layout: {self.mode} ({placed} of {len(self.wedge_of_neuron)} EPG neurons placed)"


def _wedge(angle, n_wedges: int) -> np.ndarray:
    return np.floor((np.asarray(angle) % (2 * np.pi)) / (2 * np.pi) * n_wedges).astype(int) % n_wedges


def anatomical_wedges(angle, reliable, n_wedges: int = 16) -> np.ndarray:
    return np.where(reliable, _wedge(angle, n_wedges), -1)


def tuning_wedges(decoder_weight, n_wedges: int = 16) -> np.ndarray:
    """Wedge index per EPG neuron from heading decoder weights (-1 for zero weights)."""
    w = np.asarray(decoder_weight.detach().cpu() if hasattr(decoder_weight, "detach") else decoder_weight)
    return np.where(np.hypot(w[0], w[1]) > 0, _wedge(np.arctan2(w[0], w[1]), n_wedges), -1)


def make_layout(mode: str, eb_angles=None, decoder_weight=None, n_wedges: int = 16) -> RingLayout:
    if mode not in LAYOUTS:
        raise ValueError(f"ring layout must be one of {LAYOUTS}")
    if mode == "anatomical" or (mode == "auto" and eb_angles is not None):
        if eb_angles is None:
            raise ValueError("anatomical layout needs the EB angles (scripts/epg_anatomy.py)")
        return RingLayout(anatomical_wedges(*eb_angles, n_wedges), "anatomical", n_wedges)
    if decoder_weight is None:
        raise ValueError("tuning layout needs the heading decoder weights")
    return RingLayout(tuning_wedges(decoder_weight, n_wedges), "tuning", n_wedges)


def model_layout(cfg, model, mode: str | None = None) -> RingLayout:
    """Layout for a trained model; ``mode`` defaults to ``heads.ring_layout``."""
    path = eb_angles_path(cfg)
    eb = load_eb_angles(cfg, load_base_connectome(cfg)) if path.exists() else None
    return make_layout(mode or cfg.heads.ring_layout, eb, model.heading.weight.detach(), int(cfg.heads.ring_wedges))


def ring_activity(r_epg, layout: RingLayout) -> np.ndarray:
    """Mean rate per wedge from EPG rates (NaN for empty wedges)."""
    r = np.asarray(r_epg.detach().cpu() if hasattr(r_epg, "detach") else r_epg, dtype=np.float64)
    out = np.full((*r.shape[:-1], layout.n_wedges), np.nan)
    for w in range(layout.n_wedges):
        sel = layout.wedge_of_neuron == w
        if sel.any():
            out[..., w] = r[..., sel].mean(axis=-1)
    return out


def rate_floor(rates, washout: int = 0, q: float = 5.0) -> np.ndarray:
    """Resting rate per neuron: the q-th percentile of its rate over all frames after the washout.

    rates: one (T, n_epg) array per episode. Subtract it before ``ring_activity`` to see whether
    a neuron's baseline, rather than the bump, sets the share.
    """
    return np.percentile(np.concatenate([r[washout:] for r in rates]), q, axis=0)


def bump(profile) -> dict[str, np.ndarray]:
    """Single-bump summary of ring profiles (..., n_wedges), from the first Fourier component.

    ``angle`` is the population-vector angle, wedge w centred at (w + 0.5) 2 pi / n_wedges.
    ``size`` is the length of that vector (same unit as the profile, not normalised).
    ``fraction`` is the share of the power in harmonics 1..n/2 that sits in harmonic 1: 1 for a
    pure cosine, about 1/8 for noise, 0 for a flat profile. The mean (harmonic 0) is left out of
    the denominator, so adding the same rate to every wedge doesn't change it.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = np.where(np.isnan(p), np.nanmean(p, axis=-1, keepdims=True), p)  # an empty wedge adds no bump
    n = p.shape[-1]
    z = p @ np.exp(1j * (np.arange(n) + 0.5) * 2 * np.pi / n)
    power = np.abs(np.fft.rfft(p, axis=-1)[..., 1:]) ** 2
    total = power.sum(-1)
    fraction = np.where(total > 1e-12, power[..., 0] / np.maximum(total, 1e-12), 0.0)
    return {"angle": np.angle(z), "size": np.abs(z), "fraction": fraction}


def ring_tracking(bumps: list[dict], heading: list[np.ndarray], washout: int = 0) -> dict:
    """How well the bump follows the true heading; one ``bump`` dict and one (T,) heading (rad) per episode.

    ``alignment`` is |mean exp(i (bump angle - s heading))| maximised over s = +/-1, so it is 1 when
    the bump sits at s * heading + c for any fixed c, and ignores the ring's arbitrary zero and
    handedness. ``alignment_null`` pairs each episode's bump with the next episode's heading.
    """
    angle = [b["angle"][washout:] for b in bumps]
    head = [h[washout:] for h in heading]
    score, sign, offset = alignment(np.concatenate(angle), np.concatenate(head))
    pairs = [(a, head[(i + 1) % len(head)]) for i, a in enumerate(angle)]
    m = [min(len(a), len(h)) for a, h in pairs]
    null = alignment(np.concatenate([a[:k] for (a, _), k in zip(pairs, m, strict=True)]),
                     np.concatenate([h[:k] for (_, h), k in zip(pairs, m, strict=True)]))[0]
    fraction = np.concatenate([b["fraction"][washout:] for b in bumps])
    return {
        "n_frames": int(len(fraction)),
        "fraction_mean": float(fraction.mean()),
        "fraction_median": float(np.median(fraction)),
        "alignment": score,
        "alignment_null": float(null),
        "sign": sign,
        "offset_deg": float(np.degrees(offset)),
    }


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
