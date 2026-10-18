"""Does the heading code outlast the motion input? After Seelig and Jayaraman (2015).

The core runs on cached T4/T5 up to t0. From there the input is the flyvis grey steady-state
T4/T5 (no motion) and the core keeps stepping. The network never saw this during training, so
whatever persistence shows up is not something it was asked to do. The same branch with the
recurrent weights zeroed is the reference for a network that only decays.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from flybywire.eval.ring import RingLayout, bump, ring_activity
from flybywire.eval.topography import wrap


@torch.no_grad()
def snapshots(model, x_real, t0s) -> dict:
    """Core state and rates after the first t0 steps of x_real (B, T, 8, 721), for each t0."""
    u = model.inproj(model.sensory(x_real))
    v = model.init_state(len(x_real))
    out = {}
    for t in range(max(t0s)):
        v, r = model.core(v, u[:, t])
        if t + 1 in t0s:
            out[t + 1] = (v, r)
    return out


@contextmanager
def recurrence_off(core):
    saved = core.W.data.clone()
    core.W.data.zero_()
    try:
        yield
    finally:
        core.W.data.copy_(saved)


@torch.no_grad()
def dark_branch(model, v, r, blank, n_steps: int, recurrent: bool = True) -> torch.Tensor:
    """Step the core n_steps on the no-motion input from state v. Returns (B, n_steps + 1, N) rates, index 0 = r."""
    u = model.inproj(model.sensory(blank[None])).expand(len(v), -1)
    rs = [r]
    with nullcontext() if recurrent else recurrence_off(model.core):
        for _ in range(n_steps):
            v, r = model.core(v, u)
            rs.append(r)
    return torch.stack(rs, dim=1)


def dark_series(model, r: torch.Tensor, layout: RingLayout) -> dict[str, np.ndarray]:
    """Per-sample time series, each (B, n_steps + 1), from rates r (B, n_steps + 1, N).

    ``heading_drift`` and ``ring_drift`` are the change in the decoded heading and in the ring
    bump angle since step 0 (rad); ``decoder_size`` is the length of the decoder output without
    its bias; ``ring_size`` and ``ring_fraction`` are the bump length and first-harmonic share.
    """
    with torch.no_grad():
        head = model.heading(r)
        dec = head - model.heading.bias
        epg = r.index_select(-1, model.heading.source_index)
    head, dec = head.cpu().numpy(), dec.cpu().numpy()
    angle = np.arctan2(head[..., 0], head[..., 1])
    b = bump(ring_activity(epg, layout))
    return {
        "heading_drift": wrap(angle - angle[:, :1]),
        "decoder_size": np.hypot(dec[..., 0], dec[..., 1]),
        "ring_drift": wrap(b["angle"] - b["angle"][:, :1]),
        "ring_size": b["size"],
        "ring_fraction": b["fraction"],
    }


def dark_stats(series: dict, steps, ref: dict | None = None) -> dict[str, list[float]]:
    """Statistics over samples at each step in ``steps`` after t0.

    Drifts are |angle change| in degrees. ``*_retention`` is the mean size over the mean size at
    step 0 of ``ref`` (default: the same series).
    """
    ref = ref or series
    out = {}
    for name in ("heading_drift", "ring_drift"):
        d = np.degrees(np.abs(series[name][:, steps]))
        out[f"{name}_deg_mean"] = d.mean(0)
        out[f"{name}_deg_median"] = np.median(d, 0)
        out[f"{name}_deg_q25"], out[f"{name}_deg_q75"] = np.quantile(d, [0.25, 0.75], axis=0)
    for name in ("decoder", "ring"):
        out[f"{name}_retention"] = series[f"{name}_size"][:, steps].mean(0) / ref[f"{name}_size"][:, 0].mean()
    out["ring_fraction"] = series["ring_fraction"][:, steps].mean(0)
    return {k: np.asarray(v).tolist() for k, v in out.items()}
