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
from scipy.stats import binomtest

from flybywire.clocks import clocks_from_cfg
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
def recurrence_scaled(core, scale: float):
    saved = core.W.data.clone()
    core.W.data.mul_(scale)
    try:
        yield
    finally:
        core.W.data.copy_(saved)


def recurrence_off(core):
    return recurrence_scaled(core, 0.0)


def _stepper(cfg, device, steady_state_seconds=None):
    from flybywire.vision.flyvis_stepper import FlyvisStepper

    return FlyvisStepper(
        model=cfg.vision.flyvis_model,
        device=device,
        dt=clocks_from_cfg(cfg).dt_sensory,
        cell_types=tuple(cfg.vision.cell_types),
        steady_state_seconds=float(steady_state_seconds or cfg.vision.steady_state_seconds),
    )


def grey_t4t5(cfg, device, steady_state_seconds=None) -> torch.Tensor:
    """(8, 721) T4/T5 activity of the flyvis grey steady state (``vision.steady_state_seconds`` unless given)."""
    stepper = _stepper(cfg, device, steady_state_seconds)
    return stepper.readout(stepper.steady_state(1))[0].half().float()


def static_t4t5(cfg, device, hexals, seconds: float = 1.5, chunk: int = 100) -> np.ndarray:
    """(n, 8, 721) T4/T5 activity after each frame of hexals (n, 721) is held still for ``seconds``, from grey."""
    stepper = _stepper(cfg, device)
    n_steps = int(round(seconds / stepper.dt))
    out = []
    for i in range(0, len(hexals), chunk):
        x = torch.as_tensor(np.asarray(hexals[i : i + chunk]), dtype=torch.float32, device=device)
        stepper.reset(len(x))
        for _ in range(n_steps):
            t4t5 = stepper.step(x)
        out.append(t4t5.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def hold_branch(model, v, r, s, n_steps: int, recurrent: bool = True) -> torch.Tensor:
    """Step the core n_steps from state v with the LPTC channels held at s, (8,) or one row per sample.

    Returns (B, n_steps + 1, N) rates, index 0 = r.
    """
    u = model.inproj(s if s.ndim == 2 else s[None]).expand(len(v), -1)
    rs = [r]
    with nullcontext() if recurrent else recurrence_off(model.core):
        for _ in range(n_steps):
            v, r = model.core(v, u)
            rs.append(r)
    return torch.stack(rs, dim=1)


def dark_branch(model, v, r, blank, n_steps: int, recurrent: bool = True) -> torch.Tensor:
    """hold_branch on the LPTC channels of the T4/T5 pattern ``blank`` (8, 721)."""
    with torch.no_grad():
        s = model.sensory(blank[None])[0]
    return hold_branch(model, v, r, s, n_steps, recurrent)


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


def signed_stats(drift, episode, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Which way the drift goes. ``drift``: signed drift per trial (deg); ``episode``: episode id per trial.

    Trials from one episode share a scene and are strongly correlated, so the sign test and the
    bootstrap interval work on episodes, not trials. ``mean_over_mean_abs`` is 1 in magnitude when
    every trial goes the same way and near 0 when the signs cancel.
    """
    d = np.asarray(drift, dtype=np.float64)
    ep = np.unique(episode, return_inverse=True)[1]
    total, count = np.bincount(ep, weights=d), np.bincount(ep)
    per_ep = total[count > 0] / count[count > 0]
    n_neg, n_pos = int((per_ep < 0).sum()), int((per_ep > 0).sum())
    pick = np.random.default_rng(seed).integers(0, len(total), (n_boot, len(total)))
    boot = total[pick].sum(1) / count[pick].sum(1)
    return {
        "n_trials": len(d),
        "mean": float(d.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "median": float(np.median(d)),
        "mean_abs": float(np.abs(d).mean()),
        "mean_over_mean_abs": float(d.mean() / np.abs(d).mean()),
        "frac_negative": float((d < 0).mean()),
        "episodes_negative": n_neg,
        "episodes_positive": n_pos,
        "sign_test_p": float(binomtest(n_neg, n_neg + n_pos).pvalue) if n_neg + n_pos else 1.0,
    }


def yaw_calibration(s_yaw, angle, yaw_rate, skip=100, win=25, dt=0.01):
    """Yaw-rate gain and zero-turn level of the yaw channel, from 0.25 s windows of the real episodes."""
    starts = np.arange(skip, s_yaw.shape[1] - win, win)
    chan = np.stack([s_yaw[:, a : a + win].mean(1) for a in starts], 1).ravel()
    true = np.stack([yaw_rate[:, a : a + win].mean(1) for a in starts], 1).ravel()
    dec = np.stack([wrap(angle[:, a + win] - angle[:, a]) for a in starts], 1).ravel() / (win * dt)
    ok = np.abs(dec) < 6
    out = {}
    for name, rate, keep in (("decoded", dec, ok), ("true", true, np.ones_like(ok))):
        gain, icpt = np.polyfit(chan[keep], rate[keep], 1)
        out[name] = {
            "gain_rad_s_per_unit": float(gain),
            "zero_turn_level": float(-icpt / gain),
            "corr": float(np.corrcoef(chan[keep], rate[keep])[0, 1]),
            "n_windows": int(keep.sum()),
        }
    return out


def velocity_field(drift_1s, start, width=20.0, min_n=15):
    """Median drift over the first second per bin of the heading it starts from, and its zero crossings.

    A crossing from positive to negative drift is a heading the decoded angle settles at (stable);
    from negative to positive, one it moves away from.
    """
    deg = np.degrees(wrap(start))
    edges = np.arange(-180, 180 + width, width)
    which = np.digitize(deg, edges) - 1
    centre, median, n = [], [], []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() >= min_n:
            centre.append(float(edges[b] + width / 2))
            median.append(float(np.median(drift_1s[m])))
            n.append(int(m.sum()))
    c, v = np.array(centre), np.array(median)
    adjacent = np.diff(c) <= width * 1.01
    cross = (np.sign(v[:-1]) != np.sign(v[1:])) & adjacent
    at = c[:-1] + (c[1:] - c[:-1]) * v[:-1] / (v[:-1] - v[1:] + 1e-12)
    return {"centre_deg": centre, "median_drift_1s_deg": median, "n": n,
            "stable_deg": at[cross & (v[:-1] > v[1:])].round(1).tolist(),
            "unstable_deg": at[cross & (v[:-1] < v[1:])].round(1).tolist()}
