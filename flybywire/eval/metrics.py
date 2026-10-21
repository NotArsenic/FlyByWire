"""Heading-decode metrics."""

from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def heading_r2(pred, target, mask=None) -> float:
    p = _np(pred).reshape(-1, 2)
    t = _np(target).reshape(-1, 2)
    if mask is not None:
        m = _np(mask).reshape(-1).astype(bool)
        p, t = p[m], t[m]
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def angular_error_deg(pred, target, mask=None) -> float:
    p = _np(pred).reshape(-1, 2)
    t = _np(target).reshape(-1, 2)
    if mask is not None:
        m = _np(mask).reshape(-1).astype(bool)
        p, t = p[m], t[m]
    d = np.arctan2(p[:, 0], p[:, 1]) - np.arctan2(t[:, 0], t[:, 1])
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(np.abs(d)).mean())


def lagged_heading_r2(preds, targets, lags) -> list[float]:
    """R^2 of the decoded heading at t against the true heading at t - lag, one value per lag.

    preds and targets are lists of per-episode (T, 2) arrays.
    A decoder that sees the heading late scores best at a positive lag.
    """
    return [
        heading_r2(
            np.concatenate([p[lag:] for p in preds]),
            np.concatenate([t[: len(t) - lag] for t in targets]),
        )
        for lag in lags
    ]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% by default)."""
    p = k / n
    scale = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / scale
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return float(centre - half), float(centre + half)
