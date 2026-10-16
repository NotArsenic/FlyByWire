"""Heading-decode metrics.

* :func:`heading_r2` — R² computed **jointly** on (sin, cos):
  ``1 - Σ‖pred - target‖² / Σ‖target - mean(target)‖²`` over all samples and both components.
* :func:`angular_error_deg` — mean absolute wrapped angular error in degrees between
  ``atan2(sin, cos)`` of prediction and target.
"""

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
