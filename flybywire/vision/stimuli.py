"""Synthetic visual stimuli for plumbing tests, Gate 1 and direction calibration.

* :func:`panorama_texture` + :func:`view_from_panorama` — a cylindrical random texture viewed
  through a ``fov_deg`` window; rotating the viewer by ``yaw`` shifts the texture horizontally.
  Positive yaw (turning left, body +z) moves the scene **rightward** in the image, matching the
  sign of the ``yaw`` flow template in :mod:`flybywire.vision.flowfields`.
* :func:`square_grating` — drifting square-wave grating moving along +x / -x / +y / -y (image
  coordinates, x right, y down).

All frames are float32 in [0, 1], shape (H, W).
"""

from __future__ import annotations

import cv2
import numpy as np

DIRECTIONS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


def px_per_rad(size: int, fov_deg: float) -> float:
    return size / np.deg2rad(fov_deg)


def panorama_texture(size: int = 160, fov_deg: float = 110.0, seed: int = 0, sigma_px: float = 3.0) -> np.ndarray:
    """(size, W360) float32 [0, 1] random texture covering 360° horizontally (wraps)."""
    rng = np.random.default_rng(seed)
    width = int(round(2 * np.pi * px_per_rad(size, fov_deg)))
    noise = rng.random((size, width)).astype(np.float32)
    # wrap-aware blur: pad horizontally, blur, crop
    pad = int(4 * sigma_px) + 1
    padded = np.concatenate([noise[:, -pad:], noise, noise[:, :pad]], axis=1)
    blurred = cv2.GaussianBlur(padded, (0, 0), sigma_px)[:, pad:-pad]
    blurred = (blurred - blurred.mean()) / (blurred.std() + 1e-8)
    return np.clip(0.5 + 0.2 * blurred, 0.0, 1.0).astype(np.float32)


def view_from_panorama(pano: np.ndarray, yaw: float, size: int = 160, fov_deg: float = 110.0) -> np.ndarray:
    """(size, size) window into the panorama at heading ``yaw`` (rad), sub-pixel interpolated."""
    h, w = pano.shape
    offset = -yaw * px_per_rad(size, fov_deg)  # yaw left -> window moves left -> scene moves right
    cols = (offset + np.arange(size)) % w
    c0 = np.floor(cols).astype(int)
    frac = (cols - c0).astype(np.float32)
    c1 = (c0 + 1) % w
    r0 = (h - size) // 2
    rows = pano[r0 : r0 + size]
    return rows[:, c0] * (1 - frac) + rows[:, c1] * frac


def square_grating(
    size: int, period_px: float, shift_px: float, direction: str | float, contrast: float = 1.0
) -> np.ndarray:
    """Square-wave grating whose stripes move by ``shift_px`` along ``direction``.

    ``direction`` is one of +x, -x, +y, -y or an angle in radians (``atan2(dy, dx)``, image coords).
    """
    if isinstance(direction, str):
        dx, dy = DIRECTIONS[direction]
    else:
        dx, dy = float(np.cos(direction)), float(np.sin(direction))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    pos = xx * dx + yy * dy  # coordinate along the motion axis
    phase = 2 * np.pi * (pos - shift_px) / period_px
    return (0.5 + 0.5 * contrast * np.sign(np.sin(phase))).astype(np.float32)


def grating_sequence(
    size: int, period_px: float, temporal_hz: float, n_frames: int, frame_hz: float, direction: str | float
) -> np.ndarray:
    """(n_frames, size, size) drifting grating."""
    speed = period_px * temporal_hz / frame_hz  # px per frame
    return np.stack([square_grating(size, period_px, speed * t, direction) for t in range(n_frames)])
