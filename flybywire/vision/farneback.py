"""Farneback fallback vision backend (``vision.backend: farneback``).

Computes dense optical flow between consecutive frames with OpenCV's Farneback method and projects
it onto the same 8 self-motion flow templates used by the LPTC layer, evaluated on the pixel grid:
``s_k = <f̂_k, flow>`` with ``f̂_k`` the unit-L2-norm template. Output is ``(B, 8)`` directly, so
everything downstream (InputProjection, core, heads) is unchanged. Stateful per env (previous
frame); there is nothing to learn and no gradient.

Flow is converted to focal-length units per second so its scale matches the templates:
``flow_focal = flow_px / (size / 2) * tan(fov / 2) * sensory_hz``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from flybywire.vision.flowfields import CHANNELS, flow_templates, pixel_grid_focal

FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


class FarnebackBackend:
    def __init__(
        self,
        frame_size: int = 160,
        fov_deg: float = 110.0,
        tilt_deg: float = 0.0,
        sensory_hz: int = 100,
        channels: Sequence[str] = CHANNELS,
        downsample: int = 1,
    ):
        self.size = frame_size // downsample
        self.downsample = downsample
        self.scale = np.tan(np.deg2rad(fov_deg) / 2.0) / (self.size / 2.0) * sensory_hz
        xy = pixel_grid_focal(self.size, fov_deg)
        t = flow_templates(xy, tuple(channels), tilt_deg)  # (K, P, 2)
        t = t.reshape(t.shape[0], -1)
        norms = np.linalg.norm(t, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.templates = (t / norms).astype(np.float32)  # (K, P*2)
        self.prev: list[np.ndarray | None] = []

    def reset(self, batch_size: int | None = None, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.prev = [None] * int(batch_size or len(self.prev))
        else:
            for i in env_ids:
                self.prev[int(i)] = None

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        f = np.asarray(frame)
        if f.dtype != np.uint8:
            f = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        if self.downsample > 1:
            f = cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return f

    def step(self, frames) -> np.ndarray:
        """frames: (B, H, W) uint8 or float [0, 1] -> (B, 8) float32. First frame after reset gives 0."""
        frames = np.asarray(frames.cpu() if hasattr(frames, "cpu") else frames)
        if len(self.prev) != frames.shape[0]:
            self.reset(frames.shape[0])
        out = np.zeros((frames.shape[0], self.templates.shape[0]), dtype=np.float32)
        for i, fr in enumerate(frames):
            cur = self._prep(fr)
            if self.prev[i] is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev[i], cur, None, **FARNEBACK_PARAMS)
                out[i] = self.templates @ (flow.reshape(-1).astype(np.float32) * self.scale)
            self.prev[i] = cur
        return out
