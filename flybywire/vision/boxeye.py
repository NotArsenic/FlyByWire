"""BoxEye wrapper: grayscale frames -> hexal photoreceptor inputs.

Wraps ``flyvis.datasets.rendering.eye.BoxEye`` (verified in the installed source):

* receptor centres (pixels, relative to the frame centre) are ``y = d (u + v/2)``, ``x = d v`` for
  hex coordinates ``(u, v)`` from ``flyvis.utils.hex_utils.get_hex_coords(extent)`` in that order,
  with ``d = kernel_size``;
* frames smaller than ``min_frame_size`` (391 x 391 for extent 15 / kernel 13) are bilinearly resized
  up to it, box-filtered with a ``kernel_size`` mean filter, then sampled at the receptor centres.

Input ``(B, H, W)`` uint8 in [0, 255] or float in [0, 1]; output ``(B, 721)`` float32 in [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch

from flybywire.paths import import_flyvis


class BoxEye:
    def __init__(self, extent: int = 15, kernel_size: int = 13, device: str | torch.device = "cpu"):
        import_flyvis()
        from flyvis.datasets.rendering.eye import BoxEye as _FlyvisBoxEye

        self.device = torch.device(device)
        with torch.device(self.device):
            self._eye = _FlyvisBoxEye(extent=extent, kernel_size=kernel_size)
        self.extent = extent
        self.kernel_size = kernel_size
        self.n_hexals = int(self._eye.hexals)
        self.min_frame_size = tuple(int(x) for x in self._eye.min_frame_size.tolist())

    @torch.no_grad()
    def __call__(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(frames, device=self.device)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3:
            raise ValueError(f"expected (B, H, W) frames, got {tuple(x.shape)}")
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        with torch.device(self.device):
            out = self._eye(x[:, None], ftype="mean", hex_sample=True)
        return out.reshape(x.shape[0], self.n_hexals).float()

    def receptor_centers(self) -> np.ndarray:
        """(hexals, 2) integer (row, col) pixel offsets from the centre of the resized frame."""
        return self._eye.receptor_centers.cpu().numpy()
