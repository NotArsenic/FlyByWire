"""LPTCPool: T4/T5 activity (B, 8, 721) -> 8 self-motion channels (B, 8).

Template weights ``w[k, d, p] = f_k(p) · e_d`` (flow field of channel k at hexal p, projected on
the calibrated preferred direction of subtype d), each template normalised to unit L2 norm over
(d, p). Output ``s_k = Σ_d Σ_p g[k, d] * w[k, d, p] * T_d(p)``.

Input transform: ``T_d(p) = relu(activity) - baseline_d`` — the rectified (synaptic-output)
activity, minus its grey-steady-state level so that a static scene gives ~0 (docs/DECISIONS.md).

Modes: ``frozen`` (g = 1 fixed), ``gains`` (g learnable, 64 params, default), ``full`` (w learnable,
initialised from the templates, 8 * 8 * 721 = 46,144 params).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MODES = ("frozen", "gains", "full")


def build_templates(flows: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """flows (K, P, 2), directions (D, 2) -> unit-norm templates (K, D, P)."""
    w = np.einsum("kpc,dc->kdp", flows, directions)
    norms = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
    norms[norms == 0] = 1.0
    return w / norms[:, None, None]


class LPTCPool(nn.Module):
    def __init__(self, templates: np.ndarray | torch.Tensor, baseline=None, mode: str = "gains"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"lptc mode must be one of {MODES}")
        self.mode = mode
        w = torch.as_tensor(np.asarray(templates), dtype=torch.float32)
        k, d, _ = w.shape
        if mode == "full":
            self.w = nn.Parameter(w.clone())
        else:
            self.register_buffer("w", w.clone())
        if mode == "gains":
            self.g = nn.Parameter(torch.ones(k, d))
        else:
            self.register_buffer("g", torch.ones(k, d))
        b = torch.zeros(d) if baseline is None else torch.as_tensor(np.asarray(baseline), dtype=torch.float32)
        self.register_buffer("baseline", b.reshape(d))

    @property
    def n_channels(self) -> int:
        return int(self.w.shape[0])

    def effective_weights(self) -> torch.Tensor:
        return self.g[:, :, None] * self.w  # (K, D, P)

    def forward(self, t4t5: torch.Tensor) -> torch.Tensor:
        """t4t5: (..., D, P) raw activity -> (..., K)."""
        x = torch.relu(t4t5.float()) - self.baseline[:, None]
        return torch.einsum("...dp,kdp->...k", x, self.effective_weights())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def lptc_from_cfg(cfg, direction_map: dict | None = None) -> LPTCPool:
    """Build templates from config + calibrated direction map (loaded from disk if not given)."""
    from flybywire.vision.directions import baselines, direction_vectors, load_direction_map
    from flybywire.vision.flowfields import hexal_flow_templates

    if direction_map is not None:
        dm = direction_map
    else:
        dm = load_direction_map(cfg.lptc.get("direction_map"), cell_types=tuple(cfg.vision.cell_types))
    flows = hexal_flow_templates(
        extent=int(cfg.vision.boxeye.extent),
        kernel_size=int(cfg.vision.boxeye.kernel_size),
        fov_deg=float(cfg.env.camera_fov_deg),
        tilt_deg=float(cfg.env.camera_tilt_deg),
        channels=tuple(cfg.lptc.channels),
    )
    e = direction_vectors(dm, tuple(cfg.vision.cell_types))
    return LPTCPool(build_templates(flows, e), baseline=baselines(dm, tuple(cfg.vision.cell_types)), mode=cfg.lptc.mode)
