"""CentralComplex: one recurrent rate population masked by the connectome.

    r_t     = phi(v_t) * keep                  # keep: (N,) 0/1 lesion mask
    drive_t = r_t @ (W * M) + u_t + b          # M[i, j] = 1 iff pre i -> post j
    v_{t+1} = (1 - alpha) * v_t + alpha * drive_t
    alpha   = sigmoid(a)                       # per neuron; optionally sigmoid(a + c * u_t)

The ONLY neuron-to-neuron path is ``r @ (W * M)`` (invariant 1): no residual stream, no
LayerNorm, no attention, no dense layer over the neuron axis. The optional input-dependent
alpha uses an elementwise gain ``c * u_t`` (per neuron, its own input only) so it adds no
cross-neuron interaction. ``W`` is a dense (N, N) parameter times a registered bool buffer ``M``;
gradients w.r.t. ``W`` are exactly zero off-mask (invariant 2). fp32 only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PHIS = {"softplus": F.softplus, "relu": F.relu}


class CentralComplex(nn.Module):
    def __init__(
        self,
        M: np.ndarray | torch.Tensor,
        alpha_init: float = 0.2,
        w_init_gain: float = 0.5,
        phi: str = "softplus",
        input_dependent_alpha: bool = False,
        generator: torch.Generator | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        M = torch.as_tensor(np.asarray(M), dtype=torch.bool)
        n = M.shape[0]
        if M.shape != (n, n):
            raise ValueError("M must be square")
        if phi not in PHIS:
            raise ValueError(f"phi must be one of {list(PHIS)}")
        self.n = n
        self.phi_name = phi
        self.input_dependent_alpha = input_dependent_alpha
        self.register_buffer("M", M)
        self.register_buffer("keep", torch.ones(n))

        mean_in_degree = max(float(M.sum()) / n, 1.0)
        std = w_init_gain / math.sqrt(mean_in_degree)
        W = torch.randn(n, n, generator=generator) * std
        self.W = nn.Parameter(W * M)  # off-mask entries start at 0 and never receive gradient
        self.b = nn.Parameter(torch.zeros(n))
        self.a = nn.Parameter(torch.full((n,), math.log(alpha_init / (1 - alpha_init))))
        if input_dependent_alpha:
            self.c = nn.Parameter(torch.zeros(n))
        self.w_init_std = std
        if verbose:
            print(
                f"CentralComplex: N={n} edges={int(M.sum())} mean_in_degree={mean_in_degree:.2f} "
                f"W_std={std:.4f} spectral_radius(W*M)={self.spectral_radius():.4f}"
            )

    def phi(self, v: torch.Tensor) -> torch.Tensor:
        return PHIS[self.phi_name](v)

    def effective_weight(self) -> torch.Tensor:
        return self.W * self.M

    @torch.no_grad()
    def spectral_radius(self) -> float:
        ev = torch.linalg.eigvals(self.effective_weight().detach().double().cpu())
        return float(ev.abs().max()) if ev.numel() else 0.0

    def alpha(self, u: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_dependent_alpha and u is not None:
            return torch.sigmoid(self.a + self.c * u)
        return torch.sigmoid(self.a)

    def rates(self, v: torch.Tensor) -> torch.Tensor:
        return self.phi(v) * self.keep

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.n, device=device if device is not None else self.W.device)

    def forward(self, v: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rates(v)
        drive = r @ self.effective_weight() + u + self.b
        alpha = self.alpha(u)
        v_next = (1 - alpha) * v + alpha * drive
        return v_next, self.rates(v_next)
