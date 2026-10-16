"""Input projection and readout heads.

* :class:`InputProjection` — (B, 8) LPTC channels -> (B, N); nonzero only on ER neurons (inv. 3).
* :class:`HeadingDecoder` — Linear(|EPG| -> 2), (sin, cos) of heading relative to episode start.
  Auxiliary only; never feeds the actor. Reads only EPG rates (inv. 4).
* :class:`Actor` — Linear(|PFL3| -> 4) + tanh mean, state-independent ``log_std`` (init -1.5).
  Linear on purpose: "steering is a linear readout of PFL3". Reads only PFL3 rates (inv. 4).
* :class:`Critic` — small MLP on ``r.detach()``; never backpropagates into the core (inv. 4).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _idx(index) -> torch.Tensor:
    return torch.as_tensor(np.asarray(index), dtype=torch.long)


class InputProjection(nn.Module):
    def __init__(self, n_in: int, n_neurons: int, target_index, init_std: float = 0.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.register_buffer("target_index", _idx(target_index))
        k = len(self.target_index)
        self.weight = nn.Parameter(torch.randn(n_in, k) * init_std)
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        u_target = s @ self.weight + self.bias  # (B, |ER|)
        u = s.new_zeros(*s.shape[:-1], self.n_neurons)
        return u.index_copy(-1, self.target_index, u_target)


class HeadingDecoder(nn.Module):
    def __init__(self, source_index):
        super().__init__()
        self.register_buffer("source_index", _idx(source_index))
        self.linear = nn.Linear(len(self.source_index), 2)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.linear(r.index_select(-1, self.source_index))


class Actor(nn.Module):
    def __init__(self, source_index, n_actions: int = 4, log_std_init: float = -1.5):
        super().__init__()
        self.register_buffer("source_index", _idx(source_index))
        self.linear = nn.Linear(len(self.source_index), n_actions)
        self.log_std = nn.Parameter(torch.full((n_actions,), float(log_std_init)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Action mean in [-1, 1]."""
        return torch.tanh(self.linear(r.index_select(-1, self.source_index)))

    def distribution(self, r: torch.Tensor) -> torch.distributions.Normal:
        mean = self(r)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, n_neurons: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_neurons, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r.detach()).squeeze(-1)
