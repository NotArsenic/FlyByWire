"""Input projection and readout heads."""

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
        u_target = s @ self.weight + self.bias
        u = s.new_zeros(*s.shape[:-1], self.n_neurons)
        return u.index_copy(-1, self.target_index, u_target)


class HeadingDecoder(nn.Module):
    def __init__(self, source_index):
        super().__init__()
        self.register_buffer("source_index", _idx(source_index))
        self.linear = nn.Linear(len(self.source_index), 2)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.linear(r.index_select(-1, self.source_index))


class AnatomicalHeadingDecoder(nn.Module):
    """heading = gain * sum_i r_i (sin a_i, cos a_i) + bias over the EPG neurons with a reliable EB angle a_i."""

    def __init__(self, source_index, angle, reliable, gain_init: float = 0.1):
        super().__init__()

        self.register_buffer("source_index", _idx(source_index))
        angle = torch.as_tensor(np.asarray(angle), dtype=torch.float32)
        keep = torch.as_tensor(np.asarray(reliable), dtype=torch.float32)
        self.register_buffer("unit", torch.stack([angle.sin(), angle.cos()]) * keep)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.bias = nn.Parameter(torch.zeros(2))

    @property
    def weight(self) -> torch.Tensor:
        return self.gain * self.unit

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return r.index_select(-1, self.source_index) @ self.weight.T + self.bias


class Actor(nn.Module):
    def __init__(self, source_index, n_actions: int = 4, log_std_init: float = -1.5):
        super().__init__()
        self.register_buffer("source_index", _idx(source_index))
        self.linear = nn.Linear(len(self.source_index), n_actions)
        self.log_std = nn.Parameter(torch.full((n_actions,), float(log_std_init)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(r.index_select(-1, self.source_index)))

    def distribution(self, r: torch.Tensor) -> torch.distributions.Normal:
        mean = self(r)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, n_neurons: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_neurons, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r.detach()).squeeze(-1)
