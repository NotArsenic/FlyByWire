"""Stateful, frozen flyvis stepper with per-env state reset."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from flybywire.paths import import_flyvis

DEFAULT_CELL_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


class FlyvisStepper:
    def __init__(
        self,
        model: str = "flow/0000/000",
        device: str | torch.device = "cuda",
        dt: float = 0.01,
        cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
        steady_state_seconds: float = 0.5,
        grey_value: float = 0.5,
        mode: str = "fast",
    ):
        if mode not in ("fast", "public"):
            raise ValueError("mode must be 'fast' or 'public'")
        flyvis = import_flyvis()
        self.device = torch.device(device)
        self.dt = float(dt)
        self.mode = mode
        self.model_id = model
        self.cell_types = tuple(cell_types)
        self.steady_state_seconds = float(steady_state_seconds)
        self.grey_value = float(grey_value)

        with torch.device(self.device):
            nv = flyvis.NetworkView(model)
            net = nv.init_network()
            net = net.to(self.device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.n_nodes = int(net.n_nodes)

        c = net.connectome
        self._input_index = torch.as_tensor(
            net.stimulus.input_index, device=self.device
        )
        self.n_hexals = int(self._input_index.shape[1])
        idx = [
            torch.as_tensor(c.nodes.layer_index[t][:], device=self.device)
            for t in self.cell_types
        ]
        if any(len(i) != self.n_hexals for i in idx):
            raise RuntimeError(
                "T4/T5 cell counts differ from the number of input hexals"
            )
        self._out_index = torch.stack(idx)

        with torch.no_grad(), torch.device(self.device):
            net.clamp()
            self._params = net._param_api()
            ss = net.steady_state(
                self.steady_state_seconds, self.dt, batch_size=1, value=self.grey_value
            )
            if len(ss.edges):
                raise RuntimeError(
                    "flyvis dynamics carry edge state; stepper assumes node-only state"
                )
            self._steady = ss.nodes.activity[0].detach().clone()
        self.state: torch.Tensor | None = None
        from flyvis.utils.tensor_utils import AutoDeref

        self._AutoDeref = AutoDeref

    @property
    def batch_size(self) -> int:
        return 0 if self.state is None else int(self.state.shape[0])

    def steady_state(self, batch_size: int) -> torch.Tensor:
        return self._steady.expand(batch_size, -1).clone()

    def reset(
        self,
        batch_size: int | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Reset all envs (``env_ids=None``) or only ``env_ids`` to the cached grey steady state."""
        if env_ids is None:
            b = batch_size if batch_size is not None else self.batch_size
            if b <= 0:
                raise ValueError("batch_size required for the first reset")
            self.state = self.steady_state(b)
            return
        if self.state is None:
            raise RuntimeError(
                "call reset(batch_size) before resetting individual envs"
            )
        self.state[torch.as_tensor(env_ids, device=self.device, dtype=torch.long)] = (
            self._steady
        )

    def readout(self, activity: torch.Tensor) -> torch.Tensor:
        """(B, n_nodes) -> (B, 8, 721) T4/T5 activity."""
        return activity[:, self._out_index]

    def _stimulus(self, hexals: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(hexals.shape[0], self.n_nodes, device=self.device)
        x[:, self._input_index] = hexals[:, None, :]
        return x

    @torch.no_grad()
    def step(self, hexals: torch.Tensor) -> torch.Tensor:
        """Advance one ``dt`` with a fresh frame. hexals: (B, 721) in [0, 1]. Returns (B, 8, 721)."""
        hexals = torch.as_tensor(hexals, device=self.device).float()
        if self.state is None or self.state.shape[0] != hexals.shape[0]:
            self.reset(hexals.shape[0])
        net = self.net
        with torch.device(self.device):
            x_t = self._stimulus(hexals)
            st = net._state_api(
                self._AutoDeref(
                    nodes=self._AutoDeref(activity=self.state), edges=self._AutoDeref()
                )
            )
            if self.mode == "fast":
                nxt = net._next_state(self._params, st, x_t, self.dt)
            else:
                nxt = net.forward(x_t[:, None], self.dt, state=st, as_states=True)[-1]
            self.state = nxt.nodes.activity
        return self.readout(self.state)

    @torch.no_grad()
    def run_sequence(self, hexals: torch.Tensor, reset: bool = True) -> torch.Tensor:
        """Step through (B, T, 721); returns (B, T, 8, 721). Never stores full node activity."""
        hexals = torch.as_tensor(hexals, device=self.device).float()
        if reset:
            self.reset(hexals.shape[0])
        return torch.stack(
            [self.step(hexals[:, t]) for t in range(hexals.shape[1])], dim=1
        )

    @torch.no_grad()
    def full_sequence_reference(self, hexals: torch.Tensor) -> torch.Tensor:
        """One Network.forward call over the whole sequence from the grey steady state."""
        hexals = torch.as_tensor(hexals, device=self.device).float()
        b, t, _ = hexals.shape
        AutoDeref = self._AutoDeref
        with torch.device(self.device):
            x = torch.zeros(b, t, self.n_nodes, device=self.device)
            x[:, :, self._input_index] = hexals[:, :, None, :]
            st = self.net._state_api(
                AutoDeref(
                    nodes=AutoDeref(activity=self.steady_state(b)), edges=AutoDeref()
                )
            )
            act = self.net.forward(x, self.dt, state=st, as_states=False)
        return act[:, :, self._out_index]

    def requires_grad_any(self) -> bool:
        return any(p.requires_grad for p in self.net.parameters())
