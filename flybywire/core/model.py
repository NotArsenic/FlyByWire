"""Model wiring at multiple rates.

    T4/T5 (B, 8, 721) --LPTCPool--> s (B, 8) --InputProjection--> u (B, N), ER only
        --CentralComplex (one step per sensory tick, 100 Hz)--> r (B, N)
        --HeadingDecoder (EPG, every core step) / Actor (PFL3, decision steps, 25 Hz) / Critic (r.detach())

With the Farneback backend (``lptc=None``) the input is already ``(B, 8)``.
Decision timing comes from :class:`flybywire.clocks.Clocks`: a decision is read out after core
step ``t`` iff ``clocks.is_decision_step(t)`` (global step index).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from flybywire.clocks import Clocks, clocks_from_cfg
from flybywire.connectome.celltypes import Connectome
from flybywire.core.heads import Actor, AnatomicalHeadingDecoder, Critic, HeadingDecoder, InputProjection
from flybywire.core.lesion import keep_mask
from flybywire.core.population import CentralComplex


class FlyByWireModel(nn.Module):
    def __init__(
        self,
        conn: Connectome,
        lptc: nn.Module | None,
        clocks: Clocks,
        input_group: str = "ER",
        heading_group: str = "EPG",
        actor_group: str = "PFL3",
        lesion_groups: Iterable[str] = ("EPG", "PEN", "D7"),
        n_channels: int = 8,
        n_actions: int = 4,
        alpha_init: float = 0.2,
        w_init_gain: float = 0.5,
        phi: str = "softplus",
        input_dependent_alpha: bool = False,
        log_std_init: float = -1.5,
        eb_angles: tuple | None = None,
        seed: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        self.clocks = clocks
        self.groups = {k: np.asarray(v) for k, v in conn.groups.items()}
        self.lesion_groups = tuple(lesion_groups)
        self.group_roles = {"input": input_group, "heading": heading_group, "actor": actor_group}
        self.n = conn.n
        self.lptc = lptc
        self.inproj = InputProjection(n_channels, conn.n, self.groups[input_group])
        self.core = CentralComplex(
            conn.M,
            alpha_init=alpha_init,
            w_init_gain=w_init_gain,
            phi=phi,
            input_dependent_alpha=input_dependent_alpha,
            generator=gen,
            verbose=verbose,
        )
        if eb_angles is None:
            self.heading = HeadingDecoder(self.groups[heading_group])
        else:
            self.heading = AnatomicalHeadingDecoder(self.groups[heading_group], *eb_angles)
        self.actor = Actor(self.groups[actor_group], n_actions, log_std_init)
        self.critic = Critic(conn.n)

    def lesion(self, names: Iterable[str]) -> None:
        """Silence the given cell-type groups (rates exactly 0) until :meth:`restore`."""
        self.core.keep.copy_(keep_mask(self.n, self.groups, names).to(self.core.keep.device))

    def restore(self) -> None:
        self.core.keep.fill_(1.0)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return self.core.init_state(batch_size)

    def sensory(self, x: torch.Tensor) -> torch.Tensor:
        """(..., 8, 721) T4/T5 -> (..., 8) channels; identity for the Farneback backend."""
        return self.lptc(x) if self.lptc is not None else x.float()

    def core_step(self, v: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One sensory tick: x = T4/T5 (B, 8, 721) or channels (B, 8). Returns (v_next, r_next)."""
        u = self.inproj(self.sensory(x))
        return self.core(v, u)

    def run(self, x_seq: torch.Tensor, v0: torch.Tensor, t0: int = 0) -> dict[str, torch.Tensor]:
        """Run the core over a window of T sensory ticks starting at global core step ``t0``.

        x_seq: (B, T, 8, 721) or (B, T, 8). Returns ``v`` (final state), ``r`` (B, T, N),
        ``heading`` (B, T, 2), ``action_mean`` (B, K, 4) at decision steps, ``decision_t`` (K,)
        local indices of those steps, and ``s`` (B, T, 8) sensory channels.
        """
        b, t = x_seq.shape[:2]
        s = self.sensory(x_seq)  # LPTC over the whole window at once (no recurrence there)
        u = self.inproj(s)
        v = v0
        rs = []
        for i in range(t):
            v, r = self.core(v, u[:, i])
            rs.append(r)
        r = torch.stack(rs, dim=1)
        dec = torch.tensor(
            [i for i in range(t) if self.clocks.is_decision_step(t0 + i)], dtype=torch.long, device=r.device
        )
        return {
            "v": v,
            "r": r,
            "heading": self.heading(r),
            "action_mean": self.actor(r.index_select(1, dec)),
            "decision_t": dec,
            "s": s,
        }


def build_model(cfg, conn: Connectome | None = None, lptc: nn.Module | None = None, verbose: bool = True):
    """Build the model from config. Loads the connectome / LPTC templates unless given."""
    from flybywire.connectome.masks import build_connectome, load_eb_angles

    if conn is None:
        conn = build_connectome(cfg)
    eb_angles = None
    if OmegaConf.select(cfg, "heads.decoder", default="learned") == "anatomical":
        eb_angles = load_eb_angles(cfg, conn)
    if lptc is None and cfg.vision.backend == "flyvis":
        from flybywire.vision.lptc import lptc_from_cfg

        lptc = lptc_from_cfg(cfg)
    return FlyByWireModel(
        conn,
        lptc,
        clocks_from_cfg(cfg),
        input_group=cfg.connectome.input_group,
        heading_group=cfg.connectome.heading_group,
        actor_group=cfg.connectome.actor_group,
        lesion_groups=tuple(cfg.connectome.lesion_groups),
        n_channels=len(cfg.lptc.channels),
        alpha_init=float(cfg.core.alpha_init),
        w_init_gain=float(cfg.core.w_init_gain),
        phi=cfg.core.phi,
        input_dependent_alpha=bool(cfg.core.input_dependent_alpha),
        log_std_init=float(cfg.heads.log_std_init),
        eb_angles=eb_angles,
        seed=int(cfg.seed),
        verbose=verbose,
    )
