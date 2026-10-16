"""PPO fine-tuning from the BC policy (STUB — interfaces and docstrings only).

Spec: section "Training / PPO". Written by hand: no RLlib, no SB3.

Rollouts
    Synchronous, on-policy. ``ppo.n_envs`` (12) envs in ``AsyncVectorEnv(shared_memory=True,
    autoreset_mode=SAME_STEP)`` (see ``flybywire/envs/vec.py``). Envs return frames only; BoxEye,
    flyvis (:class:`~flybywire.vision.flyvis_stepper.FlyvisStepper`, batched B = n_envs) and the
    core run in the main process on the GPU — one copy of the weights, no broadcast. Each
    rollout is ``ppo.rollout_decisions`` (128) decisions per env = 512 core steps.

Recurrence
    Store the core state ``v`` (and flyvis state) at the start of each rollout; during the update,
    recompute the core through the whole rollout from that stored state. LPTC gradients need the
    rollout's T4/T5 activity, so keep it on the GPU: 12 x 512 x 8 x 721 fp32 ≈ 140 MB.

Episode boundaries
    When env i's episode ends (SAME_STEP autoreset: the returned obs already belongs to the new
    episode, ``info["final_obs"]`` holds the last frames), zero env i's core state and reset its
    flyvis state to the cached grey steady state (``FlyvisStepper.reset(env_ids=[i])``) *before*
    consuming the new frames; mark the boundary in the buffer so recomputation resets there too.

Optimisation
    GAE (gamma 0.99, lambda 0.95), ``ppo.epochs`` (4) epochs, clip 0.2, actor lr 3e-5, critic lr 3e-4.
    Initialise from BC; warm up the critic for ``ppo.critic_warmup_iters`` (10) iterations with the
    actor frozen; KL anchor to a frozen copy of the BC policy with beta 0.1 decaying linearly to 0
    over ``ppo.kl_anchor_decay_updates`` (50) updates; keep the auxiliary heading loss on
    (``ppo.aux_weight``). The critic reads ``r.detach()`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBuffer:
    """Per-rollout storage, all on the GPU (STUB).

    Fields (n_envs = E, rollout decisions = K, core steps = 4K):
      t4t5 (E, 4K, 8, 721) fp32, actions (E, K, 4), logp (E, K), rewards (E, K), values (E, K),
      dones (E, K) bool, rel_yaw (E, 4K), core_state0 (E, N), episode_start (E, 4K) bool.
    """

    t4t5: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    rel_yaw: torch.Tensor
    core_state0: torch.Tensor
    episode_start: torch.Tensor


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
    gamma: float = 0.99, lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over (E, K) tensors -> (advantages, returns) (STUB)."""
    raise NotImplementedError("PPO phase: GAE (spec 'PPO / Hyperparameters')")


def kl_anchor_beta(update: int, beta0: float = 0.1, decay_updates: int = 50) -> float:
    """Linearly decaying KL-anchor weight (STUB)."""
    raise NotImplementedError("PPO phase: KL anchor (spec 'Protecting the BC initialization')")


def collect_rollout(model, stepper, boxeye, vec_env, n_decisions: int, state: dict) -> RolloutBuffer:
    """Run ``n_decisions`` synchronous decisions in all envs, handling per-env resets (STUB)."""
    raise NotImplementedError("PPO phase: rollouts (spec 'PPO / Rollouts', 'Episode boundaries')")


def ppo_update(model, bc_policy, buffer: RolloutBuffer, optimizers: dict, cfg, update: int) -> dict:
    """Recompute the core through the rollout and apply clipped-PPO + KL-anchor + aux loss (STUB)."""
    raise NotImplementedError("PPO phase: update (spec 'PPO / Recurrence', 'Hyperparameters')")


def train_ppo(cfg, bc_checkpoint: str, out_dir: str) -> dict:
    """Full PPO loop: critic warm-up, rollouts, updates, logging, checkpoints (STUB)."""
    raise NotImplementedError("PPO phase (spec 'Training / PPO')")
