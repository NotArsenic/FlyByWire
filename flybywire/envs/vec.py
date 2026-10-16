"""Vectorised corridor envs for PPO (STUB — implemented with PPO).

Spec: section "Training / PPO / Rollouts".

* ``gymnasium.vector.AsyncVectorEnv(env_fns, shared_memory=True, autoreset_mode=...)`` with
  ``ppo.n_envs`` (12) workers; each worker returns frames only (``(4, H, W)`` uint8 per decision).
* BoxEye, flyvis and the core run batched in the main process on the GPU.
* Autoreset (verified in installed gymnasium 1.4.0, ``gymnasium/vector/async_vector_env.py``):
  the default is ``AutoresetMode.NEXT_STEP`` — the step *after* termination ignores its action
  and returns the reset observation. We use ``AutoresetMode.SAME_STEP`` instead: the terminating
  step returns the *reset* observation with ``info["final_obs"]`` / ``info["final_info"]``, so no
  dummy step is wasted and the main process can zero that env's core state and reset its flyvis
  state before consuming the new episode's frames (docs/DECISIONS.md).
"""

from __future__ import annotations


def make_vec_env(cfg, n_envs: int | None = None, seed: int = 0):
    """Return an ``AsyncVectorEnv`` of :class:`~flybywire.envs.corridor.CorridorEnv` (STUB)."""
    raise NotImplementedError("PPO phase: vectorised envs (spec 'PPO / Rollouts')")
