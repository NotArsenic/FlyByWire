"""Corridor envs in worker processes for PPO rollouts.

Workers only render and step physics; BoxEye, flyvis and the core run batched in the main
process. With SAME_STEP autoreset (D20) the step that ends an episode already returns the next
episode's frames, plus info["final_obs"] / info["final_info"], so the main process has to reset
that env's core and flyvis state before using them. Workers are spawned, not forked, so they
don't inherit the main process's CUDA context.
"""

from __future__ import annotations

from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from flybywire.envs.corridor import CorridorEnv


def _make(cfg, seed: int):
    return lambda: CorridorEnv(cfg, seed=seed)


def make_vec_env(cfg, n_envs: int | None = None, seed: int = 0) -> AsyncVectorEnv:
    n = int(n_envs or cfg.ppo.n_envs)
    return AsyncVectorEnv(
        [_make(cfg, seed + i) for i in range(n)],
        shared_memory=True,
        context="spawn",
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
