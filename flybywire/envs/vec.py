"""Corridor envs in worker processes for PPO rollouts."""

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
