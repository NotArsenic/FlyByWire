"""Corridor env, scripted expert and vec env."""

from __future__ import annotations

import numpy as np
import pytest

from flybywire.clocks import clocks_from_cfg
from flybywire.config import load_config
from flybywire.envs.corridor import CorridorEnv, body_to_world_velocity, denormalize_action, normalize_action
from flybywire.envs.expert import ScriptedExpert, surface_distance
from flybywire.envs.vec import make_vec_env


@pytest.fixture()
def short_cfg():
    return load_config(overrides=["env.episode_seconds=0.4"])


def test_action_normalisation_roundtrip(cfg):
    lim = cfg.action.limits
    a = np.array([-1.0, -0.3, 0.0, 0.7])
    np.testing.assert_allclose(normalize_action(denormalize_action(a, lim), lim), a, atol=1e-12)
    np.testing.assert_allclose(denormalize_action(np.array([-1.0, 1.0, -1.0, 1.0]), lim),
                               [lim.v_forward[0], lim.v_lateral[1], lim.v_vertical[0], lim.yaw_rate[1]])
    np.testing.assert_allclose(denormalize_action(normalize_action(np.zeros(4), lim), lim), 0.0, atol=1e-12)


def test_body_to_world_velocity():
    np.testing.assert_allclose(body_to_world_velocity(np.array([1.0, 0, 0.5]), np.pi / 2), [0, 1, 0.5], atol=1e-12)
    np.testing.assert_allclose(body_to_world_velocity(np.array([0, 1.0, 0]), np.pi / 2), [-1, 0, 0], atol=1e-12)


def test_surface_distance():
    d, n = surface_distance(1.0, 0.0, (0, 0.0, 0.0, 0.25, 0.25))
    assert d == pytest.approx(0.75) and np.allclose(n, [1, 0])
    d, n = surface_distance(0.0, 1.0, (1, 0.0, 0.0, 0.3, 0.4))
    assert d == pytest.approx(0.6) and np.allclose(n, [0, 1])


def test_env_frames_and_timing_follow_clocks(short_cfg):
    clocks = clocks_from_cfg(short_cfg)
    env = CorridorEnv(short_cfg, seed=0)
    try:
        frames, info = env.reset(seed=1)
        size = short_cfg.env.frame_size
        assert frames.shape == (4, size, size) and frames.dtype == np.uint8
        assert frames.std() > 0
        assert list(info["physics_step"]) == [clocks.frame_physics_step(t) for t in range(4)]
        assert info["obstacles"].shape == (short_cfg.env.n_obstacles, 5)
        assert info["yaw_0"] == pytest.approx(0.0, abs=1e-6)
        n_dec = clocks.n_decisions(short_cfg.env.episode_seconds)
        for k in range(n_dec):
            frames, _, term, trunc, info = env.step(env.hover_action)
            t = clocks.n_core_steps(k + 1) + np.arange(4)
            assert list(info["physics_step"]) == [clocks.frame_physics_step(i) for i in t]
            assert not term
            assert trunc == (k == n_dec - 1)
    finally:
        env.close()


def test_yaw_rate_command_turns_left(short_cfg):
    env = CorridorEnv(short_cfg, seed=0)
    try:
        _, info = env.reset(seed=2)
        a = normalize_action(np.array([0.0, 0.0, 0.0, 1.0]), env.limits).astype(np.float32)
        for _ in range(8):
            _, _, _, _, info = env.step(a)
        assert info["rel_yaw"][-1] > 0.15
        assert info["yaw_rate"][-1] == pytest.approx(1.0, abs=0.1)
        np.testing.assert_allclose(info["rel_yaw"], info["yaw"] - info["yaw_0"], atol=1e-6)
    finally:
        env.close()


def test_wall_textures_render(tmp_path):
    frames = {}
    for tex in ("none", "checker"):
        cfg = load_config(overrides=[f"env.wall_texture={tex}", f"paths.data_dir={tmp_path}"])
        env = CorridorEnv(cfg, seed=0)
        frames[tex], _ = env.reset(seed=3)
        env.close()
    assert (tmp_path / "textures" / "checker_256.png").exists()
    assert np.abs(frames["none"].astype(int) - frames["checker"]).mean() > 1.0


@pytest.mark.parametrize("seed", [0, 1])
def test_expert_flies_full_episode_without_collision(cfg, seed):
    env = CorridorEnv(cfg)
    env.render_frame = lambda: np.zeros((env.size, env.size), np.uint8)  # dynamics only
    expert = ScriptedExpert(cfg, seed=seed)
    try:
        _, info = env.reset(seed=seed)
        expert.reset(info)
        n = 0
        while True:
            a = expert.act(info)
            assert a.shape == (4,) and np.abs(a).max() <= 1.0
            _, _, term, trunc, info = env.step(a)
            n += 1
            if term or trunc:
                break
        assert not info["collision"]
        assert n == clocks_from_cfg(cfg).n_decisions(cfg.env.episode_seconds)
        assert info["x_progress"] > 5.0
    finally:
        env.close()


def test_vec_env_same_step_autoreset(short_cfg):
    envs = make_vec_env(short_cfg, n_envs=2, seed=0)
    try:
        obs, _ = envs.reset(seed=0)
        assert obs.shape == (2, 4, short_cfg.env.frame_size, short_cfg.env.frame_size)
        n_dec = clocks_from_cfg(short_cfg).n_decisions(short_cfg.env.episode_seconds)
        for _ in range(n_dec):
            obs, _, term, trunc, info = envs.step(np.zeros((2, 4), np.float32))
        assert trunc.all() and not term.any()
        assert info["_final_obs"].all() and info["final_obs"][0].shape == obs.shape[1:]
        assert (info["physics_step"] == np.arange(1, 8, 2)).all()  # already the next episode's pre-roll
    finally:
        envs.close()
