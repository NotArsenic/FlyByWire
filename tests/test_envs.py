"""Corridor env, scripted expert and vec env."""

from __future__ import annotations

import numpy as np
import pytest

from flybywire.clocks import clocks_from_cfg
from flybywire.config import data_dir, load_config
from flybywire.data.trajectories import Trajectory, trajectory_path
from flybywire.envs.corridor import CorridorEnv, body_to_world_velocity, denormalize_action, normalize_action
from flybywire.envs.expert import ScriptedExpert, surface_distance, turn_profile
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


def fly(cfg, seed):
    """Run the expert for a full episode without rendering; returns actions and per-frame info."""
    env = CorridorEnv(cfg)
    env.render_frame = lambda: np.zeros((env.size, env.size), np.uint8)
    expert = ScriptedExpert(cfg, seed=seed)
    try:
        _, info = env.reset(seed=seed)
        expert.reset(info)
        actions, rel_yaw, yaw_rate = [], [info["rel_yaw"]], [info["yaw_rate"]]
        while True:
            actions.append(expert.act(info))
            _, _, term, trunc, info = env.step(actions[-1])
            if term or trunc:
                break
            rel_yaw.append(info["rel_yaw"])
            yaw_rate.append(info["yaw_rate"])
        return np.stack(actions), np.concatenate(rel_yaw), np.concatenate(yaw_rate), info
    finally:
        env.close()


def test_turn_profile():
    for angle in (0.3, 3.0):
        s, v, t_end = turn_profile(angle, 2.5, 8.0, 0.0)
        assert s == 0.0 and v == 0.0
        s, v, _ = turn_profile(angle, 2.5, 8.0, t_end)
        assert s == pytest.approx(angle) and v == 0.0
        t = np.linspace(0, t_end, 200)
        s, v, _ = np.array([turn_profile(angle, 2.5, 8.0, ti) for ti in t]).T
        assert (np.diff(s) >= 0).all() and v.max() <= 2.5 + 1e-9
        np.testing.assert_allclose(np.diff(s) / np.diff(t), 0.5 * (v[1:] + v[:-1]), atol=0.05)
    assert turn_profile(3.0, 2.5, 8.0, 0.0)[2] == pytest.approx(2.5 / 8.0 * 2 + (3.0 - 2.5**2 / 8.0) / 2.5)


# the episode ends before a second maneuver can start
@pytest.mark.parametrize("angle, span, seconds", [(360, 360, 4.6), (270, 360, 5.2), (135, 135, 4.6)])
def test_maneuver_turns_and_returns_to_heading_zero(angle, span, seconds):
    cfg = load_config(overrides=[
        "env.n_obstacles=0", f"env.episode_seconds={seconds}", f"expert.maneuvers.angles_deg=[{angle}]",
        "expert.maneuvers.wait_s=[1.0,1.0]", "expert.maneuvers.hold_s=[0.2,0.2]",
    ])
    actions, rel_yaw, yaw_rate, info = fly(cfg, seed=0)
    assert not info["collision"]
    assert np.degrees(np.ptp(np.unwrap(rel_yaw))) == pytest.approx(span, abs=30)
    assert abs(rel_yaw[-1]) < 0.4
    assert np.abs(yaw_rate).max() < cfg.action.limits.yaw_rate[1]
    assert np.abs(actions[:, 3]).max() < 1.0
    if span == 360:
        assert np.abs(rel_yaw).max() > 3.0  # went through +/-pi


def test_yaw_crossing_pi_has_no_kick(short_cfg):
    env = CorridorEnv(short_cfg, seed=0)
    try:
        env.reset(seed=0)
        cmd = np.array([0.0, 0.0, 0.0, 1.0])
        yaw, rate = [], []
        for _ in range(int(3.6 * short_cfg.clocks.physics_hz)):
            env._physics_step(cmd)
            s = env._state()
            yaw.append(s["rpy"][2])
            rate.append(s["yaw_rate"])
        assert (np.abs(np.diff(yaw)) > np.pi).sum() == 1
        # the unwrapped D term used to add 0.18 rad/s in one physics step here
        assert np.abs(np.diff(rate[20:])).max() < 0.02
    finally:
        env.close()


def test_expert_without_maneuvers_is_unchanged(cfg):
    plain = load_config()
    del plain.expert["maneuvers"]
    never = load_config(overrides=["expert.maneuvers.wait_s=[100,100]"])
    np.testing.assert_array_equal(fly(plain, seed=3)[0], fly(never, seed=3)[0])

    old = load_config()
    del old.expert["maneuvers"]
    old.action.limits.yaw_rate = [-1.5, 1.5]
    stored = trajectory_path(data_dir(old) / "expert", 0)
    if not stored.exists():
        pytest.skip("no stored expert data")
    actions, rel_yaw, _, _ = fly(old, seed=0)
    ref = Trajectory.load(stored, load_hexals=False)
    np.testing.assert_array_equal(actions, ref.actions)
    np.testing.assert_array_equal(rel_yaw.astype(np.float32), ref.rel_yaw)


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
