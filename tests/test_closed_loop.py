"""Closed-loop runner: stepping reproduces model.run, seeding, recording and baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import data_dir, load_config
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.eval.closed_loop import (
    Episode,
    ExpertDriver,
    PolicyDriver,
    fly,
    heading_error_deg,
    summarize,
)


@pytest.fixture()
def short_cfg():
    return load_config(overrides=["env.episode_seconds=0.4"])


def test_core_step_matches_run_on_cached_episode(cfg, model):
    try:
        cache = T4T5Cache(data_dir(cfg) / "cache")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    clocks = clocks_from_cfg(cfg)
    x = torch.from_numpy(np.array(cache.episode(0)[:40])).float()[None]
    with torch.no_grad():
        out = model.run(x, model.init_state(1))
        v, rs, actions = model.init_state(1), [], []
        for t in range(x.shape[1]):
            v, r = model.core_step(v, x[:, t])
            rs.append(r)
            if clocks.is_decision_step(t):
                actions.append(model.actor(r))
    torch.testing.assert_close(torch.stack(rs, dim=1), out["r"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.stack(actions, dim=1), out["action_mean"], atol=1e-5, rtol=1e-5)
    assert out["decision_t"].tolist() == clocks.decision_core_steps(10).tolist()


def test_heading_error_wraps():
    pred = np.stack([np.sin([3.1, -3.1, 0.2]), np.cos([3.1, -3.1, 0.2])], axis=1)
    err = heading_error_deg(pred, np.array([-3.1, 3.1, 0.2]))
    np.testing.assert_allclose(err, [np.degrees(2 * np.pi - 6.2)] * 2 + [0.0], atol=1e-6)


def test_summarize_counts_success_and_collisions(cfg):
    clocks = clocks_from_cfg(cfg)
    a = np.zeros((10, 4))
    eps = [Episode(0, 0, "time", 8.0, False, a, a), Episode(1, 1, "collision", 3.0, True, a, a + 0.1)]
    s = summarize(eps, clocks)
    assert s["success_rate"] == 0.5 and s["collisions"] == 1 and s["terminations"]["time"] == 1
    assert s["x_progress_mean_m"] == pytest.approx(5.5) and s["action_mse_vs_label"] == pytest.approx(0.005)


def test_expert_driver_matches_collect_expert_loop(short_cfg):
    from flybywire.envs.corridor import CorridorEnv
    from flybywire.envs.expert import ScriptedExpert

    eps = fly(short_cfg, [3, 4, 5], ExpertDriver(short_cfg, 2), n_envs=2)  # 3 seeds on 2 envs: one refill
    assert [e.seed for e in eps] == [3, 4, 5]
    env = CorridorEnv(short_cfg)
    try:
        for e in eps:
            expert = ScriptedExpert(short_cfg, seed=e.seed)
            _, info = env.reset(seed=e.seed)
            expert.reset(info)
            ref = []
            for _ in range(e.n_decisions):
                ref.append(expert.act(info))
                _, _, _, _, info = env.step(ref[-1])
            np.testing.assert_allclose(e.action, np.stack(ref), atol=1e-5)
            assert e.n_decisions == 10 and e.reason == "time"
    finally:
        env.close()


@pytest.mark.flyvis
@pytest.mark.gpu
def test_policy_flight_records_valid_trajectories_that_replay(short_cfg, model, tmp_path):
    clocks = clocks_from_cfg(short_cfg)
    model = model.cuda().eval()
    driver = PolicyDriver(model, short_cfg, n_envs=2)
    eps = fly(short_cfg, [0, 1, 2], driver, n_envs=2, record_dir=tmp_path)  # seed 2 runs after a refill
    assert [e.index for e in eps] == [0, 1, 2]
    for e in eps:
        tr = Trajectory.load(tmp_path / f"traj_{e.index:05d}.npz")
        tr.validate(clocks)
        assert tr.hexals.shape == (40, 721) and tr.meta["seed"] == e.seed
        np.testing.assert_allclose(tr.actions, e.label, atol=1e-6)
        x = driver.stepper.run_sequence(torch.from_numpy(tr.hexals.astype(np.float32)).cuda()[None])
        with torch.no_grad():
            replay = model.run(x, model.init_state(1))["action_mean"][0].cpu().numpy()
        np.testing.assert_allclose(replay, e.action, atol=1e-3)  # also shows the refilled env started from zero state
        assert e.heading_err_deg.shape == (40,)

    from flybywire.envs.corridor import CorridorEnv

    env = CorridorEnv(short_cfg)
    try:
        _, info = env.reset(seed=2)
    finally:
        env.close()
    np.testing.assert_allclose(Trajectory.load(tmp_path / "traj_00002.npz").state["pos"][:4], info["pos"], atol=1e-6)


@pytest.mark.flyvis
@pytest.mark.gpu
def test_labelling_expert_can_drop_maneuvers(short_cfg, model):
    from flybywire.envs.expert import ScriptedExpert

    model = model.cuda().eval()
    on = PolicyDriver(model, short_cfg, n_envs=1)
    off = PolicyDriver(model, short_cfg, n_envs=1, label_maneuvers=False)
    assert ScriptedExpert(on.label_cfg).turns is not None and ScriptedExpert(off.label_cfg).turns is None
    assert off.label_cfg.expert.wander_deg == 0 and short_cfg.expert.maneuvers is not None
