"""Pilot script helpers: the layout stand-in must match the env's, the ridge must find a linear signal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flybywire.config import load_config
from flybywire.envs.corridor import CorridorEnv

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "texture_pilot.py"


@pytest.fixture(scope="module")
def pilot():
    spec = importlib.util.spec_from_file_location("texture_pilot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_layout_matches_the_env(pilot):
    cfg = load_config(overrides=["env.episode_seconds=0.4"])
    env = CorridorEnv(cfg)
    try:
        _, info = env.reset(seed=7)
    finally:
        env.close()
    np.testing.assert_array_equal(pilot.layout(7, cfg), info["obstacles"])


def test_ridge_recovers_a_linear_target(pilot):
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((40, 30)).astype(np.float32) for _ in range(10)]
    w = rng.standard_normal((30, len(pilot.TARGETS))).astype(np.float32)
    ys = [x @ w + 0.01 * rng.standard_normal((40, len(pilot.TARGETS))).astype(np.float32) for x in xs]
    out = pilot.run(xs, ys, "cpu", n_boot=20)
    assert min(out["test_r2"].values()) > 0.95
    assert out["n_episodes"] == {"test": 2, "val": 1, "train": 7}


def test_mlp_probe_runs_and_learns_a_nonlinear_target(pilot):
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((60, 10)).astype(np.float32) for _ in range(10)]
    ys = [np.stack([np.abs(x[:, 0]), x[:, 1] * x[:, 2]] + [x[:, 3]] * (len(pilot.TARGETS) - 2), axis=1) for x in xs]
    ridge = pilot.run(xs, ys, "cpu", n_boot=20)["test_r2"]
    mlp = pilot.run(xs, ys, "cpu", n_boot=20, probe="mlp")["test_r2"]
    assert mlp["yaw_rate"] > ridge["yaw_rate"] + 0.3 and mlp["a_lateral"] > 0.9
