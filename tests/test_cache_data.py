"""Trajectory format, cache write/read roundtrip on synthetic data, and BC plumbing on CPU."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from flybywire.clocks import Clocks
from flybywire.data.cache import T4T5Cache, build_cache
from flybywire.data.trajectories import Trajectory, heading_target, trajectory_path


class FakeStepper:
    """Deterministic stand-in for flyvis: leaky integrator of the hexals, 8 'subtypes'."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.state = None

    def reset(self, batch_size=None, env_ids=None):
        self.state = torch.zeros(batch_size, 721)

    def step(self, hexals):
        self.state = 0.5 * self.state + hexals
        return torch.stack([self.state * (i + 1) / 8 for i in range(8)], dim=1)


def expected_output(hexals: np.ndarray) -> np.ndarray:
    s = np.zeros(721, np.float32)
    out = []
    for h in hexals.astype(np.float32):
        s = 0.5 * s + h
        out.append(np.stack([s * (i + 1) / 8 for i in range(8)]))
    return np.asarray(out)


def _make_traj(t_dec: int, seed: int) -> Trajectory:
    rng = np.random.default_rng(seed)
    t = 4 * t_dec
    rel = np.cumsum(rng.normal(0, 0.05, t)).astype(np.float32)
    return Trajectory(
        hexals=rng.random((t, 721)).astype(np.float16),
        actions=rng.uniform(-1, 1, (t_dec, 4)).astype(np.float32),
        rel_yaw=rel,
        yaw_rate=np.gradient(rel).astype(np.float32),
        meta={"seed": seed},
    )


@pytest.fixture()
def traj_dir(tmp_path):
    for i, k in enumerate([10, 7, 13, 4, 9]):  # different lengths
        _make_traj(k, i).save(trajectory_path(tmp_path / "expert", i))
    return tmp_path / "expert"


def test_trajectory_roundtrip_and_validation(tmp_path):
    tr = _make_traj(5, 0)
    p = tmp_path / "traj_00000.npz"
    tr.save(p)
    back = Trajectory.load(p)
    assert back.hexals.dtype == np.float16 and back.actions.dtype == np.float32
    assert np.array_equal(back.hexals, tr.hexals) and back.meta == {"seed": 0}
    bad = _make_traj(5, 1)
    bad.actions = bad.actions[:4]  # breaks T = 4 * T_dec
    with pytest.raises(ValueError):
        bad.validate(Clocks())


def test_cache_roundtrip(traj_dir, tmp_path):
    paths = sorted(traj_dir.glob("traj_*.npz"))
    out = build_cache(paths, tmp_path / "cache", FakeStepper(), manifest_extra={"flyvis_model": "fake", "dt": 0.01},
                      batch_size=2, progress=False)
    cache = T4T5Cache(out)
    assert len(cache) == 5
    assert cache.data.dtype == np.float16 and cache.data.shape[1:] == (8, 721)
    assert cache.offsets[-1] == sum(Trajectory.load(p).n_core_steps for p in paths)
    for i, p in enumerate(paths):
        exp = expected_output(Trajectory.load(p).hexals)
        got = cache.episode(i).astype(np.float32)
        assert got.shape == exp.shape
        np.testing.assert_allclose(got, exp, rtol=2e-3, atol=2e-3)  # fp16 storage
    assert cache.manifest["flyvis_model"] == "fake"
    cache.check_compatible("fake", 0.01)
    with pytest.raises(ValueError):
        cache.check_compatible("flow/0000/000", 0.01)


def test_bc_runs_on_cpu_and_uses_episode_targets(traj_dir, tmp_path, cfg, conn):
    from flybywire.train.bc import BCData, train_bc

    paths = sorted(traj_dir.glob("traj_*.npz"))
    out = build_cache(paths, tmp_path / "cache", FakeStepper(),
                      manifest_extra={"flyvis_model": cfg.vision.flyvis_model, "dt": 0.01},
                      batch_size=5, progress=False)
    data = BCData(T4T5Cache(out), Clocks())
    for i, p in enumerate(paths):  # invariant 9 at the data level
        assert np.allclose(data.targets[i].heading, heading_target(Trajectory.load(p).rel_yaw))
    from flybywire.config import load_config
    from flybywire.vision.lptc import LPTCPool

    c = load_config(overrides=["bc.batch_episodes=5", "bc.tbptt_core_steps=16"])
    lptc = LPTCPool(np.ones((8, 8, 721), np.float32) / 100)
    res = train_bc(c, out, tmp_path / "run", n_updates=6, device="cpu", verbose=False, conn=conn, lptc=lptc)
    assert len(res["history"]) == 6
    assert all(np.isfinite(h["loss"]) for h in res["history"])
    assert (tmp_path / "run" / "metrics.json").exists() and (tmp_path / "run" / "model.pt").exists()


def test_bc_module_never_imports_flyvis():
    code = "import sys, flybywire.train.bc, flybywire.data.cache; print('flyvis' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
