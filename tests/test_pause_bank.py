"""Pause bank: construction against a hand-rolled reference, splice alignment, and off-when-absent."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_cache_data import FakeStepper, _make_traj

from flybywire.clocks import Clocks
from flybywire.data.cache import T4T5Cache, build_cache
from flybywire.data.pause_bank import PauseBank, PauseSampler, build_bank, draw_entries
from flybywire.data.trajectories import trajectory_path

CLOCKS = Clocks()
HOLD, REC, K = (8, 16), 6, 40


@pytest.fixture()
def made(tmp_path):
    paths = []
    for i, k in enumerate([K, K + 5, K]):
        paths.append(trajectory_path(tmp_path / "expert", i))
        _make_traj(k, i).save(paths[-1])
    cache = build_cache(paths, tmp_path / "cache", FakeStepper(), batch_size=2, progress=False,
                        manifest_extra={"flyvis_model": "flow/0000/000", "dt": 0.01})
    cache = T4T5Cache(cache)
    build_bank(cache, FakeStepper(), tmp_path / "bank", CLOCKS, per_episode=2, hold_steps=HOLD, recovery=REC,
               batch_size=2, branch_size=3, progress=False)
    return cache, PauseBank(tmp_path / "bank"), paths


def hexals(path):
    from flybywire.data.trajectories import Trajectory

    return Trajectory.load(path).hexals.astype(np.float32)


def test_entries_sit_on_decision_boundaries_and_apart():
    lengths = [4 * 250] * 20
    ent = draw_entries(lengths, CLOCKS, 4, (48, 100), 60, np.random.default_rng(0))
    assert ((ent["t0"] + 1) % 4 == 0).all() and (ent["hold"] % 4 == 0).all()
    assert ent["hold"].min() >= 48 and ent["hold"].max() <= 100
    t0 = ent["t0"].reshape(20, 4)
    assert (np.diff(t0, axis=1) >= 120).all() and t0.min() >= 3 and t0.max() <= 999
    assert np.array_equal(ent["row"][1:], np.cumsum(ent["hold"] + ent["rec"])[:-1])
    with pytest.raises(ValueError):
        draw_entries([4 * 20], CLOCKS, 4, (48, 100), 60, np.random.default_rng(0))


def test_bank_is_the_held_frame_then_the_real_frames(made):
    cache, bank, paths = made
    assert len(bank) == 6 and bank.manifest["rerun_vs_cache_max_abs"] == 0
    bank.check_cache(cache)
    for e in range(len(bank)):
        h = hexals(paths[bank.episode[e]])
        t0, m = int(bank.t0[e]), int(bank.hold[e])
        s = np.zeros(721, np.float32)
        for t in range(t0 + 1):
            s = 0.5 * s + h[t]
        seq = [h[t0]] * m + [h[t0 + 1 + j] for j in range(int(bank.rec[e]))]
        want = []
        for f in seq:
            s = 0.5 * s + f
            want.append(np.stack([s * (i + 1) / 8 for i in range(8)]))
        got = np.concatenate(bank.segments(e)).astype(np.float32)
        np.testing.assert_allclose(got, np.stack(want), rtol=2e-3, atol=2e-3)
    rel = bank.manifest["recovery_vs_cache_rel"]
    assert len(rel) == REC and rel[-1] < rel[0]


@pytest.mark.flyvis
@pytest.mark.gpu
def test_bank_with_flyvis_branches_from_the_cache_state(tmp_path, cfg):
    from flybywire.vision.flyvis_stepper import FlyvisStepper

    paths = []
    for i in range(2):
        paths.append(trajectory_path(tmp_path / "expert", i))
        _make_traj(30, i).save(paths[-1])
    stepper = FlyvisStepper(cfg.vision.flyvis_model, "cuda", 0.01)
    cache = T4T5Cache(build_cache(paths, tmp_path / "cache", stepper, progress=False))
    build_bank(cache, stepper, tmp_path / "bank", CLOCKS, per_episode=2, hold_steps=(8, 12), recovery=8,
               progress=False)
    bank = PauseBank(tmp_path / "bank")
    assert bank.manifest["rerun_vs_cache_max_abs"] < 2e-3
    for e in (0, 3):
        h = torch.from_numpy(hexals(paths[bank.episode[e]])).cuda()
        t0, m = int(bank.t0[e]), int(bank.hold[e])
        stepper.reset(1)
        for t in range(t0 + 1):
            stepper.step(h[t][None])
        want = [stepper.step(h[t0][None]) for _ in range(m)]
        want += [stepper.step(h[t0 + 1 + j][None]) for j in range(int(bank.rec[e]))]
        got = np.concatenate(bank.segments(e)).astype(np.float32)
        np.testing.assert_allclose(got, torch.cat(want).cpu().numpy(), atol=2e-3)


def positions(n_steps, pauses):
    """Spliced timeline as (kind, original step) per core step, written the slow way."""
    out = []
    for t in range(n_steps):
        out.append(("real", t))
        out += [("pause", t)] * sum(m for t0, m in pauses if t0 == t)
    return out


def test_splice_alignment(made):
    from flybywire.train.bc import BCData

    cache, bank, paths = made
    data = BCData(cache, CLOCKS, PauseSampler(bank, per_episode=2, p=1.0))
    for i in range(len(data)):
        picked = data.pauses.draw(i, np.random.default_rng(3 + i))
        x, head, act, amask = data.episode(i, np.random.default_rng(3 + i))
        want = positions(cache.length(i), [(int(bank.t0[e]), int(bank.hold[e])) for e in picked])
        assert len(x) == len(head) == len(want) and len(act) == len(amask) == len(want) // 4
        holds = {int(bank.t0[e]): bank.segments(e) for e in picked}
        seen = {t0: 0 for t0 in holds}
        for q, (kind, t) in enumerate(want):
            np.testing.assert_array_equal(head[q], data.targets[i].heading[t])
            if kind == "pause":
                np.testing.assert_array_equal(x[q], holds[t][0][seen[t]])
                seen[t] += 1
            else:
                for t0, (_, r) in holds.items():
                    if t0 < t <= t0 + len(r):
                        np.testing.assert_array_equal(x[q], r[t - t0 - 1])
                        break
                else:
                    np.testing.assert_array_equal(x[q], cache.episode(i)[t])
        for k, q in enumerate(CLOCKS.decision_core_steps(len(act))):
            kind, t = want[q]
            assert amask[k] == (kind == "real")
            if kind == "real":
                np.testing.assert_array_equal(act[k], data.targets[i].actions[t // 4])
        assert (~amask).sum() == sum(int(bank.hold[e]) for e in picked) // 4 > 0


def old_batch(data, idx):
    """BCData.batch as it was before pauses."""
    spd = data.clocks.sensory_per_decision
    lengths = [data.cache.length(i) for i in idx]
    tmax = int(np.ceil(max(lengths) / spd) * spd)
    kmax = tmax // spd
    x = np.zeros((len(idx), tmax, *data.cache.data.shape[1:]), dtype=np.float16)
    head = np.zeros((len(idx), tmax, 2), dtype=np.float32)
    mask = np.zeros((len(idx), tmax), dtype=bool)
    act = np.zeros((len(idx), kmax, 4), dtype=np.float32)
    amask = np.zeros((len(idx), kmax), dtype=bool)
    for j, i in enumerate(idx):
        t = lengths[j]
        x[j, :t] = data.cache.episode(i)
        head[j, :t] = data.targets[i].heading
        mask[j, :t] = True
        k = data.targets[i].actions.shape[0]
        act[j, :k] = data.targets[i].actions
        amask[j, :k] = True
    return {"x16": x, "heading": head, "mask": mask, "actions": act, "amask": amask}


def test_batch_without_pauses_is_the_old_batch(made):
    from flybywire.train.bc import BCData

    cache, bank, _ = made
    idx = [2, 1, 0]
    want = old_batch(BCData(cache, CLOCKS), idx)
    rng = np.random.default_rng(0)
    for data in (BCData(cache, CLOCKS), BCData(cache, CLOCKS, PauseSampler(bank, 2, 0.0))):
        for got in (data.batch(idx, "cpu"), data.batch(idx, "cpu", rng)):
            assert got.keys() == want.keys()
            for k, v in want.items():
                assert np.array_equal(got[k].numpy(), v), k
    data = BCData(cache, CLOCKS, PauseSampler(bank, 2, 1.0))
    assert all(np.array_equal(data.batch(idx, "cpu")[k].numpy(), v) for k, v in want.items())  # no rng: unspliced
    spliced = data.batch(idx, "cpu", rng)
    assert spliced["x16"].shape[1] > want["x16"].shape[1] and spliced["x16"].shape[1] % 4 == 0


def test_action_loss_ignores_inserted_decisions(made, model):
    from flybywire.train.bc import BCData, window_loss

    cache, bank, _ = made
    data = BCData(cache, CLOCKS, PauseSampler(bank, 2, 1.0))
    batch = data.batch([0, 1, 2], "cpu", np.random.default_rng(1))
    assert (~batch["amask"][:, : batch["actions"].shape[1]]).any()
    v = model.init_state(3)
    n = batch["x16"].shape[1]
    _, la, lh, _ = window_loss(model, batch, v, 0, n, 0.5, CLOCKS)
    junk = torch.full_like(batch["actions"], 1e3)
    poisoned = {**batch, "actions": torch.where(batch["amask"][..., None], batch["actions"], junk)}
    _, la2, lh2, _ = window_loss(model, poisoned, v, 0, n, 0.5, CLOCKS)
    assert torch.equal(la, la2) and torch.equal(lh, lh2)


def run_bc(tmp_path, name, cache_dir, cfg, conn, overrides=(), drop_key=False):
    from flybywire.config import load_config
    from flybywire.train.bc import train_bc
    from flybywire.vision.lptc import LPTCPool

    c = load_config(overrides=["bc.batch_episodes=3", "bc.tbptt_core_steps=16", *overrides])
    if drop_key:
        d = OmegaConf.to_container(c)
        del d["bc"]["pauses"]
        c = OmegaConf.create(d)
    lptc = LPTCPool(np.ones((8, 8, 721), np.float32) / 100)
    return train_bc(c, cache_dir, tmp_path / name, n_updates=12, device="cpu", verbose=False, conn=conn, lptc=lptc)


def test_training_is_unchanged_without_pauses(made, tmp_path, cfg, conn):
    cache, bank, _ = made
    plain = run_bc(tmp_path, "plain", cache.dir, cfg, conn, drop_key=True)["history"]
    for name, kw in (("null", {}), ("p0", {"overrides": [f"bc.pauses.bank={bank.dir}", "bc.pauses.per_episode=2",
                                                         "bc.pauses.p=0"]})):
        other = run_bc(tmp_path, name, cache.dir, cfg, conn, **kw)["history"]
        assert [h["loss"] for h in other] == [h["loss"] for h in plain]


def test_training_with_pauses_runs_across_windows(made, tmp_path, cfg, conn):
    cache, bank, _ = made
    over = [f"bc.pauses.bank={bank.dir}", "bc.pauses.per_episode=2", "bc.pauses.p=1"]
    res = run_bc(tmp_path, "pauses", cache.dir, cfg, conn, over)
    assert all(np.isfinite(h["loss"]) for h in res["history"])
    assert max(h["window_start"] for h in res["history"]) > 16  # state carried over several windows
    assert res["metrics"]["pauses"]["per_episode"] == 2
    plain = run_bc(tmp_path, "plain", cache.dir, cfg, conn)["history"]
    assert [h["loss"] for h in res["history"]] != [h["loss"] for h in plain]


def test_bank_of_another_cache_is_refused(made, tmp_path):
    from flybywire.train.bc import BCData

    cache, bank, paths = made
    other = T4T5Cache(build_cache(paths[:2], tmp_path / "other", FakeStepper(), progress=False))
    with pytest.raises(ValueError):
        BCData(other, CLOCKS, PauseSampler(bank, 2, 1.0))
