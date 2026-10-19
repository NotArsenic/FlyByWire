"""Blank-input branch of the persistence test: state hand-off, recurrence ablation and statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybywire.eval.darkness import (
    dark_branch,
    dark_series,
    dark_stats,
    grey_t4t5,
    hold_branch,
    pause_branch,
    recurrence_scaled,
    signed_stats,
    snapshots,
    static_t4t5,
    velocity_field,
    yaw_calibration,
)
from flybywire.eval.ring import make_layout


def _x(b=2, t=12):
    return torch.randn(b, t, 8, 721, generator=torch.Generator().manual_seed(0))


def test_snapshots_match_a_plain_run(model):
    x = _x()
    snaps = snapshots(model, x, [5, 10])
    for t0, (v, r) in snaps.items():
        out = model.run(x[:, :t0], model.init_state(2))
        assert torch.allclose(v, out["v"]) and torch.allclose(r, out["r"][:, -1])


def test_dark_branch_without_recurrence_is_a_leaky_unit(model):
    x = _x()
    v, r = snapshots(model, x, [8])[8]
    blank = torch.zeros(8, 721)
    w = model.core.W.detach().clone()
    full = dark_branch(model, v, r, blank, 3)
    off = dark_branch(model, v, r, blank, 3, recurrent=False)
    assert full.shape == off.shape == (2, 4, model.n) and torch.equal(full[:, 0], r)
    assert torch.equal(model.core.W, w)
    u = model.inproj(model.sensory(blank[None]))
    v1 = (1 - model.core.alpha()) * v + model.core.alpha() * (u + model.core.b)
    assert torch.allclose(off[:, 1], model.core.phi(v1), atol=1e-6)
    assert not torch.allclose(full[:, 1], off[:, 1])


def test_dark_stats_start_at_no_drift_and_full_size(model):
    x = _x()
    v, r = snapshots(model, x, [8])[8]
    rs = dark_branch(model, v, r, torch.zeros(8, 721), 20)
    layout = make_layout("tuning", None, model.heading.weight.detach())
    series = dark_series(model, rs, layout)
    assert all(a.shape == (2, 21) for a in series.values())
    stats = dark_stats(series, [0, 10, 20])
    assert stats["heading_drift_deg_mean"][0] == 0 and stats["ring_drift_deg_median"][0] == 0
    assert np.allclose([stats["decoder_retention"][0], stats["ring_retention"][0]], 1.0)
    assert set(stats) >= {"heading_drift_deg_q25", "heading_drift_deg_q75", "ring_fraction"}


def test_hold_branch_takes_one_channel_vector_per_sample(model):
    v, r = snapshots(model, _x(), [8])[8]
    s = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    both = hold_branch(model, v, r, s, 4)
    for i in range(2):
        alone = hold_branch(model, v[i : i + 1], r[i : i + 1], s[i], 4)
        assert torch.allclose(both[i : i + 1], alone, atol=1e-6)


def test_pause_branch_follows_the_rows_then_holds_the_last(model):
    v0, r0 = snapshots(model, _x(), [8])[8]
    seq = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(2))
    got = pause_branch(model, v0, r0, seq, 6)
    assert got.shape == (2, 7, model.n) and torch.equal(got[:, 0], r0)
    v = v0
    for t in range(6):
        v, r = model.core(v, model.inproj(seq[:, min(t, 2)]))
        assert torch.allclose(got[:, t + 1], r, atol=1e-6)
    const = pause_branch(model, v0, r0, seq[:, :1], 4)
    assert torch.allclose(const, hold_branch(model, v0, r0, seq[:, 0], 4), atol=1e-6)


def test_recurrence_scaled_restores_the_weights_and_zero_is_off(model):
    v, r = snapshots(model, _x(), [8])[8]
    s = torch.zeros(8)
    w = model.core.W.detach().clone()
    off = hold_branch(model, v, r, s, 3, recurrent=False)
    with recurrence_scaled(model.core, 0.0):
        zero = hold_branch(model, v, r, s, 3)
    with recurrence_scaled(model.core, 1.0):
        same = hold_branch(model, v, r, s, 3)
    assert torch.equal(zero, off) and torch.equal(same, hold_branch(model, v, r, s, 3))
    assert torch.equal(model.core.W, w)


def test_signed_stats_one_direction_vs_cancelling_signs():
    episode = np.repeat(np.arange(20), 5)
    left = signed_stats(np.full(100, -10.0) + np.random.default_rng(0).normal(0, 1, 100), episode)
    assert left["mean_over_mean_abs"] < -0.95 and left["frac_negative"] == 1.0
    assert left["episodes_negative"] == 20 and left["sign_test_p"] < 1e-4 and left["ci95_high"] < -9
    signs = np.tile([1.0, -1.0], 50)
    flat = signed_stats(10 * signs, episode)
    assert abs(flat["mean_over_mean_abs"]) < 0.01 and flat["sign_test_p"] > 0.05


def test_yaw_calibration_recovers_gain_and_zero_level():
    rng = np.random.default_rng(0)
    rate = rng.normal(0, 1.0, (6, 400))
    s_yaw = 0.5 * rate - 0.8  # channel = 0.5 per rad/s, -0.8 at zero rotation
    angle = (np.cumsum(rate, axis=1) - rate) * 0.01  # angle[t] is the sum of the rates before step t
    cal = yaw_calibration(s_yaw, angle, rate)
    for fit in ("decoded", "true"):
        assert np.isclose(cal[fit]["gain_rad_s_per_unit"], 2.0, rtol=1e-3)
        assert np.isclose(cal[fit]["zero_turn_level"], -0.8, atol=1e-3)


def test_velocity_field_finds_the_heading_it_settles_at():
    rng = np.random.default_rng(0)
    start = np.radians(rng.uniform(-90, 90, 4000))
    drift = -0.5 * (np.degrees(start) - 20.0) + rng.normal(0, 1, len(start))
    f = velocity_field(drift, start)
    assert len(f["stable_deg"]) == 1 and abs(f["stable_deg"][0] - 20) < 3 and not f["unstable_deg"]


@pytest.mark.flyvis
@pytest.mark.gpu
def test_holding_a_grey_frame_continues_the_grey_settling(cfg):
    # the 0.5 s grey state is still settling, so 0.5 s + 0.05 s of held grey is the 0.55 s state
    later = grey_t4t5(cfg, "cuda", 0.55).cpu().numpy()
    held = static_t4t5(cfg, "cuda", np.full((3, 721), 0.5, np.float32), seconds=0.05, chunk=2)
    assert held.shape == (3, 8, 721) and np.abs(held - later).max() < 2e-3  # later is rounded to fp16
