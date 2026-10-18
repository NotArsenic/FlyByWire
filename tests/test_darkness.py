"""Blank-input branch of the persistence test: state hand-off, recurrence ablation and statistics."""

from __future__ import annotations

import numpy as np
import torch

from flybywire.eval.darkness import dark_branch, dark_series, dark_stats, snapshots
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
