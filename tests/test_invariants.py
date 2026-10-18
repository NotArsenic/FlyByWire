"""At least one test per architecture invariant (numbered as in AGENTS.md)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from flybywire.connectome import apply_mask_type, directed_rewire
from flybywire.core.lesion import lesioned
from flybywire.data.trajectories import heading_target, relative_yaw, wrap_angle


def _inputs(b=2, t=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, 8, 721, generator=g)


def test_inv1_core_has_no_cross_neuron_modules(model):
    """The core owns no submodules at all (no LayerNorm / attention / Linear over neurons)."""
    assert list(model.core.children()) == []
    banned = (nn.LayerNorm, nn.MultiheadAttention, nn.Linear, nn.BatchNorm1d, nn.GroupNorm)
    assert not any(isinstance(m, banned) for m in model.core.modules())


def test_inv1_jacobian_follows_mask(model, conn):
    """d v_next[j] / d v[i] == 0 exactly wherever M[i, j] == 0 (i != j): the only path is r @ (W*M)."""
    core = model.core
    u = torch.randn(1, core.n)
    v0 = torch.randn(1, core.n)
    J = torch.autograd.functional.jacobian(lambda v: core(v, u)[0], v0)[0, :, 0, :]  # J[j, i] = dv'_j/dv_i
    M = torch.as_tensor(conn.M)
    off = ~M.T & ~torch.eye(core.n, dtype=torch.bool)  # (j, i) entries with no edge i -> j
    assert torch.all(J[off] == 0)
    assert torch.count_nonzero(J[M.T]) > 0  # and the mask does carry signal


def test_inv2_grad_zero_off_mask(model, conn):
    out = model.run(_inputs(), model.init_state(2))
    loss = out["action_mean"].pow(2).sum() + out["heading"].pow(2).sum()
    loss.backward()
    g = model.core.W.grad
    M = torch.as_tensor(conn.M)
    assert g is not None
    assert torch.all(g[~M] == 0)
    assert torch.count_nonzero(g[M]) > 0


def test_inv3_input_only_on_er(model, conn):
    s = torch.randn(5, 8)
    u = model.inproj(s)
    er = np.zeros(conn.n, dtype=bool)
    er[conn.groups["ER"]] = True
    assert torch.all(u[:, ~torch.as_tensor(er)] == 0)
    assert torch.count_nonzero(u[:, torch.as_tensor(er)]) > 0


def test_inv4_actor_reads_only_pfl3(model, conn):
    r = torch.rand(3, conn.n)
    r2 = r.clone()
    non = np.setdiff1d(np.arange(conn.n), conn.groups["PFL3"])
    r2[:, non] += torch.randn(3, len(non)) * 10
    assert torch.equal(model.actor(r), model.actor(r2))
    r3 = r.clone()
    r3[:, conn.groups["PFL3"]] += 1.0
    assert not torch.equal(model.actor(r), model.actor(r3))


def test_inv4_heading_reads_only_epg(model, conn):
    r = torch.rand(3, conn.n)
    r2 = r.clone()
    non = np.setdiff1d(np.arange(conn.n), conn.groups["EPG"])
    r2[:, non] += torch.randn(3, len(non)) * 10
    assert torch.equal(model.heading(r), model.heading(r2))


def test_inv4_critic_does_not_backprop_into_core(model):
    out = model.run(_inputs(), model.init_state(2))
    model.critic(out["r"]).sum().backward()
    for name, p in model.named_parameters():
        if not name.startswith("critic."):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.critic.parameters())


@pytest.mark.flyvis
@pytest.mark.gpu
def test_inv5_flyvis_receives_no_gradients(random_lptc):
    from flybywire.vision.flyvis_stepper import FlyvisStepper

    st = FlyvisStepper(device="cuda", steady_state_seconds=0.1)
    assert all(not p.requires_grad for p in st.net.parameters())
    assert not st.net.training
    out = st.step(torch.full((2, 721), 0.5, device="cuda"))
    assert not out.requires_grad
    lptc = random_lptc.cuda()
    lptc(out).sum().backward()  # gradient reaches LPTC gains but stops at the flyvis output
    assert lptc.g.grad is not None
    assert all(p.grad is None for p in st.net.parameters())


def test_inv6_lesion_rates_exactly_zero_every_step(model, conn):
    x = _inputs(t=12)
    for group in ("EPG", "PEN", "D7"):
        with lesioned(model, [group]):
            r = model.run(x, model.init_state(2))["r"]
            assert torch.all(r[:, :, conn.groups[group]] == 0), group
        r_after = model.run(x, model.init_state(2))["r"]
        assert torch.count_nonzero(r_after[:, :, conn.groups[group]]) > 0  # restore() worked


def test_inv6_epg_lesion_heading_equals_bias(model):
    with lesioned(model, ["EPG"]):
        h = model.run(_inputs(), model.init_state(2))["heading"]
    bias = model.heading.linear.bias.detach()
    assert torch.allclose(h, bias.expand_as(h), atol=0, rtol=0)


def test_inv6_lesion_toggles_between_steps(model, conn):
    x = _inputs(t=6)
    v = model.init_state(2)
    epg = conn.groups["EPG"]
    for t in range(6):
        if t == 3:
            model.lesion(["EPG"])
        v, r = model.core_step(v, x[:, t])
        if t >= 3:
            assert torch.all(r[:, epg] == 0)
        else:
            assert torch.count_nonzero(r[:, epg]) > 0
    model.restore()
    assert torch.all(model.core.keep == 1)


def test_inv7_conditions_share_neurons_and_groups(conn):
    conds = {mt: apply_mask_type(conn, mt, seed=1) for mt in ("real", "rewire", "er")}
    for c in conds.values():
        assert c.n == conn.n
        assert np.array_equal(c.types, conn.types)
        assert c.groups.keys() == conn.groups.keys()
        for g in conn.groups:
            assert np.array_equal(c.groups[g], conn.groups[g])
        assert c.n_edges == conn.n_edges
    assert not np.array_equal(conds["rewire"].M, conn.M)  # only the edges differ


def test_inv8_rewire_preserves_degrees(conn):
    M2, stats = directed_rewire(conn.M, swaps_per_edge=10, seed=3)
    assert np.array_equal(M2.sum(0), conn.M.sum(0))  # in-degree
    assert np.array_equal(M2.sum(1), conn.M.sum(1))  # out-degree
    assert not M2.diagonal().any()  # no self-loops
    assert M2.sum() == conn.M.sum() == stats.n_edges  # same count (bool matrix => no duplicates)
    assert stats.swaps_done == stats.swaps_target
    assert stats.surviving_fraction < 1.0


def test_inv9_heading_target_relative_to_start():
    yaw0 = 3.0
    yaw = np.array([3.0, 3.1, -3.1, 2.9, 3.0 + 2 * np.pi])
    rel = relative_yaw(yaw, yaw0)
    expected = wrap_angle(yaw - yaw0)
    assert np.allclose(rel, expected)
    assert np.all((rel >= -np.pi) & (rel < np.pi))
    assert np.isclose(rel[0], 0.0) and np.isclose(rel[-1], 0.0, atol=1e-9)
    assert np.isclose(rel[2], (-3.1 - 3.0) + 2 * np.pi)  # wraps across +/-pi
    tgt = heading_target(rel)
    assert tgt.shape == (5, 2)
    assert np.allclose(tgt[:, 0], np.sin(rel)) and np.allclose(tgt[:, 1], np.cos(rel))
    assert np.allclose(tgt[0], [0.0, 1.0])


def test_inv10_all_floating_tensors_fp32(model):
    """Every floating-point parameter and buffer is fp32 (bool masks / long index buffers are not floats)."""
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32, name
    for name, b in model.named_buffers():
        if b.is_floating_point():
            assert b.dtype == torch.float32, name
        else:
            assert b.dtype in (torch.bool, torch.long), name
    out = model.run(_inputs().half(), model.init_state(2))  # fp16 cache input is upcast
    assert out["r"].dtype == torch.float32 and out["action_mean"].dtype == torch.float32
