"""Flow templates, LPTC, Farneback backend, metrics and ring layouts (CPU), plus flyvis GPU tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybywire.eval.metrics import angular_error_deg, heading_r2
from flybywire.eval.ring import make_layout, ring_activity
from flybywire.vision.flowfields import (
    CHANNELS,
    channel_flow,
    hexal_flow_templates,
    hexal_normalized_coords,
    pixel_grid_focal,
)
from flybywire.vision.lptc import LPTCPool, build_templates


def test_hexal_coords_match_boxeye_geometry():
    xy = hexal_normalized_coords(15, 13)
    assert xy.shape == (721, 2)
    assert np.abs(xy).max() < 1.0
    # centre hexal (u = v = 0, index 360) sits on pixel 195 of the odd 391-px grid = exact image centre
    assert np.allclose(xy[360], [0.0, 0.0], atol=1e-12)
    # neighbour v = +1 is one kernel (13 px) to the right
    assert np.isclose(xy[361, 0], 13 / 391 * 2) and np.isclose(xy[361, 1], 13 * 0.5 / 391 * 2)


def test_yaw_flow_is_rightward_and_forward_expands():
    xy = pixel_grid_focal(9, 110.0)
    yaw = channel_flow(xy, "yaw", tilt_deg=0.0)
    assert np.all(yaw[:, 0] > 0)  # turning left (+yaw) moves the scene right
    fwd = channel_flow(xy, "forward", tilt_deg=0.0)
    radial = (fwd * xy).sum(1)
    assert np.all(radial >= -1e-12)  # expansion away from the focus
    ll, lr = channel_flow(xy, "loom_left"), channel_flow(xy, "loom_right")
    assert np.all(ll[xy[:, 0] > 0] == 0) and np.all(lr[xy[:, 0] < 0] == 0)


def test_templates_unit_norm_and_lptc_modes():
    flows = hexal_flow_templates(tilt_deg=10.0)
    assert flows.shape == (8, 721, 2)
    ang = np.deg2rad(np.arange(8) * 45.0)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    w = build_templates(flows, dirs)
    assert np.allclose(np.linalg.norm(w.reshape(8, -1), axis=1), 1.0)
    assert LPTCPool(w, mode="gains").n_trainable() == 64
    assert LPTCPool(w, mode="frozen").n_trainable() == 0
    assert LPTCPool(w, mode="full").n_trainable() == 8 * 8 * 721
    lptc = LPTCPool(w, baseline=np.full(8, 0.1), mode="gains")
    x = torch.zeros(2, 8, 721)  # relu(0) - 0.1 everywhere
    out = lptc(x)
    assert out.shape == (2, 8)
    assert torch.allclose(out, -0.1 * torch.as_tensor(w.sum(axis=(1, 2)), dtype=torch.float32).expand(2, 8),
                          atol=1e-5)


def test_farneback_recovers_yaw_sign():
    from flybywire.vision.farneback import FarnebackBackend
    from flybywire.vision.stimuli import panorama_texture, view_from_panorama

    fb = FarnebackBackend(160, 110.0, tilt_deg=0.0, sensory_hz=100)
    pano = panorama_texture(160, 110.0, seed=1)
    out = []
    for t in range(4):
        out.append(fb.step(view_from_panorama(pano, 0.5 * t / 100, 160, 110.0)[None])[0])
    yaw_idx = CHANNELS.index("yaw")
    assert np.all(np.asarray(out)[1:, yaw_idx] > 0)
    assert np.all(np.asarray(out)[0] == 0)  # first frame after reset has no flow


def test_metrics():
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, 500)
    tgt = np.stack([np.sin(th), np.cos(th)], 1)
    assert heading_r2(tgt, tgt) == pytest.approx(1.0)
    assert heading_r2(np.tile(tgt.mean(0), (500, 1)), tgt) == pytest.approx(0.0, abs=1e-12)
    assert angular_error_deg(tgt, tgt) == pytest.approx(0.0, abs=1e-9)
    rot = np.stack([np.sin(th + np.deg2rad(10)), np.cos(th + np.deg2rad(10))], 1)
    assert angular_error_deg(rot, tgt) == pytest.approx(10.0, abs=1e-6)
    mask = np.zeros(500, bool)
    mask[:10] = True
    assert heading_r2(tgt, tgt, mask) == pytest.approx(1.0)


def test_ring_layouts(toy):
    epg = toy.groups["EPG"]
    lay = make_layout("auto", toy.instances[epg])
    assert lay.mode == "anatomical" and lay.parsed_fraction == 1.0
    assert sorted(lay.wedge_of_neuron.tolist()) == list(range(16))
    assert lay.wedge_of_neuron.tolist() == toy.meta["wedge"][epg].tolist()
    ang = np.deg2rad(np.arange(16) * 22.5 + 1)
    w = np.stack([np.sin(ang), np.cos(ang)])
    lay2 = make_layout("auto", np.array(["EPG"] * 16), decoder_weight=w)
    assert lay2.mode == "tuning" and lay2.wedge_of_neuron.tolist() == list(range(16))
    assert "tuning" in lay2.label
    act = ring_activity(np.eye(16), lay2)
    assert np.allclose(act, np.eye(16))


def test_ring_plot_labels_layout(toy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from flybywire.eval.ring import plot_ring

    lay = make_layout("anatomical", toy.instances[toy.groups["EPG"]])
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_ring(ax, np.arange(16.0), lay, heading_rad=0.3)
    assert "anatomical" in ax.get_title()
    plt.close(fig)


@pytest.mark.flyvis
@pytest.mark.gpu
def test_stepper_state_carry_and_per_env_reset():
    from flybywire.vision.boxeye import BoxEye
    from flybywire.vision.flyvis_stepper import FlyvisStepper
    from flybywire.vision.stimuli import panorama_texture, view_from_panorama

    eye = BoxEye(device="cuda")
    st = FlyvisStepper(device="cuda", steady_state_seconds=0.2)
    pano = panorama_texture(seed=0)
    frames = np.stack([view_from_panorama(pano, t / 100) for t in range(20)])
    hexals = eye(torch.from_numpy(frames).cuda())
    assert hexals.shape == (20, 721) and torch.get_default_device().type == "cpu"
    x = hexals[None].expand(2, -1, -1)
    ref = st.full_sequence_reference(x)
    out = st.run_sequence(x)
    assert (out - ref).abs().max() / ref.abs().max() < 1e-5
    # per-env reset touches only that env
    st.reset(2)
    for t in range(5):
        st.step(x[:, t])
    before = st.state.clone()
    st.reset(env_ids=[1])
    assert torch.equal(st.state[0], before[0])
    assert torch.equal(st.state[1], st.steady_state(1)[0])
