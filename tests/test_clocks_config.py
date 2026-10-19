"""Clock divisibility, decision alignment, and config validation."""

from __future__ import annotations

import numpy as np
import pytest

from flybywire.clocks import Clocks
from flybywire.config import ConfigError, config_hash, load_config, obstacle_texture, validate_config


def test_default_ratios():
    c = Clocks()
    assert (c.physics_per_sensory, c.sensory_per_decision, c.physics_per_decision) == (2, 4, 8)
    assert c.dt_sensory == pytest.approx(0.01)


@pytest.mark.parametrize("rates", [(200, 75, 25), (200, 100, 30), (0, 100, 25), (200, 100, 25.0)])
def test_bad_rates_raise(rates):
    with pytest.raises(ValueError):
        Clocks(*rates)


def test_decision_alignment():
    c = Clocks()
    k = 250
    t = c.n_core_steps(k)
    assert t == 1000
    dec = c.decision_core_steps(k)
    assert dec[0] == 3 and dec[-1] == t - 1 and len(dec) == k
    assert np.all(np.diff(dec) == 4)
    assert all(c.is_decision_step(int(s)) for s in dec)
    assert sum(c.is_decision_step(s) for s in range(t)) == k
    for kk in (0, 1, 17):
        assert c.decision_core_step(kk) == dec[kk]
        assert c.decision_of_core_step(c.decision_core_step(kk)) == kk
        # action a_k is held while frames of the *next* window are rendered
        assert list(c.action_core_steps(kk)) == list(range(4 * (kk + 1), 4 * (kk + 2)))
        assert list(c.action_physics_steps(kk)) == list(range(8 * (kk + 1), 8 * (kk + 2)))
    # frame t is rendered after physics step 2t+1
    assert c.frame_physics_step(0) == 1 and c.frame_physics_step(3) == 7
    assert c.frame_time(0) == pytest.approx(0.01)
    assert c.n_decisions(10.0) == 250
    with pytest.raises(ValueError):
        c.n_decisions(10.01)


def test_windows_cover_and_respect_decisions():
    c = Clocks()
    w = c.windows(1000, 128)
    assert w[0] == (0, 128) and w[-1] == (896, 1000)
    assert sum(e - s for s, e in w) == 1000
    assert all(s % 4 == 0 for s, _ in w)
    with pytest.raises(ValueError):
        c.windows(1000, 130)


def test_default_config_loads():
    cfg = load_config()
    assert cfg.clocks.sensory_hz == 100
    assert list(cfg.lptc.channels)[0] == "yaw"


def test_overrides_apply():
    cfg = load_config(overrides=["core.phi=relu", "connectome.mask_type=rewire"])
    assert cfg.core.phi == "relu" and cfg.connectome.mask_type == "rewire"


@pytest.mark.parametrize(
    "override",
    [
        "core.phi=tanh",
        "lptc.mode=huge",
        "connectome.mask_type=random",
        "clocks.sensory_hz=75",
        "bc.tbptt_core_steps=130",
        "core.alpha_init=1.5",
        "connectome.input_group=XX",
        "connectome.lesion_groups=[EPG,FOO]",
        "action.limits.yaw_rate=[1.0,-1.0]",
        "env.episode_seconds=10.01",
        "env.obstacle_texture=plaid",
    ],
)
def test_invalid_config_raises(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_config_without_obstacle_texture_is_valid_and_means_none():
    cfg = load_config()
    del cfg.env["obstacle_texture"]
    validate_config(cfg)
    assert obstacle_texture(cfg) == "none"


def test_config_hash_stable_and_sensitive():
    a = load_config()
    b = load_config()
    c = load_config(overrides=["vision.flyvis_model=flow/0000/001"])
    keys = ["vision.flyvis_model", "clocks.sensory_hz"]
    assert config_hash(a, keys) == config_hash(b, keys)
    assert config_hash(a, keys) != config_hash(c, keys)
