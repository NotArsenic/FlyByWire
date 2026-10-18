"""Bump share on synthetic profiles, the resting-rate helper, and the synthetic-bump widths."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flybywire.eval.ring import bump, rate_floor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bump_sharpness_check.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("bump_sharpness_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_share_ignores_a_uniform_floor():
    centres = (np.arange(16) + 0.5) * 2 * np.pi / 16
    profile = np.exp(2 * np.cos(centres - 1.0))
    assert bump(profile + 0.69)["fraction"] == pytest.approx(bump(profile)["fraction"], abs=1e-12)
    assert bump(profile + 50.0)["fraction"] == pytest.approx(bump(profile)["fraction"], abs=1e-12)


def test_a_single_hot_wedge_spreads_over_all_harmonics():
    spike = np.zeros(16)
    spike[5] = 1.0
    assert bump(spike)["fraction"] == pytest.approx(1 / 8)


def test_rate_floor_uses_frames_after_the_washout():
    a = np.zeros((10, 3))
    a[:4] = 100.0
    b = np.arange(10.0)[:, None] * np.ones((1, 3))
    floor = rate_floor([a, b], washout=4, q=50)
    assert floor.shape == (3,)
    assert floor == pytest.approx(np.median(np.concatenate([a[4:], b[4:]]), axis=0))


@pytest.mark.parametrize("name", ["gaussian", "von_mises"])
def test_bump_shapes_halve_at_half_the_fwhm(check, name):
    shape = check.SHAPES[name]
    for fwhm in (30, 90, 150):
        half = np.radians(fwhm / 2)
        assert shape(np.array(half), fwhm) / shape(np.array(0.0), fwhm) == pytest.approx(0.5, abs=0.01)


def test_share_of_a_53_degree_bump_is_about_a_half(check):
    assert check.ideal_share(check.gaussian, 53).mean() == pytest.approx(0.49, abs=0.01)
    assert check.ideal_share(check.gaussian, 53, floor=0.69).mean() == pytest.approx(
        check.ideal_share(check.gaussian, 53).mean(), abs=1e-12)
