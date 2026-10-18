"""Circular statistics, tuning fits and ring geometry used by the topography eval."""

from __future__ import annotations

import numpy as np
import pytest

from flybywire.eval.topography import (
    abs_circ_corr,
    alignment,
    alignment_score,
    bootstrap_tuning,
    circ_corr,
    cos_over_sin_se,
    decoder_angle,
    episode_moments,
    fit_tuning,
    harmonic_r2,
    permutation_test,
    phi_spread,
    ring_angles,
    ring_frame,
    stack_moments,
    val_split,
    wrap,
)

RNG = np.random.default_rng(0)


def test_alignment_ignores_rotation_and_reflection():
    theta = RNG.uniform(-np.pi, np.pi, 40)
    score, sign, offset = alignment(theta + 0.7, theta)
    assert score == pytest.approx(1.0) and sign == 1 and offset == pytest.approx(0.7)
    score, sign, offset = alignment(-theta + 0.3, theta)
    assert score == pytest.approx(1.0) and sign == -1 and offset == pytest.approx(0.3)
    assert alignment(RNG.uniform(-np.pi, np.pi, 40), theta)[0] < 0.5


def test_circ_corr_signs():
    theta = RNG.uniform(-np.pi, np.pi, 40)
    assert circ_corr(theta + 1.0, theta) == pytest.approx(1.0)
    assert circ_corr(-theta, theta) == pytest.approx(-1.0)
    assert abs_circ_corr(-theta, theta) == pytest.approx(1.0)


def test_permutation_test_separates_aligned_from_random():
    theta = RNG.uniform(-np.pi, np.pi, 30)
    aligned = permutation_test(alignment_score, theta + 0.4 + 0.1 * RNG.standard_normal(30), theta, n_perm=500)
    random = permutation_test(alignment_score, RNG.uniform(-np.pi, np.pi, 30), theta, n_perm=500)
    assert aligned["p"] < 0.01 < random["p"]


def test_harmonic_r2_is_invariant_to_rotation_and_reflection():
    theta = RNG.uniform(-np.pi, np.pi, 40)
    y = 2 * np.cos(theta + 0.4) + 0.05 * RNG.standard_normal(40)
    assert harmonic_r2(y, theta) > 0.98
    assert harmonic_r2(y, -theta + 1.3) == pytest.approx(harmonic_r2(y, theta))
    assert harmonic_r2(RNG.standard_normal(40), theta) < 0.3


def _moments(headings, phi, noise, n_episodes=20, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_episodes):
        h = headings * rng.uniform(-1, 1, 300)
        rate = 1.0 + 0.5 * np.cos(h[:, None] - phi[None]) + noise * rng.standard_normal((300, len(phi)))
        eps.append(episode_moments(rate, h))
    return stack_moments(eps)


def test_tuning_fit_recovers_phi_and_small_headings_blur_it():
    phi = np.array([-2.0, 0.3, 1.8])
    wide = _moments(np.pi, phi, noise=0.05)
    fit = fit_tuning(wide)
    np.testing.assert_allclose(wrap(fit["phi"] - phi), 0, atol=0.05)
    np.testing.assert_allclose(fit["amplitude"], 0.5, atol=0.05)
    narrow = _moments(np.deg2rad(15), phi, noise=0.05)
    spread_wide = phi_spread(fit["phi"], bootstrap_tuning(wide, 100)["phi"])
    fit_narrow = fit_tuning(narrow)
    spread_narrow = phi_spread(fit_narrow["phi"], bootstrap_tuning(narrow, 100)["phi"])
    assert (spread_narrow > 3 * spread_wide).all()
    assert cos_over_sin_se(narrow) > 3 * cos_over_sin_se(wide)


def test_decoder_angle_uses_sin_then_cos_rows():
    w = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(decoder_angle(w), [np.pi / 2, 0.0, -np.pi / 2])


def test_val_split_matches_bc():
    idx = val_split(300, 0)
    np.testing.assert_array_equal(idx, np.random.default_rng(0).permutation(300)[:30])
    assert len(val_split(5, 0)) == 0


def _tilted_ring(n=24, tilt=0.5, axes_um=(30.0, 18.0)):
    t = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(tilt), np.sin(tilt)
    ex, ey, ez = np.array([1.0, 0, 0]), np.array([0, c, s]), np.array([0, -s, c])
    center = np.array([500.0, 180.0, 90.0])
    pts = center + axes_um[0] * np.cos(t)[:, None] * ex + axes_um[1] * np.sin(t)[:, None] * ey
    pts += 1.5 * RNG.standard_normal(n)[:, None] * ez
    return pts, center, ez, t


def test_ring_frame_normal_and_even_angles():
    pts, center, ez, t = _tilted_ring()
    axes, sd = ring_frame(pts, center)
    assert abs(axes[2] @ ez) > 0.99
    assert abs(axes[0] @ axes[1]) < 1e-9 and axes[0][0] > 0 and axes[1][1] < 0
    ang = np.sort(ring_angles(pts, center, axes, sd))
    steps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
    assert steps.std() < 0.2 * steps.mean()
    r = np.hypot(*((pts - center) @ axes[:2].T / sd).T)
    assert r.std() / r.mean() < 0.1  # whitening turned the ellipse into a circle


def test_ring_angle_zero_is_dorsal_and_positive_is_plus_x():
    pts, center, _, _ = _tilted_ring()
    axes, sd = ring_frame(pts, center)
    dorsal, lateral = center + 10 * axes[1], center + 10 * axes[0]
    a = ring_angles(np.stack([dorsal, lateral]), center, axes, sd)
    assert a[0] == pytest.approx(0.0, abs=1e-9) and a[1] > 0
