"""Failure classification and approach statistics of scripts/failure_breakdown.py, on made-up flights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flybywire.eval.metrics import wilson_interval

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "failure_breakdown.py"
POLE = np.array([[0, 5.0, 0.0, 0.2, 0.2]])  # cylinder on the axis, front face at x = 4.8


@pytest.fixture(scope="module")
def fb():
    spec = importlib.util.spec_from_file_location("failure_breakdown", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(100, 200)
    assert (lo, hi) == pytest.approx((0.4314, 0.5686), abs=1e-3)
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0713, abs=1e-3)


@pytest.mark.parametrize(
    "pos, kind",
    [([4.7, 0.05, 1.0], "obstacle"), ([2.0, 1.45, 1.0], "side_wall"), ([2.0, 0.0, 0.03], "floor"),
     ([2.0, 0.0, 2.45], "ceiling")],
)
def test_end_is_the_nearest_surface(fb, pos, kind):
    assert fb.end_kind("collision", np.array(pos), POLE, 1.5, 2.5)[0] == kind


def test_non_collisions_keep_their_reason(fb):
    assert fb.end_kind("time", np.zeros(3), POLE, 1.5, 2.5) == ("time", None)


def test_sustained_needs_a_run(fb):
    assert fb.sustained(np.array([1, 1, 0, 1, 1, 1], bool), 3)
    assert not fb.sustained(np.array([1, 1, 0, 1, 1, 0], bool), 3)


def flight(y_of_x, x_end, k=60):
    x = np.linspace(1.0, x_end, k)
    pos = np.stack([x, y_of_x(x), np.full(k, 1.2)], axis=1).astype(np.float32)
    vel = np.zeros((k, 3), np.float32)
    vel[:, 0] = 1.0
    return pos, vel


def test_approach_reports_clearance_and_onset(fb):
    pos, vel = flight(lambda x: np.where(x > 3.5, 0.6 * (x - 3.5) / 1.3, 0.0), 6.0)
    (row,) = fb.approaches(pos, vel, POLE, len(pos), hit=None)
    assert row["on_course"] and not row["hit"] and row["rank"] == 1
    assert row["clearance_3.0"] == pytest.approx(-0.2, abs=1e-3)  # on the axis, centre to surface
    assert row["clearance_0.5"] > 0.1
    assert 0.8 < row["onset_m"] < 1.1  # 15 cm sideways at x = 3.5 + 0.15 / 0.46


def test_straight_into_the_pole_is_a_hit_with_no_onset(fb):
    pos, vel = flight(lambda x: np.zeros_like(x), 4.75)
    (row,) = fb.approaches(pos, vel, POLE, len(pos), hit=0)
    assert row["hit"] and row["onset_m"] is None


def test_summary_counts_and_clear_rate(fb):
    n = 4
    pos, vel = flight(lambda x: np.zeros_like(x), 4.75)
    fl = {
        "seeds": np.arange(n), "reasons": np.array(["collision", "time", "collision", "time"]),
        "length": np.full(n, len(pos)), "obstacles": np.stack([POLE] * n),
        "pos": np.stack([pos] * n), "vel": np.stack([vel] * n),
        "rel_yaw": np.zeros((n, len(pos)), np.float32), "yaw_rate": np.zeros((n, len(pos)), np.float32),
        "heading_err": np.zeros((n, len(pos)), np.float32),
    }
    fl["heading_err"][0, -20:] = 45.0
    rows = fb.episode_rows(fl, 1.5, 2.5, drift_deg=30, drift_s=0.5, tail_s=3.0)
    assert [r["drift"] for r in rows] == [True, False, False, False]
    s = fb.summarize_rows(rows)
    assert s["success"] == 0.5 and s["ends"]["obstacle"] == 2 and s["ends"]["time"] == 2
    assert s["drift_rate"]["failures"] == 0.5 and s["drift_rate"]["successes"] == 0.0
    assert s["hit_rank"] == {1: 2} and s["on_course"]["n"] == 4
