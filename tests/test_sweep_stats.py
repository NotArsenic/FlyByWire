"""Exact two-sample permutation test of the seed comparison."""

from __future__ import annotations

import importlib.util
import itertools
from math import comb
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_stats.py"


@pytest.fixture(scope="module")
def stats():
    spec = importlib.util.spec_from_file_location("sweep_stats", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matches_brute_force_on_a_small_case(stats):
    real, rewire = [0.9, 0.5, 0.7], [0.1, 0.6, 0.2, 0.3]
    x = np.array(real + rewire)
    obs = abs(np.mean(real) - np.mean(rewire))
    diffs = [abs(x[list(c)].mean() - np.delete(x, list(c)).mean()) for c in itertools.combinations(range(7), 3)]
    out = stats.exact_permutation(real, rewire)
    assert out["n_relabelings"] == comb(7, 3)
    assert out["p"] == pytest.approx(np.mean([d >= obs - 1e-12 for d in diffs]))
    assert out["difference"] == pytest.approx(np.mean(real) - np.mean(rewire))


def test_perfect_separation_gives_the_smallest_possible_p(stats):
    out = stats.exact_permutation([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
    assert out["p"] == pytest.approx(2 / comb(10, 5))


def test_identical_groups_are_not_significant(stats):
    assert stats.exact_permutation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])["p"] == 1.0
