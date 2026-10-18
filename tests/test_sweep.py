"""Sweep job grid and config validation of the heading decoder option."""

from __future__ import annotations

from pathlib import Path

import pytest

from flybywire.config import ConfigError, load_config
from flybywire.train.sweep import build_jobs


def test_jobs_cover_mask_decoder_seed_grid():
    cfg = load_config(overrides=["sweep.decoders=[learned,anatomical]", "sweep.seeds=[0,1]"])
    jobs = build_jobs(cfg, "data/cache_turns", Path("runs/x"), [], None)
    assert len(jobs) == 8
    j = next(j for j in jobs if j["condition"] == "rewire" and j["decoder"] == "anatomical" and j["seed"] == 1)
    assert j["out"] == Path("runs/x/rewire_anatomical_s1")
    assert {"connectome.mask_type=rewire", "heads.decoder=anatomical", "seed=1"} <= set(j["cmd"])


def test_unknown_decoder_is_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["heads.decoder=ring"])
    with pytest.raises(ConfigError):
        load_config(overrides=["sweep.decoders=[learned,ring]"])
