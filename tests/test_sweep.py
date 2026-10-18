"""Sweep job grid and config validation of the heading decoder option."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flybywire.config import ConfigError, load_config
from flybywire.train.sweep import build_jobs, summarize


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


def test_summary_keeps_rows_of_an_earlier_sweep(tmp_path):
    def job(decoder, seed):
        out = tmp_path / f"real_{decoder}_s{seed}"
        out.mkdir(exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"heading_r2": 0.9}))
        return {"condition": "real", "decoder": decoder, "seed": seed, "out": out}

    summarize([job("learned", 0)], tmp_path)
    summarize([job("anatomical", 0), job("anatomical", 1)], tmp_path)
    rows = json.loads((tmp_path / "summary.json").read_text())["runs"]
    assert sorted((r["decoder"], r["seed"]) for r in rows) == [("anatomical", 0), ("anatomical", 1), ("learned", 0)]
    summarize([job("anatomical", 1)], tmp_path)
    assert len(json.loads((tmp_path / "summary.json").read_text())["runs"]) == 3
