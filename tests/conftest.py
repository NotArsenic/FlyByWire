"""Shared fixtures. GPU / flyvis / FlyWire-graph tests are skipped (with a reason) when unavailable."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybywire.config import load_config
from flybywire.connectome.masks import load_base_connectome
from flybywire.core.model import build_model
from flybywire.vision.lptc import LPTCPool


def _flyvis_available() -> tuple[bool, str]:
    try:
        from flybywire.paths import setup_flyvis_env

        root = setup_flyvis_env()
        import importlib.util

        if importlib.util.find_spec("flyvis") is None:
            return False, "flyvis not installed"
        if not (root / "results" / "flow" / "0000" / "000").exists():
            return False, f"flyvis pretrained models missing under {root}"
        return True, ""
    except Exception as e:
        return False, f"flyvis unavailable: {e}"


def pytest_collection_modifyitems(config, items):
    has_gpu = torch.cuda.is_available()
    fv_ok, fv_reason = _flyvis_available()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="no CUDA GPU available"))
        if "flyvis" in item.keywords and not fv_ok:
            item.add_marker(pytest.mark.skip(reason=fv_reason))


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def conn(cfg):
    try:
        return load_base_connectome(cfg)
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture()
def random_lptc():
    rng = np.random.default_rng(0)
    return LPTCPool(rng.standard_normal((8, 8, 721)).astype(np.float32) / 50, mode="gains")


@pytest.fixture()
def model(cfg, conn, random_lptc):
    torch.manual_seed(0)
    m = build_model(cfg, conn=conn, lptc=random_lptc, verbose=False)
    # make dynamics non-trivial: random biases so rates differ across neurons
    with torch.no_grad():
        m.core.b.normal_(0, 0.5)
    return m
