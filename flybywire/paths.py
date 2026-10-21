"""Repo paths and safe flyvis import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_flyvis_env() -> Path:
    env_file = dotenv_values(REPO_ROOT / ".env")
    raw = (
        os.environ.get("FLYVIS_ROOT_DIR")
        or env_file.get("FLYVIS_ROOT_DIR")
        or "./data/flyvis"
    )
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    os.environ["FLYVIS_ROOT_DIR"] = str(root)
    return root


def import_flyvis():
    """Import flyvis with an absolute root dir, restoring torch's default device afterwards."""
    import sys

    import torch

    if "flyvis" in sys.modules:
        return sys.modules["flyvis"]
    setup_flyvis_env()
    prev = torch.get_default_device()
    import flyvis

    torch.set_default_device(prev)
    return flyvis
