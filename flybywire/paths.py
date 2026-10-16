"""Repo paths and safe flyvis import.

flyvis has two import-time side effects we neutralise (docs/DECISIONS.md D11, D12):

1. It resolves ``FLYVIS_ROOT_DIR`` relative to the *current working directory* (and mkdirs it).
   :func:`setup_flyvis_env` sets an absolute path first.
2. It calls ``torch.set_default_device("cuda")`` globally. :func:`import_flyvis` restores the
   previous default device; flyvis calls must then run inside ``with torch.device(dev):`` because
   flyvis allocates tensors without an explicit device.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_flyvis_env() -> Path:
    try:
        from dotenv import dotenv_values

        env_file = dotenv_values(REPO_ROOT / ".env")
    except ImportError:
        env_file = {}
    raw = os.environ.get("FLYVIS_ROOT_DIR") or env_file.get("FLYVIS_ROOT_DIR") or "./data/flyvis"
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
