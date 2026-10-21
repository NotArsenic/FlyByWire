"""Lesion helpers: silence cell-type groups by zeroing their rates (keep = 0)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager

import numpy as np
import torch


def keep_mask(
    n: int, groups: Mapping[str, np.ndarray], names: Iterable[str]
) -> torch.Tensor:
    keep = torch.ones(n)
    for name in names:
        if name not in groups:
            raise KeyError(
                f"unknown lesion group {name!r}; available: {sorted(groups)}"
            )
        keep[torch.as_tensor(np.asarray(groups[name]), dtype=torch.long)] = 0.0
    return keep


@contextmanager
def lesioned(model, names: Iterable[str]):
    """Context manager to temporarily lesion a model."""
    prev = model.core.keep.clone()
    model.lesion(list(names))
    try:
        yield model
    finally:
        model.core.keep.copy_(prev)
