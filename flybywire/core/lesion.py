"""Lesion helpers: silence cell-type groups by zeroing their rates (``keep = 0``).

A lesioned neuron's rate is exactly 0 at every step (invariant 6), so it influences nothing
downstream through ``r @ (W * M)`` or any head. Lesions can be toggled between any two steps.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager

import numpy as np
import torch


def keep_mask(n: int, groups: Mapping[str, np.ndarray], names: Iterable[str]) -> torch.Tensor:
    keep = torch.ones(n)
    for name in names:
        if name not in groups:
            raise KeyError(f"unknown lesion group {name!r}; available: {sorted(groups)}")
        keep[torch.as_tensor(np.asarray(groups[name]), dtype=torch.long)] = 0.0
    return keep


@contextmanager
def lesioned(model, names: Iterable[str]):
    """``with lesioned(model, ["EPG"]): ...`` — lesion inside the block, restore afterwards."""
    prev = model.core.keep.clone()
    model.lesion(list(names))
    try:
        yield model
    finally:
        model.core.keep.copy_(prev)
