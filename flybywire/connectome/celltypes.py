"""Connectome container, cell-type group matching, and npz I/O.

Convention (used everywhere): ``M[i, j] = True`` iff **presynaptic i -> postsynaptic j**.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass
class Connectome:
    M: np.ndarray  # (N, N) bool, M[pre, post]
    neuron_ids: np.ndarray  # (N,) int64
    types: np.ndarray  # (N,) str, cell-type label
    instances: np.ndarray  # (N,) str, instance / name
    groups: dict[str, np.ndarray]  # group name -> sorted int64 indices
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=bool)
        n = self.M.shape[0]
        if self.M.shape != (n, n):
            raise ValueError(f"M must be square, got {self.M.shape}")
        for name, arr in (("neuron_ids", self.neuron_ids), ("types", self.types), ("instances", self.instances)):
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected N={n}")
        self.groups = {k: np.asarray(sorted(v), dtype=np.int64) for k, v in self.groups.items()}

    @property
    def n(self) -> int:
        return self.M.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.M.sum())

    @property
    def density(self) -> float:
        n = self.n
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    def in_degree(self) -> np.ndarray:
        return self.M.sum(axis=0)

    def out_degree(self) -> np.ndarray:
        return self.M.sum(axis=1)

    def with_mask(self, M: np.ndarray, **meta) -> Connectome:
        """Same neurons and groups, different edges."""
        return replace(self, M=np.asarray(M, dtype=bool), groups=dict(self.groups), meta={**self.meta, **meta})

    def summary_table(self) -> str:
        lines = [f"{'group':<8} {'n':>6}  types"]
        for g, idx in self.groups.items():
            tnames = sorted(set(self.types[idx].tolist()))
            shown = ", ".join(tnames[:8]) + (" ..." if len(tnames) > 8 else "")
            lines.append(f"{g:<8} {len(idx):>6}  {shown}")
        lines.append(f"N={self.n}  edges={self.n_edges}  density={self.density:.4f}  source={self.source}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "M": self.M,
            "neuron_ids": np.asarray(self.neuron_ids, dtype=np.int64),
            "types": np.asarray(self.types, dtype=str),
            "instances": np.asarray(self.instances, dtype=str),
            "group_names": np.asarray(list(self.groups), dtype=str),
            "source": np.asarray(self.source),
        }
        for g, idx in self.groups.items():
            arrays[f"group__{g}"] = idx
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Connectome:
        with np.load(path, allow_pickle=False) as z:
            groups = {str(g): z[f"group__{g}"] for g in z["group_names"]}
            return cls(
                M=z["M"],
                neuron_ids=z["neuron_ids"],
                types=z["types"],
                instances=z["instances"],
                groups=groups,
                source=str(z["source"]),
            )


def match_groups(types: Sequence[str], patterns: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """Assign neurons to groups by fnmatch patterns on their type label.

    Raises if a neuron matches more than one group (groups must be disjoint so lesions are clean).
    """
    types = [str(t) for t in types]
    owner: dict[int, str] = {}
    groups: dict[str, list[int]] = {g: [] for g in patterns}
    for g, pats in patterns.items():
        for i, t in enumerate(types):
            if any(fnmatch.fnmatchcase(t, p) for p in pats):
                if i in owner and owner[i] != g:
                    raise ValueError(f"neuron {i} (type {t!r}) matches both {owner[i]!r} and {g!r}")
                owner[i] = g
                groups[g].append(i)
    return {g: np.asarray(v, dtype=np.int64) for g, v in groups.items()}

