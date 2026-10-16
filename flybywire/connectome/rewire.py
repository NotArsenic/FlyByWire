"""Directed, degree-preserving rewiring by repeated two-edge swaps.

Pick edges ``a->b`` and ``c->d`` with four distinct endpoints; replace them with ``a->d`` and
``c->b`` iff neither new edge already exists (and neither is a self-loop, which distinctness
already guarantees). Every node keeps its exact in-degree and out-degree, the edge count is
unchanged, and no self-loops or duplicate edges are introduced.

``networkx.double_edge_swap`` is deliberately NOT used: it is undirected-only and would destroy
the in/out-degree split (see docs/DECISIONS.md, D2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewireStats:
    n_edges: int
    swaps_target: int
    swaps_done: int
    attempts: int
    surviving_fraction: float  # fraction of original edges still present


def directed_rewire(
    M: np.ndarray, swaps_per_edge: float = 10.0, seed: int = 0, max_attempts_factor: float = 100.0
) -> tuple[np.ndarray, RewireStats]:
    """Return a rewired copy of boolean adjacency ``M`` (``M[pre, post]``) and statistics.

    Performs ``round(swaps_per_edge * |E|)`` successful swaps (seeded). Gives up after
    ``max_attempts_factor * target`` attempts and reports the number actually done.
    """
    M = np.asarray(M, dtype=bool)
    n = M.shape[0]
    if M.diagonal().any():
        raise ValueError("input graph has self-loops")
    src, dst = np.nonzero(M)
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    n_edges = len(src)
    target = int(round(swaps_per_edge * n_edges))
    stats = RewireStats(n_edges, target, 0, 0, 1.0)
    if n_edges < 2 or target == 0:
        return M.copy(), stats

    rng = np.random.default_rng(seed)
    edges = set((src * n + dst).tolist())
    max_attempts = int(max_attempts_factor * target)
    done = attempts = 0
    batch = 4096
    while done < target and attempts < max_attempts:
        ii = rng.integers(0, n_edges, size=batch)
        jj = rng.integers(0, n_edges, size=batch)
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            attempts += 1
            a, b, c, d = src[i], dst[i], src[j], dst[j]
            if a == c or a == d or b == c or b == d:  # need four distinct endpoints
                continue
            ad, cb = a * n + d, c * n + b
            if ad in edges or cb in edges:
                continue
            edges.discard(a * n + b)
            edges.discard(c * n + d)
            edges.add(ad)
            edges.add(cb)
            dst[i], dst[j] = d, b
            done += 1
            if done >= target or attempts >= max_attempts:
                break

    out = np.zeros_like(M)
    out[src, dst] = True
    stats.swaps_done = done
    stats.attempts = attempts
    stats.surviving_fraction = float((out & M).sum() / n_edges)
    return out, stats


def erdos_renyi_like(M: np.ndarray, seed: int = 0) -> np.ndarray:
    """Directed random graph with the same N and edge count, uniform edges, no self-loops."""
    M = np.asarray(M, dtype=bool)
    n = M.shape[0]
    n_edges = int(M.sum())
    rng = np.random.default_rng(seed)
    off = rng.choice(n * (n - 1), size=n_edges, replace=False)
    rows = off // (n - 1)
    cols = off % (n - 1)
    cols = cols + (cols >= rows)  # skip the diagonal
    out = np.zeros_like(M)
    out[rows, cols] = True
    return out
