"""Mask conditions: ``real``, ``rewire`` (directed degree-preserving) and ``er`` (Erdős–Rényi).

All conditions share the same N, neuron order and cell-type index sets; only the edges differ
(invariant 7).
"""

from __future__ import annotations

import logging

from flybywire.connectome.celltypes import Connectome
from flybywire.connectome.rewire import directed_rewire, erdos_renyi_like

log = logging.getLogger(__name__)

MASK_TYPES = ("real", "rewire", "er")


def apply_mask_type(conn: Connectome, mask_type: str, seed: int = 0, swaps_per_edge: float = 10.0) -> Connectome:
    if mask_type == "real":
        return conn.with_mask(conn.M.copy(), mask_type="real")
    if mask_type == "rewire":
        M, stats = directed_rewire(conn.M, swaps_per_edge=swaps_per_edge, seed=seed)
        log.info(
            "rewire: %d/%d swaps (%d attempts), %.1f%% of original edges survive",
            stats.swaps_done,
            stats.swaps_target,
            stats.attempts,
            100 * stats.surviving_fraction,
        )
        if stats.swaps_done < stats.swaps_target:
            log.warning("rewire: only %d of %d target swaps succeeded", stats.swaps_done, stats.swaps_target)
        return conn.with_mask(M, mask_type="rewire", rewire_stats=stats.__dict__, rewire_seed=seed)
    if mask_type == "er":
        return conn.with_mask(erdos_renyi_like(conn.M, seed=seed), mask_type="er", er_seed=seed)
    raise ValueError(f"unknown mask_type {mask_type!r}; expected one of {MASK_TYPES}")


def load_base_connectome(cfg) -> Connectome:
    """The unmodified FlyWire graph from ``data/flywire/processed/cx_graph.npz``."""
    from flybywire.config import data_dir

    path = data_dir(cfg) / "flywire" / "processed" / "cx_graph.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the FlyWire tables into data/flywire/raw/ "
            "(see README 'Data') and run `python -m flybywire.connectome.flywire`."
        )
    conn = Connectome.load(path)
    missing = [g for g in cfg.connectome.groups if g not in conn.groups]
    if missing:
        raise ValueError(f"{path} lacks groups {missing}; re-run the loader with the current config")
    return conn


def build_connectome(cfg, seed: int | None = None) -> Connectome:
    """Base graph with the configured mask type applied.

    ``seed`` (default ``cfg.seed``) seeds the rewire / ER null model, so each training seed gets
    its own null graph.
    """
    conn = load_base_connectome(cfg)
    seed = int(cfg.seed if seed is None else seed)
    return apply_mask_type(
        conn, cfg.connectome.mask_type, seed=seed, swaps_per_edge=float(cfg.connectome.rewire_swaps_per_edge)
    )
