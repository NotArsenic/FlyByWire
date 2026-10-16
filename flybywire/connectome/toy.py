"""Toy central-complex graph for tests and smoke runs ONLY.

~96 neurons (32 ER, 16 EPG, 16 PEN, 16 PEG, 8 D7, 8 PFL3) wired with plausible ring motifs:

* ER -> EPG broad (random, p = 0.5)
* EPG -> PEN and EPG -> PEG, same wedge
* PEN -> EPG shifted +1 wedge (R side) or -1 wedge (L side)
* PEG -> EPG same wedge
* EPG -> D7 broad, D7 -> EPG broad (random, p = 0.5)
* EPG -> PFL3 (two adjacent wedges per PFL3 neuron)

Instance names follow a ``<TYPE>_<side><glomerulus>`` pattern so that the anatomical ring layout
has something to parse (wedge ``w`` <-> side ``R`` if ``w`` even else ``L``, glomerulus ``w // 2 + 1``;
see :func:`flybywire.connectome.celltypes.pb_glomerulus_to_wedge`).

**Never use this graph for reported results.** Every call logs a loud warning.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from flybywire.connectome.celltypes import Connectome

log = logging.getLogger(__name__)

N_WEDGES = 16
TOY_WARNING = (
    "\n" + "!" * 78 + "\n"
    "!!  USING THE TOY CONNECTOME — synthetic graph for tests / smoke runs only.  !!\n"
    "!!  Never report results obtained with connectome.source=toy.                 !!\n"
    + "!" * 78
)


def _wedge_name(w: int) -> str:
    side = "R" if w % 2 == 0 else "L"
    return f"{side}{w // 2 + 1}"


def generate_toy(seed: int = 0, p_broad: float = 0.5, warn: bool = True) -> Connectome:
    if warn:
        log.warning(TOY_WARNING)
        warnings.warn("toy connectome in use (tests / smoke only)", stacklevel=2)
    rng = np.random.default_rng(seed)

    types: list[str] = []
    instances: list[str] = []
    wedge: list[int] = []
    groups: dict[str, list[int]] = {g: [] for g in ("ER", "EPG", "PEN", "PEG", "D7", "PFL3")}

    def add(group: str, typ: str, inst: str, w: int) -> None:
        groups[group].append(len(types))
        types.append(typ)
        instances.append(inst)
        wedge.append(w)

    for i in range(32):  # ring neurons: 4 subtypes x 8, no wedge
        side = "R" if i % 2 == 0 else "L"
        add("ER", f"ER{i % 4 + 1}", f"ER{i % 4 + 1}_{side}_{i // 2:02d}", -1)
    for w in range(N_WEDGES):
        add("EPG", "EPG", f"EPG_{_wedge_name(w)}", w)
    for w in range(N_WEDGES):
        add("PEN", "PEN_a", f"PEN_a_{_wedge_name(w)}", w)
    for w in range(N_WEDGES):
        add("PEG", "PEG", f"PEG_{_wedge_name(w)}", w)
    for i in range(8):
        add("D7", "Delta7", f"Delta7_{i:02d}", -1)
    for i in range(8):  # PFL3 i reads wedges 2i, 2i+1
        add("PFL3", "PFL3", f"PFL3_{'R' if i % 2 == 0 else 'L'}{i + 1}", 2 * i)

    n = len(types)
    M = np.zeros((n, n), dtype=bool)
    ER, EPG, PEN, PEG, D7, PFL3 = (np.asarray(groups[g]) for g in ("ER", "EPG", "PEN", "PEG", "D7", "PFL3"))

    M[np.ix_(ER, EPG)] = rng.random((len(ER), len(EPG))) < p_broad
    for w in range(N_WEDGES):
        M[EPG[w], PEN[w]] = True
        M[EPG[w], PEG[w]] = True
        shift = +1 if w % 2 == 0 else -1  # R side shifts +1, L side -1
        M[PEN[w], EPG[(w + shift) % N_WEDGES]] = True
        M[PEG[w], EPG[w]] = True
    M[np.ix_(EPG, D7)] = rng.random((len(EPG), len(D7))) < p_broad
    M[np.ix_(D7, EPG)] = rng.random((len(D7), len(EPG))) < p_broad
    for i, j in enumerate(PFL3):
        M[EPG[2 * i], j] = True
        M[EPG[2 * i + 1], j] = True
    np.fill_diagonal(M, False)

    return Connectome(
        M=M,
        neuron_ids=np.arange(n, dtype=np.int64),
        types=np.asarray(types),
        instances=np.asarray(instances),
        groups={g: np.asarray(v) for g, v in groups.items()},
        source="toy",
        meta={"seed": seed, "wedge": np.asarray(wedge)},
    )
