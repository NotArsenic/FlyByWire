"""FlyWire connectome, mask conditions and directed rewiring.

Convention: ``M[i, j] = True`` iff presynaptic i -> postsynaptic j.
"""

from flybywire.connectome.celltypes import Connectome, match_groups
from flybywire.connectome.masks import MASK_TYPES, apply_mask_type, build_connectome, load_base_connectome
from flybywire.connectome.rewire import directed_rewire, erdos_renyi_like

__all__ = [
    "Connectome",
    "MASK_TYPES",
    "apply_mask_type",
    "build_connectome",
    "directed_rewire",
    "erdos_renyi_like",
    "load_base_connectome",
    "match_groups",
]
