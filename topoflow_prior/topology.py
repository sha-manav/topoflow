"""AMD GPU topology specs for Topo-Flow Prior."""

from __future__ import annotations

from .schemas import TopologySpec

MI300X = TopologySpec(
    name="MI300X",
    arch="gfx942",
    num_xcds=8,
    l2_per_xcd_mb=4.0,
    cu_per_xcd=38,
)

MI355X = TopologySpec(
    name="MI355X",
    arch="gfx950",
    num_xcds=8,
    l2_per_xcd_mb=4.0,
    cu_per_xcd=38,
)

_REGISTRY: dict[str, TopologySpec] = {
    "mi300x": MI300X,
    "gfx942": MI300X,
    "mi355x": MI355X,
    "gfx950": MI355X,
}


def get_topology(name_or_arch: str) -> TopologySpec:
    key = name_or_arch.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown arch: {name_or_arch!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
