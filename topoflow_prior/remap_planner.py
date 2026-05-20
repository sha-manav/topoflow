"""Workgroup remapping orderings for attention kernels on MI300X.

These functions compute a *linearized program id* given grid coordinates
(pid_m, pid_h, pid_b). The MI300X XCD scheduler assigns consecutive program
ids to XCDs round-robin; by reordering ids we can place blocks that reuse
K/V data onto the same XCD's L2.

All four functions are permutations of [0, num_m * num_h * num_b).
"""

from __future__ import annotations

from .schemas import TopologyPlan

REMAP_KINDS: tuple[str, ...] = (
    "naive_block_first",
    "naive_head_first",
    "swizzled_block_first",
    "swizzled_head_first",
)


def naive_block_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    return pid_b * num_h * num_m + pid_h * num_m + pid_m


def naive_head_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    return pid_b * num_h * num_m + pid_m * num_h + pid_h


def swizzled_block_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    # Group by m-block first so blocks with the same query tile land on the
    # same XCD.  Linearise as m-outer, b-middle, h-inner so that the MI300X
    # round-robin scheduler places consecutive m-block IDs on consecutive XCDs,
    # keeping each XCD's L2 warm for the corresponding Q tile.
    return pid_m * num_b * num_h + pid_b * num_h + pid_h


def swizzled_head_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    # Group by head first so all m-blocks of a head receive consecutive program
    # IDs (arXiv 2511.02132).  With num_xcds XCDs and B*M blocks per head the
    # scheduler places a head's blocks on at most min(B*M, num_xcds) XCDs,
    # maximising K/V reuse in each XCD's 4 MB L2.
    # Linearise as h-outer, b-middle, m-inner.
    return pid_h * num_b * num_m + pid_b * num_m + pid_m


_REMAP_NOTES = {
    "naive_block_first": "Baseline: m-blocks contiguous; XCD reuse only across batches.",
    "naive_head_first": "Heads swept within m-block; little K/V locality on XCDs.",
    "swizzled_block_first": "M-blocks grouped per XCD; helpful when Q tile is reused.",
    "swizzled_head_first": (
        "Heads grouped per XCD; all blocks of a head land on same XCD so K/V "
        "fits in 4MB L2 (paper: arXiv 2511.02132)."
    ),
}


def attention_remap_plans() -> list[TopologyPlan]:
    """Return one TopologyPlan per remap kind."""
    plans: list[TopologyPlan] = []
    for kind in REMAP_KINDS:
        plans.append(
            TopologyPlan(
                use_workgroup_remap=kind.startswith("swizzled"),
                remap_kind=kind,
                notes=_REMAP_NOTES[kind],
            )
        )
    return plans
