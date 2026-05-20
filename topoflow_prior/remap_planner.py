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


def swizzled_head_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    """Pack all blocks of a head onto the same XCD.

    Models the MI300X round-robin XCD scheduler (program i runs on XCD
    i % num_xcds). Head h is assigned to XCD (h % num_xcds); heads h and
    h+num_xcds share an XCD so K/V for those heads can hit one L2.
    Requires num_h % num_xcds == 0.
    See arXiv 2511.02132 (NUMA attention).
    """
    if num_h % num_xcds != 0:
        raise ValueError(
            f"swizzled_head_first requires num_h ({num_h}) divisible by "
            f"num_xcds ({num_xcds})"
        )
    xcd = pid_h % num_xcds
    head_in_xcd = pid_h // num_xcds
    program_in_batch = (head_in_xcd * num_m + pid_m) * num_xcds + xcd
    return pid_b * num_h * num_m + program_in_batch


def swizzled_block_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    """Pack all m-blocks at the same m-index across heads onto the same XCD.

    Models the MI300X round-robin XCD scheduler. m-block m is assigned to
    XCD (m % num_xcds). m-blocks m and m+num_xcds share an XCD.
    Within each XCD group, heads vary fastest so that consecutive program ids
    step through heads before advancing to the next m-index group; this is
    distinct from naive_block_first (where m varies fastest).
    Requires num_m % num_xcds == 0.
    """
    if num_m % num_xcds != 0:
        raise ValueError(
            f"swizzled_block_first requires num_m ({num_m}) divisible by "
            f"num_xcds ({num_xcds})"
        )
    xcd = pid_m % num_xcds
    block_in_xcd = pid_m // num_xcds
    program_in_batch = (block_in_xcd * num_h + pid_h) * num_xcds + xcd
    return pid_b * num_h * num_m + program_in_batch


_REMAP_NOTES = {
    "naive_block_first": "Baseline: m-blocks contiguous; XCD reuse only across batches.",
    "naive_head_first": "Heads swept within m-block; little K/V locality on XCDs.",
    "swizzled_block_first": (
        "M-blocks at the same m-index share an XCD via round-robin scheduling "
        "(requires num_m % num_xcds == 0)."
    ),
    "swizzled_head_first": (
        "All blocks of one head land on the same XCD via round-robin scheduling "
        "(requires num_h % num_xcds == 0). K/V for that head fits in 4MB L2 "
        "(arXiv 2511.02132)."
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
