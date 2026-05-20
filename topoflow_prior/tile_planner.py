"""Enumerate tile-plan candidates for the fused SiLU+Mul+FP8 op."""

from __future__ import annotations

from .schemas import TilePlan

_BLOCK_M_CANDIDATES = (8, 16, 32)
_BLOCK_H_CANDIDATES = (64, 128, 256)
_NUM_WARPS_CANDIDATES = (4, 8)


def tile_plans_fused_silu_mul_fp8(H: int, group_size: int = 128) -> list[TilePlan]:
    """Enumerate (block_m, block_h, num_warps) tile candidates.

    Filters block_h so that either block_h % group_size == 0 (multi-group tile)
    or group_size % block_h == 0 (sub-group tile). Also filters block_h > H.
    """
    plans: list[TilePlan] = []
    for bm in _BLOCK_M_CANDIDATES:
        for bh in _BLOCK_H_CANDIDATES:
            if bh > H:
                continue
            if not (bh % group_size == 0 or group_size % bh == 0):
                continue
            for nw in _NUM_WARPS_CANDIDATES:
                plans.append(TilePlan(block_m=bm, block_h=bh, num_warps=nw))
    return plans
