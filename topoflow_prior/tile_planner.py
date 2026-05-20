"""Enumerate tile-plan candidates per target op."""

from __future__ import annotations

from .schemas import TilePlan

_BLOCK_M_CANDIDATES = (8, 16, 32)
_BLOCK_H_CANDIDATES = (64, 128, 256)
_NUM_WARPS_CANDIDATES = (4, 8)


def _next_power_of_2(n: int) -> int:
    """Smallest power of two >= n; matches Triton's row-coverage convention."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


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


_RMSNORM_BLOCK_M_CANDIDATES = (1, 2, 4, 8)


def tile_plans_rmsnorm_residual(N: int) -> list[TilePlan]:
    """Enumerate (BLOCK_M, BLOCK_N, num_warps) tile candidates for RMSNorm+residual.

    BLOCK_N is fixed to next_power_of_2(N) so each program covers a full row;
    the mean-of-squares reduction then collapses to a single tl.sum. BLOCK_M is
    rows-per-program; we sweep [1, 2, 4, 8] which span the regime from
    "row latency-bound" to "row throughput-bound" on MI300X.
    """
    block_n = _next_power_of_2(N)
    plans: list[TilePlan] = []
    for bm in _RMSNORM_BLOCK_M_CANDIDATES:
        for nw in _NUM_WARPS_CANDIDATES:
            plans.append(TilePlan(block_m=bm, block_h=block_n, num_warps=nw))
    return plans


_BIAS_GELU_BLOCK_M_CANDIDATES = (1, 2, 4, 8)
_BIAS_GELU_BLOCK_N_CANDIDATES = (512, 1024, 2048, 4096)


def tile_plans_bias_gelu_dropout(N: int) -> list[TilePlan]:
    """Enumerate (BLOCK_M, BLOCK_N, num_warps) tile candidates for bias+GELU+dropout.

    Pointwise op — no row constraint. We sweep small-to-large BLOCK_N
    (cache-line vs LDS-friendly) and BLOCK_M in [1, 2, 4, 8]. BLOCK_N values
    exceeding N are filtered (the kernel would over-load and waste registers).
    """
    plans: list[TilePlan] = []
    for bm in _BIAS_GELU_BLOCK_M_CANDIDATES:
        for bn in _BIAS_GELU_BLOCK_N_CANDIDATES:
            if bn > N:
                continue
            for nw in _NUM_WARPS_CANDIDATES:
                plans.append(TilePlan(block_m=bm, block_h=bn, num_warps=nw))
    return plans
