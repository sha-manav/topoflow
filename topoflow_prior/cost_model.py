"""Memory-traffic-based cost estimator.

Not accurate in absolute terms; goal is to rank candidates better than random.
"""

from __future__ import annotations

from .schemas import CostEstimate, TilePlan

_BYTES_BF16 = 2
_BYTES_FP8 = 1
_BYTES_FP32 = 4

# Rough VGPR budget per wavefront on MI300X, in bytes of fp32 register state
# the kernel can hold without spilling to scratch (LDS / HBM). This is a
# coarse proxy — exact pressure depends on liveness analysis the cost model
# does not perform — but it ranks tiles better than random.
_REGISTER_BUDGET_BYTES = 16 * 1024

# Cache-line-aligned bf16 vector load width on MI300X is 16 bytes = 8 elements.
_VECTOR_LANES_BF16 = 8

# Discount applied when BLOCK_H is divisible by the vector width.
_VECTORIZATION_BONUS = 0.95


def estimate_fused_silu_mul_fp8_quant(
    E: int, T: int, H: int, group_size: int = 128
) -> CostEstimate:
    if H % group_size != 0:
        raise ValueError(
            f"group_size ({group_size}) must divide H ({H}); got H % group_size = {H % group_size}"
        )

    input_read = E * T * 2 * H * _BYTES_BF16
    output_write = E * T * H * _BYTES_FP8
    scale_write = E * T * (H // group_size) * _BYTES_FP32
    fused = input_read + output_write + scale_write

    # Unfused path: write bf16 intermediate (silu*mul output) then read it again
    unfused_intermediate = E * T * H * _BYTES_BF16 * 2
    unfused = fused + unfused_intermediate

    return CostEstimate(
        fused_bytes=fused,
        unfused_bytes=unfused,
        bytes_saved=unfused - fused,
        score=fused / unfused,
    )


def estimate_tile_cost(
    E: int, T: int, H: int, group_size: int, tile_plan: TilePlan
) -> CostEstimate:
    """Per-tile cost: shape-level memory traffic adjusted by tile penalties.

    Penalties:
    - **Two-pass re-read**: when BLOCK_H < group_size, the kernel makes one
      pass to compute the per-group amax and a second pass to quantize, so
      the bf16 inputs are read twice. Adds an extra input_read worth of bytes
      to the fused traffic.
    - **Register spill**: holding a BLOCK_M x BLOCK_H fp32 activation tile in
      registers requires BLOCK_M * BLOCK_H * 4 bytes per wavefront; exceeding
      ~16KB triggers spills to LDS/HBM. Multiply fused bytes by the spill
      ratio (registers / budget).
    - **Vectorization bonus**: BLOCK_H divisible by 8 (16-byte bf16 vector
      width on MI300X) lets the loads use vectorized intrinsics, a small
      ~5% discount.

    Score is fused_bytes / unfused_bytes; lower is better. bytes_saved may be
    negative when a tile's penalties make the fused path more expensive than
    the unfused baseline (a strong "do not pick this tile" signal).
    """
    base = estimate_fused_silu_mul_fp8_quant(E, T, H, group_size)
    fused = float(base.fused_bytes)

    if tile_plan.block_h < group_size:
        input_read_extra = E * T * 2 * H * _BYTES_BF16
        fused += input_read_extra

    register_bytes = tile_plan.block_m * tile_plan.block_h * _BYTES_FP32
    if register_bytes > _REGISTER_BUDGET_BYTES:
        spill_factor = register_bytes / _REGISTER_BUDGET_BYTES
        fused *= spill_factor

    if tile_plan.block_h % _VECTOR_LANES_BF16 == 0:
        fused *= _VECTORIZATION_BONUS

    fused_bytes = int(fused)
    unfused_bytes = base.unfused_bytes
    return CostEstimate(
        fused_bytes=fused_bytes,
        unfused_bytes=unfused_bytes,
        bytes_saved=unfused_bytes - fused_bytes,
        score=fused_bytes / unfused_bytes,
    )
