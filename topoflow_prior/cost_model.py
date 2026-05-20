"""Memory-traffic-based cost estimator.

Not accurate in absolute terms; goal is to rank candidates better than random.
"""

from __future__ import annotations

from .schemas import CostEstimate

_BYTES_BF16 = 2
_BYTES_FP8 = 1
_BYTES_FP32 = 4


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
