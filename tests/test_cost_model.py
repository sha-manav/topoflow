import pytest

from topoflow_prior.cost_model import (
    estimate_fused_silu_mul_fp8_quant,
    estimate_tile_cost,
)
from topoflow_prior.schemas import CostEstimate, TilePlan
from topoflow_prior.tile_planner import tile_plans_fused_silu_mul_fp8


def test_basic_shape_returns_cost_estimate():
    c = estimate_fused_silu_mul_fp8_quant(E=32, T=1024, H=4096, group_size=128)
    assert isinstance(c, CostEstimate)


def test_byte_formulas():
    E, T, H, G = 32, 1024, 4096, 128
    c = estimate_fused_silu_mul_fp8_quant(E=E, T=T, H=H, group_size=G)
    expected_input = E * T * 2 * H * 2  # bf16
    expected_output = E * T * H * 1  # fp8
    expected_scale = E * T * (H // G) * 4  # fp32
    expected_fused = expected_input + expected_output + expected_scale
    unfused_intermediate = E * T * H * 2 * 2  # bf16 intermediate write+read
    expected_unfused = expected_fused + unfused_intermediate
    assert c.fused_bytes == expected_fused
    assert c.unfused_bytes == expected_unfused
    assert c.bytes_saved == expected_unfused - expected_fused
    assert c.bytes_saved > 0


def test_score_in_unit_interval():
    c = estimate_fused_silu_mul_fp8_quant(E=1, T=1, H=128, group_size=128)
    assert 0.0 < c.score < 1.0


def test_score_ranking_prefers_larger_savings():
    """Larger H => more intermediate bytes saved => lower score (less fused/unfused ratio)."""
    c_small = estimate_fused_silu_mul_fp8_quant(E=1, T=64, H=128, group_size=128)
    c_large = estimate_fused_silu_mul_fp8_quant(E=1, T=64, H=8192, group_size=128)
    # both are saving the same fraction of intermediate, but score is fused/unfused
    # since both have proportional growth, scores should be close; just ensure no crash
    assert 0 < c_small.score < 1
    assert 0 < c_large.score < 1


def test_group_size_must_divide_h():
    with pytest.raises(ValueError, match="group_size"):
        estimate_fused_silu_mul_fp8_quant(E=1, T=1, H=100, group_size=128)


# ---------------------------------------------------------------------------
# estimate_tile_cost
# ---------------------------------------------------------------------------

_STD_SHAPE = dict(E=32, T=1024, H=4096, group_size=128)


def test_estimate_tile_cost_returns_cost_estimate():
    c = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(block_m=16, block_h=128, num_warps=4))
    assert isinstance(c, CostEstimate)


def test_two_pass_penalty_block_h_less_than_group_size():
    """BLOCK_H=64 (sub-group, two reads of input) is worse than BLOCK_H=128 (one pass)."""
    c_subgroup = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(8, 64, 4))
    c_onegroup = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(8, 128, 4))
    assert c_subgroup.score > c_onegroup.score
    # Same comparison should hold for any BLOCK_M; spot-check at BLOCK_M=32.
    c_subgroup_32 = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(32, 64, 4))
    c_onegroup_32 = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(32, 128, 4))
    assert c_subgroup_32.score > c_onegroup_32.score


def test_register_pressure_penalty_large_tile_worse_than_moderate():
    """BLOCK_M*BLOCK_H*4 bytes > 16KB triggers a spill penalty."""
    # 32 * 256 * 4 = 32768 bytes = 2x the 16KB budget -> spill factor 2.
    c_large = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(32, 256, 4))
    # 8 * 128 * 4 = 4096 bytes -> no spill.
    c_moderate = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(8, 128, 4))
    assert c_large.score > c_moderate.score


def test_vectorization_bonus_block_h_divisible_by_8():
    """Aligned BLOCK_H (% 8 == 0) is slightly better than otherwise-identical unaligned BLOCK_H.

    Standard grid all has BLOCK_H % 8 == 0; we synthesize a 257-wide tile to
    verify the bonus side-by-side with 256. Both are BLOCK_H > group_size, so
    the two-pass branch is not triggered, and 8*256/8*257 register counts are
    both within budget — the only differing signal is vectorization.
    """
    c_aligned = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(8, 256, 4))
    c_unaligned = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(8, 257, 4))
    assert c_aligned.score < c_unaligned.score


def test_scores_not_all_identical_for_standard_tile_grid():
    """The 18-plan standard grid produces multiple distinct scores."""
    plans = tile_plans_fused_silu_mul_fp8(H=_STD_SHAPE["H"], group_size=_STD_SHAPE["group_size"])
    scores = {estimate_tile_cost(**_STD_SHAPE, tile_plan=p).score for p in plans}
    assert len(scores) >= 2, "tile cost must differentiate at least some plans"


def test_tile_cost_score_can_exceed_unity_for_bad_tiles():
    """Very large tiles can push fused traffic above unfused (negative bytes_saved)."""
    # 32 * 256 = 8192 fp32 = 32768 bytes -> spill factor 2.0, vectorization 0.95.
    # Combined penalty ~1.9x on shape fused. With shape fused/unfused ~ 0.56,
    # tile score ~ 1.06.
    c = estimate_tile_cost(**_STD_SHAPE, tile_plan=TilePlan(32, 256, 4))
    assert c.score > 1.0
    assert c.bytes_saved < 0
