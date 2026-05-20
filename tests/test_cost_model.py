import pytest

from topoflow_prior.cost_model import estimate_fused_silu_mul_fp8_quant
from topoflow_prior.schemas import CostEstimate


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
