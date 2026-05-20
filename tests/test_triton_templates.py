import ast

import pytest

from topoflow_prior.remap_planner import REMAP_KINDS
from topoflow_prior.schemas import TilePlan
from topoflow_prior.triton_templates import (
    render_attention_remap,
    render_bias_gelu_dropout,
    render_fused_silu_mul_fp8_quant,
    render_rmsnorm_residual,
)


@pytest.fixture
def shape():
    return {"E": 32, "T": 1024, "H": 4096, "group_size": 128}


@pytest.fixture
def plan():
    return TilePlan(block_m=16, block_h=128, num_warps=4)


def test_render_returns_non_empty(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert isinstance(code, str)
    assert len(code) > 200


def test_rendered_code_is_valid_python(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    ast.parse(code)  # raises SyntaxError on failure


def test_rendered_code_contains_topoflow_intent(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "# TOPOFLOW_INTENT" in code
    assert code.count("# TOPOFLOW_INTENT") >= 3


def test_rendered_code_contains_triton_jit(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "@triton.jit" in code


def test_rendered_code_embeds_tile_constants(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "BLOCK_M" in code
    assert f"BLOCK_M = {plan.block_m}" in code
    assert f"BLOCK_H = {plan.block_h}" in code
    assert f"num_warps={plan.num_warps}" in code


def test_module_level_constants_are_not_constexpr_annotated(plan, shape):
    """Module-level constants must be plain ints; tl.constexpr is only valid
    on @triton.jit kernel parameters, not on module-level assignments."""
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    # Bug guard: never reintroduce module-level `NAME: tl.constexpr = ...`.
    for name in ("BLOCK_M", "BLOCK_H", "GROUP_SIZE", "FP8_E4M3_MAX"):
        assert f"{name}: tl.constexpr = " not in code, (
            f"{name} is annotated tl.constexpr at module level; should be plain int"
        )


def test_different_plans_produce_different_code(shape):
    a = render_fused_silu_mul_fp8_quant(TilePlan(block_m=8, block_h=128, num_warps=4), shape)
    b = render_fused_silu_mul_fp8_quant(TilePlan(block_m=32, block_h=256, num_warps=8), shape)
    assert a != b


def test_three_branches_cover_block_h_relative_to_group(shape):
    """Each (block_h < / == / > group_size) branch must produce parseable code."""
    for bh in (64, 128, 256):
        code = render_fused_silu_mul_fp8_quant(
            TilePlan(block_m=16, block_h=bh, num_warps=4), shape
        )
        ast.parse(code)


# ---------------------------------------------------------------------------
# render_attention_remap
# ---------------------------------------------------------------------------


@pytest.fixture
def attn_plan():
    return TilePlan(block_m=64, block_h=64, num_warps=4)


@pytest.fixture
def attn_shape():
    return {"head_dim": 128, "num_xcds": 8}


def test_render_attention_remap_all_four_kinds_parse(attn_plan, attn_shape):
    """Every REMAP_KIND must render as valid Python."""
    rendered = {}
    for kind in REMAP_KINDS:
        code = render_attention_remap(attn_plan, kind, attn_shape)
        ast.parse(code)
        rendered[kind] = code
    # Each variant must have a distinct decomposition body — no two should be byte-identical.
    assert len(set(rendered.values())) == len(REMAP_KINDS)


def test_render_attention_remap_unknown_kind_raises(attn_plan, attn_shape):
    with pytest.raises(ValueError, match="remap_kind"):
        render_attention_remap(attn_plan, "bogus_kind", attn_shape)


def test_render_attention_remap_contains_topoflow_intent_and_kind(attn_plan, attn_shape):
    code = render_attention_remap(attn_plan, "swizzled_head_first", attn_shape)
    assert "# TOPOFLOW_INTENT" in code
    assert code.count("# TOPOFLOW_INTENT") >= 3
    assert "swizzled_head_first" in code
    assert "@triton.jit" in code


def test_render_attention_remap_embeds_tile_and_shape_constants():
    plan = TilePlan(block_m=64, block_h=128, num_warps=8)
    shape = {"head_dim": 64, "num_xcds": 8}
    code = render_attention_remap(plan, "naive_block_first", shape)
    assert "BLOCK_M = 64" in code
    assert "BLOCK_N = 128" in code
    assert "HEAD_DIM = 64" in code
    assert "NUM_XCDS = 8" in code
    assert "num_warps=8" in code


# ---------------------------------------------------------------------------
# render_rmsnorm_residual
# ---------------------------------------------------------------------------


def test_render_rmsnorm_residual_is_valid_python():
    plan = TilePlan(block_m=4, block_h=4096, num_warps=4)
    code = render_rmsnorm_residual(plan, {"M": 2048, "N": 4096})
    ast.parse(code)
    assert "# TOPOFLOW_INTENT" in code
    assert code.count("# TOPOFLOW_INTENT") >= 3
    assert "@triton.jit" in code
    assert "BLOCK_M = 4" in code
    assert "BLOCK_N = 4096" in code
    assert "num_warps=4" in code
    # Module-level constants are plain ints, NOT tl.constexpr.
    assert "BLOCK_M: tl.constexpr = " not in code
    assert "BLOCK_N: tl.constexpr = " not in code


def test_render_rmsnorm_residual_different_plans_differ():
    shape = {"M": 2048, "N": 4096}
    a = render_rmsnorm_residual(TilePlan(1, 4096, 4), shape)
    b = render_rmsnorm_residual(TilePlan(8, 4096, 8), shape)
    assert a != b


def test_render_rmsnorm_residual_handles_n_up_to_8192():
    """Spec: kernel handles N up to 8192 in a single row per program."""
    plan = TilePlan(block_m=2, block_h=8192, num_warps=8)
    code = render_rmsnorm_residual(plan, {"M": 2048, "N": 8192})
    ast.parse(code)
    assert "BLOCK_N = 8192" in code


# ---------------------------------------------------------------------------
# render_bias_gelu_dropout
# ---------------------------------------------------------------------------


def test_render_bias_gelu_dropout_is_valid_python():
    plan = TilePlan(block_m=4, block_h=2048, num_warps=4)
    code = render_bias_gelu_dropout(plan, {"M": 2048, "N": 16384})
    ast.parse(code)
    assert "# TOPOFLOW_INTENT" in code
    assert code.count("# TOPOFLOW_INTENT") >= 3
    assert "@triton.jit" in code
    assert "BLOCK_M = 4" in code
    assert "BLOCK_N = 2048" in code
    assert "num_warps=4" in code
    # Dropout via tl.rand with deterministic seed.
    assert "tl.rand(" in code
    # Module-level constants are plain ints, not tl.constexpr.
    assert "BLOCK_M: tl.constexpr = " not in code
    assert "BLOCK_N: tl.constexpr = " not in code


def test_render_bias_gelu_dropout_different_plans_differ():
    shape = {"M": 2048, "N": 16384}
    a = render_bias_gelu_dropout(TilePlan(1, 512, 4), shape)
    b = render_bias_gelu_dropout(TilePlan(8, 4096, 8), shape)
    assert a != b


def test_render_bias_gelu_dropout_uses_tanh_approximation():
    code = render_bias_gelu_dropout(
        TilePlan(4, 2048, 4), {"M": 2048, "N": 16384}
    )
    # tanh approximation constants must be present.
    assert "SQRT_2_OVER_PI" in code
    assert "0.044715" in code
    assert "tl.math.tanh" in code


def test_render_attention_remap_decompositions_use_different_arithmetic(attn_plan, attn_shape):
    """Each variant's program-id decomposition body must reference different
    intermediate variables, so GEAK/LLM can see which mapping is in effect."""
    naive_bf = render_attention_remap(attn_plan, "naive_block_first", attn_shape)
    naive_hf = render_attention_remap(attn_plan, "naive_head_first", attn_shape)
    swiz_bf = render_attention_remap(attn_plan, "swizzled_block_first", attn_shape)
    swiz_hf = render_attention_remap(attn_plan, "swizzled_head_first", attn_shape)

    # Only naive_block_first should compute pid_h = rem // num_m
    assert "pid_h = rem // num_m" in naive_bf
    assert "pid_h = rem // num_m" not in naive_hf
    # Only naive_head_first should compute pid_m = rem // num_heads
    assert "pid_m = rem // num_heads" in naive_hf
    assert "pid_m = rem // num_heads" not in naive_bf
    # Only swizzled variants reference NUM_XCDS in the decomposition.
    assert "xcd = rem % NUM_XCDS" in swiz_bf
    assert "xcd = rem % NUM_XCDS" in swiz_hf
    assert "xcd = rem % NUM_XCDS" not in naive_bf
    assert "xcd = rem % NUM_XCDS" not in naive_hf
    # swizzled_block_first references blocks_per_xcd; swizzled_head_first references head_in_xcd.
    assert "blocks_per_xcd" in swiz_bf and "blocks_per_xcd" not in swiz_hf
    assert "head_in_xcd" in swiz_hf and "head_in_xcd" not in swiz_bf
