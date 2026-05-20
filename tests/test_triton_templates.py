import ast

import pytest

from topoflow_prior.schemas import TilePlan
from topoflow_prior.triton_templates import render_fused_silu_mul_fp8_quant


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
    assert f"BLOCK_M: tl.constexpr = {plan.block_m}" in code
    assert f"BLOCK_H: tl.constexpr = {plan.block_h}" in code
    assert f"num_warps={plan.num_warps}" in code


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
