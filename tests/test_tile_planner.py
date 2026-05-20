from topoflow_prior.schemas import TilePlan
from topoflow_prior.tile_planner import (
    tile_plans_bias_gelu_dropout,
    tile_plans_fused_silu_mul_fp8,
    tile_plans_rmsnorm_residual,
)


def test_returns_tile_plans():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    assert all(isinstance(p, TilePlan) for p in plans)
    assert len(plans) >= 12


def test_expected_count_for_default_grid():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    # bm in [8,16,32], bh in [64,128,256] (all satisfy bh%128==0 or 128%bh==0),
    # num_warps in [4,8] => 3*3*2 = 18
    assert len(plans) == 18


def test_block_m_values_subset():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    bms = {p.block_m for p in plans}
    assert bms == {8, 16, 32}


def test_block_h_values_satisfy_group_compat():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    for p in plans:
        assert p.block_h % 128 == 0 or 128 % p.block_h == 0


def test_num_warps_values_subset():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    nws = {p.num_warps for p in plans}
    assert nws == {4, 8}


def test_plans_are_unique():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    tuples = {(p.block_m, p.block_h, p.num_warps) for p in plans}
    assert len(tuples) == len(plans)


def test_block_h_does_not_exceed_h():
    """If H is small, very large BLOCK_H should be filtered out."""
    plans = tile_plans_fused_silu_mul_fp8(H=128, group_size=128)
    for p in plans:
        assert p.block_h <= 128


# ---------------------------------------------------------------------------
# tile_plans_rmsnorm_residual
# ---------------------------------------------------------------------------


def test_rmsnorm_plans_block_n_is_next_power_of_2():
    plans = tile_plans_rmsnorm_residual(N=4096)
    # 4096 is already a power of two.
    assert all(p.block_h == 4096 for p in plans)
    # Non-power-of-2 N is rounded up.
    plans_5000 = tile_plans_rmsnorm_residual(N=5000)
    assert all(p.block_h == 8192 for p in plans_5000)


def test_rmsnorm_plan_count_is_eight():
    """4 BLOCK_M values x 2 num_warps = 8 plans."""
    plans = tile_plans_rmsnorm_residual(N=4096)
    assert len(plans) == 8
    bms = {p.block_m for p in plans}
    assert bms == {1, 2, 4, 8}
    nws = {p.num_warps for p in plans}
    assert nws == {4, 8}


def test_rmsnorm_plans_are_unique():
    plans = tile_plans_rmsnorm_residual(N=4096)
    tuples = {(p.block_m, p.block_h, p.num_warps) for p in plans}
    assert len(tuples) == len(plans)


# ---------------------------------------------------------------------------
# tile_plans_bias_gelu_dropout
# ---------------------------------------------------------------------------


def test_bias_gelu_plan_count_filters_block_n_gt_n():
    """For N=4096, all 4 BLOCK_N values (512, 1024, 2048, 4096) are valid."""
    plans = tile_plans_bias_gelu_dropout(N=4096)
    assert len(plans) == 4 * 4 * 2  # BLOCK_M(4) x BLOCK_N(4) x num_warps(2) = 32
    for p in plans:
        assert p.block_h <= 4096


def test_bias_gelu_plan_count_small_n_filters_large_block_n():
    """For N=1024, only BLOCK_N in {512, 1024} survive."""
    plans = tile_plans_bias_gelu_dropout(N=1024)
    assert all(p.block_h <= 1024 for p in plans)
    bhs = {p.block_h for p in plans}
    assert bhs == {512, 1024}
    assert len(plans) == 4 * 2 * 2  # BLOCK_M(4) x BLOCK_N(2) x num_warps(2)


def test_bias_gelu_plans_have_at_least_eight():
    plans = tile_plans_bias_gelu_dropout(N=4096)
    assert len(plans) >= 8


def test_bias_gelu_plans_are_unique():
    plans = tile_plans_bias_gelu_dropout(N=4096)
    tuples = {(p.block_m, p.block_h, p.num_warps) for p in plans}
    assert len(tuples) == len(plans)
