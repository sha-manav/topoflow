from topoflow_prior.schemas import TilePlan
from topoflow_prior.tile_planner import tile_plans_fused_silu_mul_fp8


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
