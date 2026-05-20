import itertools

import pytest

from topoflow_prior.remap_planner import (
    REMAP_KINDS,
    attention_remap_plans,
    naive_block_first,
    naive_head_first,
    swizzled_block_first,
    swizzled_head_first,
)


@pytest.fixture
def grid():
    return dict(num_m=16, num_h=16, num_b=2, num_xcds=8)


def _all_outputs(fn, grid):
    out = []
    for pid_b, pid_h, pid_m in itertools.product(
        range(grid["num_b"]), range(grid["num_h"]), range(grid["num_m"])
    ):
        out.append(fn(pid_m, pid_h, pid_b, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]))
    return out


@pytest.mark.parametrize(
    "fn",
    [naive_block_first, naive_head_first, swizzled_block_first, swizzled_head_first],
)
def test_each_mapping_is_a_permutation(fn, grid):
    outputs = _all_outputs(fn, grid)
    total = grid["num_m"] * grid["num_h"] * grid["num_b"]
    assert sorted(outputs) == list(range(total))


def test_mappings_are_distinct(grid):
    sigs = {
        name: tuple(_all_outputs(fn, grid))
        for name, fn in [
            ("naive_block_first", naive_block_first),
            ("naive_head_first", naive_head_first),
            ("swizzled_block_first", swizzled_block_first),
            ("swizzled_head_first", swizzled_head_first),
        ]
    }
    assert len(set(sigs.values())) == 4, sigs


def test_naive_block_first_layout(grid):
    # naive block-first: m varies fastest, then h, then b
    assert naive_block_first(0, 0, 0, **{k: grid[k] for k in ("num_m", "num_h", "num_b", "num_xcds")}) == 0
    assert naive_block_first(1, 0, 0, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]) == 1
    assert naive_block_first(0, 1, 0, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]) == grid["num_m"]


def test_swizzled_head_first_groups_heads_to_xcds_property(grid):
    """All blocks of head h must hit XCD (h % num_xcds) under round-robin scheduling."""
    g = grid
    for pid_h in range(g["num_h"]):
        expected_xcd = pid_h % g["num_xcds"]
        for pid_m in range(g["num_m"]):
            for pid_b in range(g["num_b"]):
                pid = swizzled_head_first(
                    pid_m, pid_h, pid_b,
                    g["num_m"], g["num_h"], g["num_b"], g["num_xcds"],
                )
                assert pid % g["num_xcds"] == expected_xcd, (
                    f"head {pid_h} block (m={pid_m}, b={pid_b}) landed on "
                    f"XCD {pid % g['num_xcds']}, expected {expected_xcd}"
                )


def test_swizzled_block_first_groups_blocks_to_xcds_property(grid):
    """All blocks at m-index m must hit XCD (m % num_xcds)."""
    g = grid
    for pid_m in range(g["num_m"]):
        expected_xcd = pid_m % g["num_xcds"]
        for pid_h in range(g["num_h"]):
            for pid_b in range(g["num_b"]):
                pid = swizzled_block_first(
                    pid_m, pid_h, pid_b,
                    g["num_m"], g["num_h"], g["num_b"], g["num_xcds"],
                )
                assert pid % g["num_xcds"] == expected_xcd


def test_swizzle_raises_when_not_divisible():
    with pytest.raises(ValueError, match="divisible"):
        swizzled_head_first(0, 0, 0, num_m=4, num_h=7, num_b=1, num_xcds=8)
    with pytest.raises(ValueError, match="divisible"):
        swizzled_block_first(0, 0, 0, num_m=7, num_h=8, num_b=1, num_xcds=8)


def test_remap_kinds_constant():
    assert set(REMAP_KINDS) == {
        "naive_block_first",
        "naive_head_first",
        "swizzled_block_first",
        "swizzled_head_first",
    }


def test_attention_remap_plans_returns_four_topology_plans():
    plans = attention_remap_plans()
    assert len(plans) == 4
    kinds = {p.remap_kind for p in plans}
    assert kinds == set(REMAP_KINDS)
    # swizzled variants set use_workgroup_remap=True
    swizzled = [p for p in plans if p.remap_kind.startswith("swizzled")]
    assert all(p.use_workgroup_remap for p in swizzled)
    naive = [p for p in plans if p.remap_kind.startswith("naive")]
    assert all(not p.use_workgroup_remap for p in naive)
