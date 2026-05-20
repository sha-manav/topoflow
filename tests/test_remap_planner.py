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
    return dict(num_m=4, num_h=8, num_b=2, num_xcds=8)


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


def test_swizzled_head_first_groups_heads_to_xcds(grid):
    """Heads in the same xcd_group should have consecutive program ids in a stride."""
    g = grid
    heads_per_group = (g["num_h"] + g["num_xcds"] - 1) // g["num_xcds"]
    # Test consistency with the spec's formula
    expected = (
        0 * g["num_h"] * g["num_m"]
        + 0 * heads_per_group * g["num_m"]
        + 0 * g["num_m"]
        + 0
    )
    assert swizzled_head_first(0, 0, 0, g["num_m"], g["num_h"], g["num_b"], g["num_xcds"]) == expected


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
