"""Jinja2-based Triton kernel rendering."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from .remap_planner import REMAP_KINDS
from .schemas import TilePlan

_ENV = Environment(
    loader=PackageLoader("topoflow_prior", "templates/triton"),
    autoescape=select_autoescape(default=False),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=False,
    lstrip_blocks=False,
)


def render_fused_silu_mul_fp8_quant(
    tile_plan: TilePlan, shape: dict[str, Any]
) -> str:
    """Render the fused SiLU+Mul+FP8 Triton kernel for one tile plan.

    Args:
        tile_plan: BLOCK_M / BLOCK_H / num_warps.
        shape: dict with at least key "group_size".
    Returns:
        Python source code (str) -- guaranteed to be syntactically valid Python.
    """
    template = _ENV.get_template("fused_silu_mul_fp8_quant.py.j2")
    return template.render(
        BLOCK_M=tile_plan.block_m,
        BLOCK_H=tile_plan.block_h,
        GROUP_SIZE=int(shape.get("group_size", 128)),
        num_warps=tile_plan.num_warps,
    )


def render_attention_remap(
    tile_plan: TilePlan, remap_kind: str, shape: dict[str, Any]
) -> str:
    """Render the naive single-head attention kernel with the given remap.

    Args:
        tile_plan: BLOCK_M and BLOCK_H (reused as BLOCK_N for the K/V tile),
            plus num_warps.
        remap_kind: one of REMAP_KINDS (`naive_block_first`,
            `naive_head_first`, `swizzled_block_first`, `swizzled_head_first`).
        shape: dict with keys ``head_dim`` (default 128) and ``num_xcds``
            (default 8).
    Returns:
        Python source code (str) -- guaranteed to be syntactically valid Python.
    """
    if remap_kind not in REMAP_KINDS:
        raise ValueError(
            f"unknown remap_kind {remap_kind!r}; expected one of {REMAP_KINDS}"
        )
    template = _ENV.get_template("attention_remap.py.j2")
    return template.render(
        BLOCK_M=tile_plan.block_m,
        BLOCK_N=tile_plan.block_h,
        head_dim=int(shape.get("head_dim", 128)),
        num_xcds=int(shape.get("num_xcds", 8)),
        num_warps=tile_plan.num_warps,
        remap_kind=remap_kind,
    )
