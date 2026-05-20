"""Jinja2-based Triton kernel rendering."""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

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
