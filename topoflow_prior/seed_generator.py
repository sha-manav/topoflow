"""Seed bundle orchestrator: planners + cost model + template renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cost_model import estimate_fused_silu_mul_fp8_quant, estimate_tile_cost
from .dataflow import fused_silu_mul_fp8_quant_dfg
from .schemas import (
    CostEstimate,
    SeedCandidate,
    TilePlan,
    TopologyPlan,
    TopologySpec,
)
from .tile_planner import tile_plans_fused_silu_mul_fp8
from .triton_templates import render_fused_silu_mul_fp8_quant

_NOTES_TEMPLATE = """\
# Topo-Flow seed: {candidate_id}

This candidate fuses SiLU, multiply, and per-group FP8 quantization for the
batched MoE post-projection on {arch} ({topology_name}). Intent: avoid
writing the bf16 silu(gate)*up intermediate to HBM. Keep per-group amax/scale
local to the tile.

## Tile plan
- BLOCK_M = {block_m}
- BLOCK_H = {block_h}
- GROUP_SIZE = {group_size}
- num_warps = {num_warps}

## Cost model — tile level (lower score = better)
- fused bytes (tile-adjusted):   {fused_bytes:,}
- unfused bytes:                 {unfused_bytes:,}
- bytes saved (may be negative): {bytes_saved:,}
- score:                         {score:.4f}

## Cost model — shape level (op-level fused vs unfused)
- shape fused bytes:   {shape_fused_bytes:,}
- shape unfused bytes: {shape_unfused_bytes:,}
- shape bytes saved:   {shape_bytes_saved:,}
- shape score:         {shape_score:.4f}

## Risks
- Register pressure: BLOCK_M * BLOCK_H fp32 in registers may reduce occupancy.
- FP8 scale precision must match the reference (per-group amax / FP8_E4M3_MAX).
- BLOCK_H sub-group / multi-group variants take different code paths
  (see TOPOFLOW_INTENT branch comments in kernel.py).

## Suggested mutations for GEAK
- BLOCK_M in [8, 16, 32]
- BLOCK_H in [64, 128, 256] (keep BLOCK_H % GROUP_SIZE == 0 or
  GROUP_SIZE % BLOCK_H == 0)
- num_warps in [4, 8]
- Try swapping the order of the two passes in the BLOCK_H < GROUP_SIZE branch.
"""


def _build_metadata(
    candidate_id: str,
    shape: dict[str, Any],
    topology: TopologySpec,
    tile_plan: TilePlan,
    topology_plan: TopologyPlan,
    shape_cost: CostEstimate,
    tile_cost: CostEstimate,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "op": "fused_silu_mul_fp8_quant",
        "target_arch": topology.arch,
        "topology": {
            "name": topology.name,
            "num_xcds": topology.num_xcds,
            "l2_per_xcd_mb": topology.l2_per_xcd_mb,
            "cu_per_xcd": topology.cu_per_xcd,
        },
        "shape": dict(shape),
        "dtype": {"input": "bf16", "output": "fp8_e4m3", "scale": "fp32"},
        "tile_plan": {
            "BLOCK_M": tile_plan.block_m,
            "BLOCK_H": tile_plan.block_h,
            "num_warps": tile_plan.num_warps,
        },
        "topology_plan": {
            "use_workgroup_remap": topology_plan.use_workgroup_remap,
            "remap_kind": topology_plan.remap_kind,
            "keep_group_quant_local": True,
            "notes": topology_plan.notes,
        },
        # Per-candidate score with tile-level penalties applied
        # (register pressure, two-pass re-read, vectorization).
        "cost_model": {
            "fused_bytes": tile_cost.fused_bytes,
            "unfused_bytes": tile_cost.unfused_bytes,
            "bytes_saved": tile_cost.bytes_saved,
            "score": tile_cost.score,
        },
        # Shape-level estimate: how much HBM traffic fusion saves at this
        # (E, T, H, group_size) regardless of tile choice. Always positive.
        "shape_cost": {
            "fused_bytes": shape_cost.fused_bytes,
            "unfused_bytes": shape_cost.unfused_bytes,
            "bytes_saved": shape_cost.bytes_saved,
            "score": shape_cost.score,
        },
        "fusion_plan": [
            "silu",
            "mul",
            "group_amax",
            "scale",
            "fp8_quantize",
        ],
        "suggested_mutations": [
            "BLOCK_M in [8, 16, 32]",
            "BLOCK_H in [64, 128, 256] (bh % gs == 0 or gs % bh == 0)",
            "num_warps in [4, 8]",
        ],
    }


def _write_candidate(
    out_dir: Path,
    candidate_id: str,
    kernel_code: str,
    metadata: dict[str, Any],
    notes: str,
) -> Path:
    folder = out_dir / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(kernel_code)
    (folder / "topoflow_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (folder / "topoflow_notes.md").write_text(notes)
    return folder


def generate_seeds_for_fused_silu_mul_fp8_quant(
    shape: dict[str, Any],
    topology: TopologySpec,
    out_dir: str | Path,
) -> list[SeedCandidate]:
    """Produce one candidate per tile plan; write to disk; return objects.

    shape must contain keys E, T, H, group_size.
    """
    required = {"E", "T", "H", "group_size"}
    missing = required - set(shape)
    if missing:
        raise ValueError(f"shape missing required keys: {sorted(missing)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataflow loads (used downstream by GEAK introspection).
    _ = fused_silu_mul_fp8_quant_dfg()

    shape_cost = estimate_fused_silu_mul_fp8_quant(
        E=shape["E"], T=shape["T"], H=shape["H"], group_size=shape["group_size"]
    )

    plans = tile_plans_fused_silu_mul_fp8(H=shape["H"], group_size=shape["group_size"])
    topology_plan = TopologyPlan(
        use_workgroup_remap=False,
        remap_kind=None,
        notes="keep_group_quant_local: BLOCK_H related to GROUP_SIZE",
    )

    candidates: list[SeedCandidate] = []
    for i, plan in enumerate(plans):
        cid = f"fused_silu_mul_fp8_quant_v{i:03d}"
        kernel_code = render_fused_silu_mul_fp8_quant(plan, shape)
        tile_cost = estimate_tile_cost(
            E=shape["E"],
            T=shape["T"],
            H=shape["H"],
            group_size=shape["group_size"],
            tile_plan=plan,
        )
        metadata = _build_metadata(
            candidate_id=cid,
            shape=shape,
            topology=topology,
            tile_plan=plan,
            topology_plan=topology_plan,
            shape_cost=shape_cost,
            tile_cost=tile_cost,
        )
        notes = _NOTES_TEMPLATE.format(
            candidate_id=cid,
            arch=topology.arch,
            topology_name=topology.name,
            block_m=plan.block_m,
            block_h=plan.block_h,
            group_size=shape["group_size"],
            num_warps=plan.num_warps,
            fused_bytes=tile_cost.fused_bytes,
            unfused_bytes=tile_cost.unfused_bytes,
            bytes_saved=tile_cost.bytes_saved,
            score=tile_cost.score,
            shape_fused_bytes=shape_cost.fused_bytes,
            shape_unfused_bytes=shape_cost.unfused_bytes,
            shape_bytes_saved=shape_cost.bytes_saved,
            shape_score=shape_cost.score,
        )
        _write_candidate(out_dir, cid, kernel_code, metadata, notes)
        candidates.append(
            SeedCandidate(
                candidate_id=cid,
                op_name="fused_silu_mul_fp8_quant",
                kernel_code=kernel_code,
                tile_plan=plan,
                topology_plan=topology_plan,
                cost=tile_cost,
                metadata=metadata,
                notes=notes,
            )
        )

    return candidates
