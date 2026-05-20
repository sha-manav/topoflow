"""Seed bundle orchestrator: planners + cost model + template renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cost_model import (
    estimate_bias_gelu_dropout,
    estimate_fused_silu_mul_fp8_quant,
    estimate_rmsnorm_residual,
    estimate_tile_cost,
    estimate_tile_cost_bias_gelu_dropout,
    estimate_tile_cost_rmsnorm_residual,
)
from .dataflow import (
    fused_bias_gelu_dropout_dfg,
    fused_rmsnorm_residual_dfg,
    fused_silu_mul_fp8_quant_dfg,
)
from .schemas import (
    CostEstimate,
    SeedCandidate,
    TilePlan,
    TopologyPlan,
    TopologySpec,
)
from .tile_planner import (
    tile_plans_bias_gelu_dropout,
    tile_plans_fused_silu_mul_fp8,
    tile_plans_rmsnorm_residual,
)
from .triton_templates import (
    render_bias_gelu_dropout,
    render_fused_silu_mul_fp8_quant,
    render_rmsnorm_residual,
)

_COST_BLOCK = """\
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
"""


_FUSED_SILU_NOTES = """\
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

""" + _COST_BLOCK + """
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


_RMSNORM_NOTES = """\
# Topo-Flow seed: {candidate_id}

This candidate fuses RMSNorm + residual add for a transformer epilogue on
{arch} ({topology_name}). The kernel reads (x, residual, weight), computes
x_residual = x + residual in registers, normalizes against per-row RMS, and
stores both the scaled output AND x_residual (residual stream feeding the
next layer) in one pass. Intent: avoid the round trip of x_residual through
HBM that an unfused (add then rmsnorm) pipeline would incur.

## Tile plan
- BLOCK_M = {block_m} rows per program
- BLOCK_N = {block_h} (covers the full hidden dim in one row)
- num_warps = {num_warps}

""" + _COST_BLOCK + """
## Risks
- BLOCK_M * BLOCK_N * 4 fp32 bytes is held in registers between the
  reduction pass and the normalize pass; large N or BLOCK_M may spill.
- The reduction precision uses fp32 (sum-of-squares accumulator); keep this
  even if mutating BLOCK_M / num_warps.
- Numerical stability relies on the +eps before sqrt; do not remove.

## Suggested mutations for GEAK
- BLOCK_M in [1, 2, 4, 8]
- num_warps in [4, 8]
- Keep BLOCK_N = next_power_of_2(N); shrinking it forces a two-pass reduction
  this kernel does not implement.
"""


_BIAS_GELU_NOTES = """\
# Topo-Flow seed: {candidate_id}

This candidate fuses bias-add, GELU (tanh approximation), and dropout for a
transformer FFN forward on {arch} ({topology_name}). Three pointwise launches
collapse into one, eliminating two intermediate HBM round trips.

## Tile plan
- BLOCK_M = {block_m} rows per program
- BLOCK_N = {block_h} cols per program
- num_warps = {num_warps}

""" + _COST_BLOCK + """
## Risks
- Register pressure: BLOCK_M * BLOCK_N * 4 fp32 bytes for the GELU cube
  intermediate; large tiles spill.
- Dropout draws random numbers via tl.rand(seed, offset); changing the offset
  formula changes the mask and breaks reproducibility across runs.
- The output is scaled by 1/(1-dropout_p); do not move this scale out of the
  fused kernel without adjusting downstream consumers.

## Suggested mutations for GEAK
- BLOCK_M in [1, 2, 4, 8]
- BLOCK_N in [512, 1024, 2048, 4096] (keep BLOCK_N <= N)
- num_warps in [4, 8]
- Swap the tanh approximation for the erf form if the target hardware has a
  fast erf intrinsic.
"""


def _build_metadata(
    candidate_id: str,
    op_name: str,
    shape: dict[str, Any],
    topology: TopologySpec,
    tile_plan: TilePlan,
    topology_plan: TopologyPlan,
    shape_cost: CostEstimate,
    tile_cost: CostEstimate,
    *,
    dtype: dict[str, str],
    fusion_plan: list[str],
    suggested_mutations: list[str],
    block_h_label: str = "BLOCK_H",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "op": op_name,
        "target_arch": topology.arch,
        "topology": {
            "name": topology.name,
            "num_xcds": topology.num_xcds,
            "l2_per_xcd_mb": topology.l2_per_xcd_mb,
            "cu_per_xcd": topology.cu_per_xcd,
        },
        "shape": dict(shape),
        "dtype": dict(dtype),
        "tile_plan": {
            "BLOCK_M": tile_plan.block_m,
            block_h_label: tile_plan.block_h,
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
        "fusion_plan": list(fusion_plan),
        "suggested_mutations": list(suggested_mutations),
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
            op_name="fused_silu_mul_fp8_quant",
            shape=shape,
            topology=topology,
            tile_plan=plan,
            topology_plan=topology_plan,
            shape_cost=shape_cost,
            tile_cost=tile_cost,
            dtype={"input": "bf16", "output": "fp8_e4m3", "scale": "fp32"},
            fusion_plan=["silu", "mul", "group_amax", "scale", "fp8_quantize"],
            suggested_mutations=[
                "BLOCK_M in [8, 16, 32]",
                "BLOCK_H in [64, 128, 256] (bh % gs == 0 or gs % bh == 0)",
                "num_warps in [4, 8]",
            ],
            block_h_label="BLOCK_H",
        )
        notes = _FUSED_SILU_NOTES.format(
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


def generate_seeds_for_rmsnorm_residual(
    shape: dict[str, Any],
    topology: TopologySpec,
    out_dir: str | Path,
) -> list[SeedCandidate]:
    """Produce one candidate per RMSNorm+residual tile plan.

    shape must contain keys M and N.
    """
    required = {"M", "N"}
    missing = required - set(shape)
    if missing:
        raise ValueError(f"shape missing required keys: {sorted(missing)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ = fused_rmsnorm_residual_dfg()

    shape_cost = estimate_rmsnorm_residual(M=shape["M"], N=shape["N"])

    plans = tile_plans_rmsnorm_residual(N=shape["N"])
    topology_plan = TopologyPlan(
        use_workgroup_remap=False,
        remap_kind=None,
        notes="full-row tile; mean-of-squares is one tl.sum",
    )

    candidates: list[SeedCandidate] = []
    for i, plan in enumerate(plans):
        cid = f"rmsnorm_residual_v{i:03d}"
        kernel_code = render_rmsnorm_residual(plan, shape)
        tile_cost = estimate_tile_cost_rmsnorm_residual(
            M=shape["M"], N=shape["N"], tile_plan=plan
        )
        metadata = _build_metadata(
            candidate_id=cid,
            op_name="fused_rmsnorm_residual",
            shape=shape,
            topology=topology,
            tile_plan=plan,
            topology_plan=topology_plan,
            shape_cost=shape_cost,
            tile_cost=tile_cost,
            dtype={"input": "bf16", "output": "bf16", "weight": "bf16"},
            fusion_plan=["add", "square", "mean", "rsqrt", "norm", "scale"],
            suggested_mutations=[
                "BLOCK_M in [1, 2, 4, 8]",
                "num_warps in [4, 8]",
                "Keep BLOCK_N = next_power_of_2(N); single-row coverage required",
            ],
            block_h_label="BLOCK_N",
        )
        notes = _RMSNORM_NOTES.format(
            candidate_id=cid,
            arch=topology.arch,
            topology_name=topology.name,
            block_m=plan.block_m,
            block_h=plan.block_h,
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
                op_name="fused_rmsnorm_residual",
                kernel_code=kernel_code,
                tile_plan=plan,
                topology_plan=topology_plan,
                cost=tile_cost,
                metadata=metadata,
                notes=notes,
            )
        )
    return candidates


def generate_seeds_for_bias_gelu_dropout(
    shape: dict[str, Any],
    topology: TopologySpec,
    out_dir: str | Path,
) -> list[SeedCandidate]:
    """Produce one candidate per bias+GELU+dropout tile plan.

    shape must contain keys M and N. Optionally ``dropout_p`` and ``seed`` are
    passed through into metadata (the kernel itself takes them as runtime args).
    """
    required = {"M", "N"}
    missing = required - set(shape)
    if missing:
        raise ValueError(f"shape missing required keys: {sorted(missing)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ = fused_bias_gelu_dropout_dfg()

    shape_cost = estimate_bias_gelu_dropout(M=shape["M"], N=shape["N"])

    plans = tile_plans_bias_gelu_dropout(N=shape["N"])
    topology_plan = TopologyPlan(
        use_workgroup_remap=False,
        remap_kind=None,
        notes="pointwise tile; BLOCK_M and BLOCK_N vary independently",
    )

    candidates: list[SeedCandidate] = []
    for i, plan in enumerate(plans):
        cid = f"bias_gelu_dropout_v{i:03d}"
        kernel_code = render_bias_gelu_dropout(plan, shape)
        tile_cost = estimate_tile_cost_bias_gelu_dropout(
            M=shape["M"], N=shape["N"], tile_plan=plan
        )
        metadata = _build_metadata(
            candidate_id=cid,
            op_name="fused_bias_gelu_dropout",
            shape=shape,
            topology=topology,
            tile_plan=plan,
            topology_plan=topology_plan,
            shape_cost=shape_cost,
            tile_cost=tile_cost,
            dtype={"input": "bf16", "output": "bf16", "mask": "bool"},
            fusion_plan=["add_bias", "gelu_tanh", "dropout"],
            suggested_mutations=[
                "BLOCK_M in [1, 2, 4, 8]",
                "BLOCK_N in [512, 1024, 2048, 4096] (keep BLOCK_N <= N)",
                "num_warps in [4, 8]",
            ],
            block_h_label="BLOCK_N",
        )
        notes = _BIAS_GELU_NOTES.format(
            candidate_id=cid,
            arch=topology.arch,
            topology_name=topology.name,
            block_m=plan.block_m,
            block_h=plan.block_h,
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
                op_name="fused_bias_gelu_dropout",
                kernel_code=kernel_code,
                tile_plan=plan,
                topology_plan=topology_plan,
                cost=tile_cost,
                metadata=metadata,
                notes=notes,
            )
        )
    return candidates
