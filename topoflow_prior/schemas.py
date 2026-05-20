"""Dataclasses for Topo-Flow Prior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DType = Literal["fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8"]


@dataclass
class TensorSpec:
    name: str
    shape: dict[str, int | str]
    dtype: DType
    role: Literal["input", "output", "intermediate", "scale"]


@dataclass
class DataflowNode:
    name: str
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataflowGraph:
    op_name: str
    tensors: list[TensorSpec]
    nodes: list[DataflowNode]
    fusion_opportunities: list[list[str]]
    memory_bound: bool = True


@dataclass
class TopologySpec:
    name: str
    arch: str
    num_xcds: int
    l2_per_xcd_mb: float = 4.0
    cu_per_xcd: int = 38


@dataclass
class TilePlan:
    block_m: int
    block_h: int
    num_warps: int = 4


@dataclass
class TopologyPlan:
    use_workgroup_remap: bool = False
    remap_kind: str | None = None
    notes: str = ""


@dataclass
class CostEstimate:
    fused_bytes: int
    unfused_bytes: int
    bytes_saved: int
    score: float


@dataclass
class SeedCandidate:
    candidate_id: str
    op_name: str
    kernel_code: str
    tile_plan: TilePlan
    topology_plan: TopologyPlan
    cost: CostEstimate
    metadata: dict[str, Any]
    notes: str
