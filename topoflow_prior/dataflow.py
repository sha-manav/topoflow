"""Hardcoded dataflow graphs per target op."""

from __future__ import annotations

from .schemas import DataflowGraph, DataflowNode, TensorSpec


def fused_silu_mul_fp8_quant_dfg() -> DataflowGraph:
    return DataflowGraph(
        op_name="fused_silu_mul_fp8_quant",
        tensors=[
            TensorSpec("x", {"E": "E", "T": "T", "2H": "2H"}, "bf16", "input"),
            TensorSpec("y_fp8", {"E": "E", "T": "T", "H": "H"}, "fp8_e4m3", "output"),
            TensorSpec("scale", {"E": "E", "T": "T", "G": "H/128"}, "fp32", "scale"),
        ],
        nodes=[
            DataflowNode("split", "split", ["x"], ["gate", "up"]),
            DataflowNode("silu", "silu", ["gate"], ["silu_out"]),
            DataflowNode("mul", "multiply", ["silu_out", "up"], ["activated"]),
            DataflowNode(
                "amax",
                "reduce_max_abs",
                ["activated"],
                ["amax"],
                attrs={"group_size": 128},
            ),
            DataflowNode("scale_compute", "div_by_fp8_max", ["amax"], ["scale"]),
            DataflowNode("quant", "fp8_quantize", ["activated", "scale"], ["y_fp8"]),
        ],
        fusion_opportunities=[["silu", "mul", "amax", "scale_compute", "quant"]],
        memory_bound=True,
    )
