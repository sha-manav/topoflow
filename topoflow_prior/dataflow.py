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


def fused_rmsnorm_residual_dfg() -> DataflowGraph:
    """RMSNorm with residual addition: every transformer layer's epilogue.

    Pre-norm path: x + residual -> rms-normalize -> scale by weight.
    The residual stream (x + residual) is emitted as a second output for the
    next layer, so fusion saves one HBM round trip of the activation tensor.
    """
    return DataflowGraph(
        op_name="fused_rmsnorm_residual",
        tensors=[
            TensorSpec("x", {"M": "M", "N": "N"}, "bf16", "input"),
            TensorSpec("residual", {"M": "M", "N": "N"}, "bf16", "input"),
            TensorSpec("weight", {"N": "N"}, "bf16", "input"),
            TensorSpec("output", {"M": "M", "N": "N"}, "bf16", "output"),
            TensorSpec("x_residual", {"M": "M", "N": "N"}, "bf16", "output"),
        ],
        nodes=[
            DataflowNode("add", "add", ["x", "residual"], ["x_residual"]),
            DataflowNode("sq", "square", ["x_residual"], ["x_sq"]),
            DataflowNode(
                "mean",
                "mean",
                ["x_sq"],
                ["mean_sq"],
                attrs={"axis": "N"},
            ),
            DataflowNode("rrms", "rsqrt_plus_eps", ["mean_sq"], ["rrms"]),
            DataflowNode("norm", "multiply", ["x_residual", "rrms"], ["normalized"]),
            DataflowNode("scale", "multiply", ["normalized", "weight"], ["output"]),
        ],
        fusion_opportunities=[
            ["add", "sq", "mean", "rrms", "norm", "scale"],
        ],
        memory_bound=True,
    )


def fused_bias_gelu_dropout_dfg() -> DataflowGraph:
    """Bias-add + GELU + Dropout: transformer FFN activation chain.

    Pointwise (no reductions). Fusion collapses three launches into one and
    avoids two intermediate HBM round trips (post-bias and post-GELU).
    """
    return DataflowGraph(
        op_name="fused_bias_gelu_dropout",
        tensors=[
            TensorSpec("x", {"M": "M", "N": "N"}, "bf16", "input"),
            TensorSpec("bias", {"N": "N"}, "bf16", "input"),
            TensorSpec("y", {"M": "M", "N": "N"}, "bf16", "output"),
            TensorSpec("dropout_mask", {"M": "M", "N": "N"}, "bool", "output"),
        ],
        nodes=[
            DataflowNode("add_bias", "add", ["x", "bias"], ["preact"]),
            DataflowNode("gelu", "gelu_tanh_approx", ["preact"], ["activated"]),
            DataflowNode(
                "dropout",
                "dropout",
                ["activated"],
                ["y", "dropout_mask"],
                attrs={"dropout_p": "dropout_p", "seed": "seed"},
            ),
        ],
        fusion_opportunities=[["add_bias", "gelu", "dropout"]],
        memory_bound=True,
    )
