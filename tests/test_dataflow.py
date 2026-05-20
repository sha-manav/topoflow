from topoflow_prior.dataflow import (
    fused_bias_gelu_dropout_dfg,
    fused_rmsnorm_residual_dfg,
    fused_silu_mul_fp8_quant_dfg,
)
from topoflow_prior.schemas import DataflowGraph


def test_dfg_op_name():
    g = fused_silu_mul_fp8_quant_dfg()
    assert isinstance(g, DataflowGraph)
    assert g.op_name == "fused_silu_mul_fp8_quant"


def test_dfg_tensors():
    g = fused_silu_mul_fp8_quant_dfg()
    names = [t.name for t in g.tensors]
    assert names == ["x", "y_fp8", "scale"]
    roles = {t.name: t.role for t in g.tensors}
    assert roles == {"x": "input", "y_fp8": "output", "scale": "scale"}
    dtypes = {t.name: t.dtype for t in g.tensors}
    assert dtypes == {"x": "bf16", "y_fp8": "fp8_e4m3", "scale": "fp32"}


def test_dfg_nodes_in_order():
    g = fused_silu_mul_fp8_quant_dfg()
    names = [n.name for n in g.nodes]
    assert names == ["split", "silu", "mul", "amax", "scale_compute", "quant"]


def test_dfg_fusion_opportunity_spans_post_split():
    g = fused_silu_mul_fp8_quant_dfg()
    assert len(g.fusion_opportunities) == 1
    assert g.fusion_opportunities[0] == ["silu", "mul", "amax", "scale_compute", "quant"]


def test_dfg_amax_has_group_size():
    g = fused_silu_mul_fp8_quant_dfg()
    amax = next(n for n in g.nodes if n.name == "amax")
    assert amax.attrs["group_size"] == 128


def test_dfg_memory_bound():
    g = fused_silu_mul_fp8_quant_dfg()
    assert g.memory_bound is True


# ---------------------------------------------------------------------------
# fused_rmsnorm_residual
# ---------------------------------------------------------------------------


def test_rmsnorm_dfg_op_name_and_memory_bound():
    g = fused_rmsnorm_residual_dfg()
    assert isinstance(g, DataflowGraph)
    assert g.op_name == "fused_rmsnorm_residual"
    assert g.memory_bound is True


def test_rmsnorm_dfg_tensors():
    g = fused_rmsnorm_residual_dfg()
    names = [t.name for t in g.tensors]
    assert names == ["x", "residual", "weight", "output", "x_residual"]
    roles = {t.name: t.role for t in g.tensors}
    assert roles == {
        "x": "input",
        "residual": "input",
        "weight": "input",
        "output": "output",
        "x_residual": "output",
    }
    dtypes = {t.dtype for t in g.tensors}
    assert dtypes == {"bf16"}


def test_rmsnorm_dfg_nodes_in_order():
    g = fused_rmsnorm_residual_dfg()
    assert [n.name for n in g.nodes] == [
        "add",
        "sq",
        "mean",
        "rrms",
        "norm",
        "scale",
    ]
    mean = next(n for n in g.nodes if n.name == "mean")
    assert mean.attrs["axis"] == "N"


def test_rmsnorm_dfg_fusion_opportunity_spans_all_six_nodes():
    g = fused_rmsnorm_residual_dfg()
    assert len(g.fusion_opportunities) == 1
    assert g.fusion_opportunities[0] == ["add", "sq", "mean", "rrms", "norm", "scale"]


# ---------------------------------------------------------------------------
# fused_bias_gelu_dropout
# ---------------------------------------------------------------------------


def test_bias_gelu_dfg_op_name_and_memory_bound():
    g = fused_bias_gelu_dropout_dfg()
    assert isinstance(g, DataflowGraph)
    assert g.op_name == "fused_bias_gelu_dropout"
    assert g.memory_bound is True


def test_bias_gelu_dfg_tensors():
    g = fused_bias_gelu_dropout_dfg()
    names = [t.name for t in g.tensors]
    assert names == ["x", "bias", "y", "dropout_mask"]
    roles = {t.name: t.role for t in g.tensors}
    assert roles == {
        "x": "input",
        "bias": "input",
        "y": "output",
        "dropout_mask": "output",
    }
    dtypes = {t.name: t.dtype for t in g.tensors}
    assert dtypes["dropout_mask"] == "bool"
    assert dtypes["x"] == "bf16"
    assert dtypes["bias"] == "bf16"
    assert dtypes["y"] == "bf16"


def test_bias_gelu_dfg_nodes_and_fusion():
    g = fused_bias_gelu_dropout_dfg()
    assert [n.name for n in g.nodes] == ["add_bias", "gelu", "dropout"]
    assert g.fusion_opportunities == [["add_bias", "gelu", "dropout"]]
    dropout = next(n for n in g.nodes if n.name == "dropout")
    assert "dropout_p" in dropout.attrs
    assert "seed" in dropout.attrs
