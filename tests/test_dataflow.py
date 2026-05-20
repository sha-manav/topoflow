from topoflow_prior.dataflow import fused_silu_mul_fp8_quant_dfg
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
