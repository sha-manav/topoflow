from topoflow_prior.schemas import (
    TensorSpec,
    DataflowNode,
    DataflowGraph,
    TopologySpec,
    TilePlan,
    TopologyPlan,
    CostEstimate,
    SeedCandidate,
)


def test_tensor_spec_fields():
    t = TensorSpec(name="x", shape={"E": 32, "T": 1024, "2H": 8192}, dtype="bf16", role="input")
    assert t.name == "x"
    assert t.shape["E"] == 32
    assert t.dtype == "bf16"
    assert t.role == "input"


def test_dataflow_node_default_attrs():
    n = DataflowNode(name="silu", op="silu", inputs=["gate"], outputs=["silu_out"])
    assert n.attrs == {}


def test_dataflow_graph_construction():
    g = DataflowGraph(
        op_name="op",
        tensors=[TensorSpec("x", {"N": 1}, "fp32", "input")],
        nodes=[DataflowNode("n", "noop", ["x"], ["x"])],
        fusion_opportunities=[["n"]],
    )
    assert g.op_name == "op"
    assert g.memory_bound is True
    assert g.fusion_opportunities == [["n"]]


def test_topology_spec_defaults():
    t = TopologySpec(name="MI300X", arch="gfx942", num_xcds=8)
    assert t.l2_per_xcd_mb == 4.0
    assert t.cu_per_xcd == 38


def test_tile_plan_defaults():
    p = TilePlan(block_m=16, block_h=128)
    assert p.num_warps == 4


def test_topology_plan_defaults():
    tp = TopologyPlan()
    assert tp.use_workgroup_remap is False
    assert tp.remap_kind is None
    assert tp.notes == ""


def test_cost_estimate_fields():
    c = CostEstimate(fused_bytes=100, unfused_bytes=200, bytes_saved=100, score=0.5)
    assert c.bytes_saved == 100


def test_seed_candidate_construction():
    c = SeedCandidate(
        candidate_id="x_v000",
        op_name="x",
        kernel_code="# code",
        tile_plan=TilePlan(block_m=8, block_h=128),
        topology_plan=TopologyPlan(),
        cost=CostEstimate(1, 2, 1, 0.5),
        metadata={"shape": {"E": 1}},
        notes="hello",
    )
    assert c.candidate_id == "x_v000"
    assert c.tile_plan.block_m == 8
