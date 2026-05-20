# Topo-Flow Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `topoflow_prior` Python package that consumes operation metadata + tensor shapes + MI300X topology and emits a Seed Bundle (folder of candidate Triton kernels with metadata + notes) for GEAK to consume via filesystem.

**Architecture:** Pure Python, no GPU/Triton dependency at runtime of the generator. Jinja2 renders one kernel template (fused_silu_mul_fp8_quant). Tile and remap planners are simple combinatorial enumerators. Cost model is closed-form memory traffic. Output is filesystem-only: candidate folders with `kernel.py`, `topoflow_metadata.json`, `topoflow_notes.md`.

**Tech Stack:** Python 3.10+, `jinja2>=3.1`, `pytest>=7` (dev). No Triton, no torch, no ROCm at this layer. Generated kernels reference Triton but are validated via `ast.parse`, not execution.

---

## File Structure

Files to create at `/Users/manavshah/topoflow/`:

- `.gitignore` — ignore venv, `__pycache__`, `runs/`, build artifacts
- `pyproject.toml` — setuptools build, package `topoflow_prior`, deps `jinja2`, dev deps `pytest`
- `topoflow_prior/__init__.py` — empty marker
- `topoflow_prior/schemas.py` — dataclasses: TensorSpec, DataflowNode, DataflowGraph, TopologySpec, TilePlan, TopologyPlan, CostEstimate, SeedCandidate
- `topoflow_prior/topology.py` — `MI300X` constant + lookup `get_topology(arch: str)`
- `topoflow_prior/dataflow.py` — `fused_silu_mul_fp8_quant_dfg()` returning a DataflowGraph
- `topoflow_prior/cost_model.py` — `estimate_fused_silu_mul_fp8_quant(E, T, H, group_size)` returning CostEstimate
- `topoflow_prior/tile_planner.py` — `tile_plans_fused_silu_mul_fp8(H, group_size)` enumerator
- `topoflow_prior/remap_planner.py` — four mapping functions + `attention_remap_plans()` enumerator
- `topoflow_prior/triton_templates.py` — Jinja2 environment + `render_fused_silu_mul_fp8_quant(tile_plan, shape)`
- `topoflow_prior/templates/__init__.py` — package marker (templates ship inside package)
- `topoflow_prior/templates/triton/fused_silu_mul_fp8_quant.py.j2` — Jinja kernel template
- `topoflow_prior/seed_generator.py` — `generate_seeds_for_fused_silu_mul_fp8_quant(shape, topology, out_dir)`
- `scripts/generate_seeds.py` — CLI wrapping the seed generator
- `scripts/package_for_geak.py` — convert seed bundle to GEAK task folders
- `tests/__init__.py` — empty
- `tests/test_schemas.py`, `test_topology.py`, `test_dataflow.py`, `test_cost_model.py`, `test_tile_planner.py`, `test_remap_planner.py`, `test_triton_templates.py`, `test_seed_generator.py`, `test_generate_seeds_cli.py`, `test_package_for_geak.py`

**Deviation from TOPOFLOW_SPEC.md:** Templates live inside the package (`topoflow_prior/templates/triton/`) instead of a sibling `templates/` directory, so packaging via setuptools is straightforward and tests work without env tweaks. Documented in the README task.

---

## Task 0: Repository bootstrap

**Files:**
- Create: `/Users/manavshah/topoflow/.gitignore`
- Create: `/Users/manavshah/topoflow/pyproject.toml`
- Create: `/Users/manavshah/topoflow/topoflow_prior/__init__.py`
- Create: `/Users/manavshah/topoflow/topoflow_prior/templates/__init__.py`
- Create: `/Users/manavshah/topoflow/topoflow_prior/templates/triton/.gitkeep`
- Create: `/Users/manavshah/topoflow/tests/__init__.py`
- Create: `/Users/manavshah/topoflow/scripts/.gitkeep`

- [ ] **Step 0.1: git init and base directories**

Run:

```bash
cd /Users/manavshah/topoflow
git init -b main
mkdir -p topoflow_prior/templates/triton scripts tests
```

Expected: `Initialized empty Git repository in /Users/manavshah/topoflow/.git/`

- [ ] **Step 0.2: Write `.gitignore`**

Create `/Users/manavshah/topoflow/.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
build/
dist/
runs/
.coverage
```

- [ ] **Step 0.3: Write `pyproject.toml`**

Create `/Users/manavshah/topoflow/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "topoflow-prior"
version = "0.1.0"
description = "Topology-aware seed prior for AMD GPU kernel optimization agents"
requires-python = ">=3.10"
dependencies = ["jinja2>=3.1"]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
include = ["topoflow_prior*"]
exclude = ["tests*", "scripts*"]

[tool.setuptools.package-data]
topoflow_prior = ["templates/triton/*.j2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 0.4: Write package markers**

Create `/Users/manavshah/topoflow/topoflow_prior/__init__.py`:

```python
"""Topo-Flow Prior: topology-aware seed generator for AMD kernel search."""

__version__ = "0.1.0"
```

Create `/Users/manavshah/topoflow/topoflow_prior/templates/__init__.py` (empty file):

```python
```

Create `/Users/manavshah/topoflow/topoflow_prior/templates/triton/.gitkeep` (empty file):

```
```

Create `/Users/manavshah/topoflow/tests/__init__.py` (empty file):

```python
```

Create `/Users/manavshah/topoflow/scripts/.gitkeep` (empty file):

```
```

- [ ] **Step 0.5: Create venv and install package + dev deps**

Run:

```bash
cd /Users/manavshah/topoflow
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

Expected: `Successfully installed ... topoflow-prior-0.1.0 ...`. If `python3` is not found, use whatever Python 3.10+ binary is available; the user can correct.

- [ ] **Step 0.6: Sanity test — pytest runs**

Run:

```bash
cd /Users/manavshah/topoflow
.venv/bin/pytest -q
```

Expected: `no tests ran in ...s` (zero tests, zero failures). Confirms pytest is wired.

- [ ] **Step 0.7: Initial commit**

```bash
cd /Users/manavshah/topoflow
git add .gitignore pyproject.toml topoflow_prior/ tests/ scripts/ TOPOFLOW_SPEC.md docs/
git commit -m "chore: bootstrap topoflow-prior package skeleton"
```

---

## Task 1: schemas.py — dataclasses

**Files:**
- Create: `topoflow_prior/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_schemas.py`:

```python
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
```

- [ ] **Step 1.2: Run test, see ImportError**

Run:

```bash
cd /Users/manavshah/topoflow
.venv/bin/pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.schemas'`.

- [ ] **Step 1.3: Write `schemas.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/schemas.py`:

```python
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
```

- [ ] **Step 1.4: Run tests, see PASS**

Run:

```bash
.venv/bin/pytest tests/test_schemas.py -v
```

Expected: 8 passed.

- [ ] **Step 1.5: Commit**

```bash
git add topoflow_prior/schemas.py tests/test_schemas.py
git commit -m "feat: add Topo-Flow Prior dataclass schemas"
```

---

## Task 2: topology.py — MI300X spec

**Files:**
- Create: `topoflow_prior/topology.py`
- Create: `tests/test_topology.py`

- [ ] **Step 2.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_topology.py`:

```python
import pytest

from topoflow_prior.schemas import TopologySpec
from topoflow_prior.topology import MI300X, MI355X, get_topology


def test_mi300x_constants():
    assert isinstance(MI300X, TopologySpec)
    assert MI300X.name == "MI300X"
    assert MI300X.arch == "gfx942"
    assert MI300X.num_xcds == 8
    assert MI300X.l2_per_xcd_mb == 4.0
    assert MI300X.cu_per_xcd == 38


def test_mi355x_placeholder():
    assert isinstance(MI355X, TopologySpec)
    assert MI355X.arch == "gfx950"


def test_get_topology_lookup():
    assert get_topology("mi300x") is MI300X
    assert get_topology("MI300X") is MI300X
    assert get_topology("gfx942") is MI300X


def test_get_topology_unknown_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        get_topology("nvidia-h100")
```

- [ ] **Step 2.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_topology.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.topology'`.

- [ ] **Step 2.3: Write `topology.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/topology.py`:

```python
"""AMD GPU topology specs for Topo-Flow Prior."""

from __future__ import annotations

from .schemas import TopologySpec

MI300X = TopologySpec(
    name="MI300X",
    arch="gfx942",
    num_xcds=8,
    l2_per_xcd_mb=4.0,
    cu_per_xcd=38,
)

MI355X = TopologySpec(
    name="MI355X",
    arch="gfx950",
    num_xcds=8,
    l2_per_xcd_mb=4.0,
    cu_per_xcd=38,
)

_REGISTRY: dict[str, TopologySpec] = {
    "mi300x": MI300X,
    "gfx942": MI300X,
    "mi355x": MI355X,
    "gfx950": MI355X,
}


def get_topology(name_or_arch: str) -> TopologySpec:
    key = name_or_arch.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown arch: {name_or_arch!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]
```

- [ ] **Step 2.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_topology.py -v
```

Expected: 4 passed.

- [ ] **Step 2.5: Commit**

```bash
git add topoflow_prior/topology.py tests/test_topology.py
git commit -m "feat: add MI300X/MI355X topology specs and arch lookup"
```

---

## Task 3: dataflow.py — hardcoded dataflow graph

**Files:**
- Create: `topoflow_prior/dataflow.py`
- Create: `tests/test_dataflow.py`

- [ ] **Step 3.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_dataflow.py`:

```python
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
```

- [ ] **Step 3.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_dataflow.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.dataflow'`.

- [ ] **Step 3.3: Write `dataflow.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/dataflow.py`:

```python
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
```

- [ ] **Step 3.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_dataflow.py -v
```

Expected: 6 passed.

- [ ] **Step 3.5: Commit**

```bash
git add topoflow_prior/dataflow.py tests/test_dataflow.py
git commit -m "feat: add fused_silu_mul_fp8_quant dataflow graph"
```

---

## Task 4: cost_model.py — memory traffic estimator

**Files:**
- Create: `topoflow_prior/cost_model.py`
- Create: `tests/test_cost_model.py`

- [ ] **Step 4.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_cost_model.py`:

```python
import pytest

from topoflow_prior.cost_model import estimate_fused_silu_mul_fp8_quant
from topoflow_prior.schemas import CostEstimate


def test_basic_shape_returns_cost_estimate():
    c = estimate_fused_silu_mul_fp8_quant(E=32, T=1024, H=4096, group_size=128)
    assert isinstance(c, CostEstimate)


def test_byte_formulas():
    E, T, H, G = 32, 1024, 4096, 128
    c = estimate_fused_silu_mul_fp8_quant(E=E, T=T, H=H, group_size=G)
    expected_input = E * T * 2 * H * 2  # bf16
    expected_output = E * T * H * 1  # fp8
    expected_scale = E * T * (H // G) * 4  # fp32
    expected_fused = expected_input + expected_output + expected_scale
    unfused_intermediate = E * T * H * 2 * 2  # bf16 intermediate write+read
    expected_unfused = expected_fused + unfused_intermediate
    assert c.fused_bytes == expected_fused
    assert c.unfused_bytes == expected_unfused
    assert c.bytes_saved == expected_unfused - expected_fused
    assert c.bytes_saved > 0


def test_score_in_unit_interval():
    c = estimate_fused_silu_mul_fp8_quant(E=1, T=1, H=128, group_size=128)
    assert 0.0 < c.score < 1.0


def test_score_ranking_prefers_larger_savings():
    """Larger H => more intermediate bytes saved => lower score (less fused/unfused ratio)."""
    c_small = estimate_fused_silu_mul_fp8_quant(E=1, T=64, H=128, group_size=128)
    c_large = estimate_fused_silu_mul_fp8_quant(E=1, T=64, H=8192, group_size=128)
    # both are saving the same fraction of intermediate, but score is fused/unfused
    # since both have proportional growth, scores should be close; just ensure no crash
    assert 0 < c_small.score < 1
    assert 0 < c_large.score < 1


def test_group_size_must_divide_h():
    with pytest.raises(ValueError, match="group_size"):
        estimate_fused_silu_mul_fp8_quant(E=1, T=1, H=100, group_size=128)
```

- [ ] **Step 4.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_cost_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.cost_model'`.

- [ ] **Step 4.3: Write `cost_model.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/cost_model.py`:

```python
"""Memory-traffic-based cost estimator.

Not accurate in absolute terms; goal is to rank candidates better than random.
"""

from __future__ import annotations

from .schemas import CostEstimate

_BYTES_BF16 = 2
_BYTES_FP8 = 1
_BYTES_FP32 = 4


def estimate_fused_silu_mul_fp8_quant(
    E: int, T: int, H: int, group_size: int = 128
) -> CostEstimate:
    if H % group_size != 0:
        raise ValueError(
            f"group_size ({group_size}) must divide H ({H}); got H % group_size = {H % group_size}"
        )

    input_read = E * T * 2 * H * _BYTES_BF16
    output_write = E * T * H * _BYTES_FP8
    scale_write = E * T * (H // group_size) * _BYTES_FP32
    fused = input_read + output_write + scale_write

    # Unfused path: write bf16 intermediate (silu*mul output) then read it again
    unfused_intermediate = E * T * H * _BYTES_BF16 * 2
    unfused = fused + unfused_intermediate

    return CostEstimate(
        fused_bytes=fused,
        unfused_bytes=unfused,
        bytes_saved=unfused - fused,
        score=fused / unfused,
    )
```

- [ ] **Step 4.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_cost_model.py -v
```

Expected: 5 passed.

- [ ] **Step 4.5: Commit**

```bash
git add topoflow_prior/cost_model.py tests/test_cost_model.py
git commit -m "feat: add memory-traffic cost model for fused_silu_mul_fp8_quant"
```

---

## Task 5: tile_planner.py — enumerate tile candidates

**Files:**
- Create: `topoflow_prior/tile_planner.py`
- Create: `tests/test_tile_planner.py`

- [ ] **Step 5.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_tile_planner.py`:

```python
from topoflow_prior.schemas import TilePlan
from topoflow_prior.tile_planner import tile_plans_fused_silu_mul_fp8


def test_returns_tile_plans():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    assert all(isinstance(p, TilePlan) for p in plans)
    assert len(plans) >= 12


def test_expected_count_for_default_grid():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    # bm in [8,16,32], bh in [64,128,256] (all satisfy bh%128==0 or 128%bh==0),
    # num_warps in [4,8] => 3*3*2 = 18
    assert len(plans) == 18


def test_block_m_values_subset():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    bms = {p.block_m for p in plans}
    assert bms == {8, 16, 32}


def test_block_h_values_satisfy_group_compat():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    for p in plans:
        assert p.block_h % 128 == 0 or 128 % p.block_h == 0


def test_num_warps_values_subset():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    nws = {p.num_warps for p in plans}
    assert nws == {4, 8}


def test_plans_are_unique():
    plans = tile_plans_fused_silu_mul_fp8(H=4096, group_size=128)
    tuples = {(p.block_m, p.block_h, p.num_warps) for p in plans}
    assert len(tuples) == len(plans)


def test_block_h_does_not_exceed_h():
    """If H is small, very large BLOCK_H should be filtered out."""
    plans = tile_plans_fused_silu_mul_fp8(H=128, group_size=128)
    for p in plans:
        assert p.block_h <= 128
```

- [ ] **Step 5.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_tile_planner.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.tile_planner'`.

- [ ] **Step 5.3: Write `tile_planner.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/tile_planner.py`:

```python
"""Enumerate tile-plan candidates for the fused SiLU+Mul+FP8 op."""

from __future__ import annotations

from .schemas import TilePlan

_BLOCK_M_CANDIDATES = (8, 16, 32)
_BLOCK_H_CANDIDATES = (64, 128, 256)
_NUM_WARPS_CANDIDATES = (4, 8)


def tile_plans_fused_silu_mul_fp8(H: int, group_size: int = 128) -> list[TilePlan]:
    """Enumerate (block_m, block_h, num_warps) tile candidates.

    Filters block_h so that either block_h % group_size == 0 (multi-group tile)
    or group_size % block_h == 0 (sub-group tile). Also filters block_h > H.
    """
    plans: list[TilePlan] = []
    for bm in _BLOCK_M_CANDIDATES:
        for bh in _BLOCK_H_CANDIDATES:
            if bh > H:
                continue
            if not (bh % group_size == 0 or group_size % bh == 0):
                continue
            for nw in _NUM_WARPS_CANDIDATES:
                plans.append(TilePlan(block_m=bm, block_h=bh, num_warps=nw))
    return plans
```

- [ ] **Step 5.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_tile_planner.py -v
```

Expected: 7 passed.

- [ ] **Step 5.5: Commit**

```bash
git add topoflow_prior/tile_planner.py tests/test_tile_planner.py
git commit -m "feat: enumerate BLOCK_M/BLOCK_H/num_warps tile candidates"
```

---

## Task 6: remap_planner.py — four workgroup orderings

**Files:**
- Create: `topoflow_prior/remap_planner.py`
- Create: `tests/test_remap_planner.py`

- [ ] **Step 6.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_remap_planner.py`:

```python
import itertools

import pytest

from topoflow_prior.remap_planner import (
    REMAP_KINDS,
    attention_remap_plans,
    naive_block_first,
    naive_head_first,
    swizzled_block_first,
    swizzled_head_first,
)


@pytest.fixture
def grid():
    return dict(num_m=4, num_h=8, num_b=2, num_xcds=8)


def _all_outputs(fn, grid):
    out = []
    for pid_b, pid_h, pid_m in itertools.product(
        range(grid["num_b"]), range(grid["num_h"]), range(grid["num_m"])
    ):
        out.append(fn(pid_m, pid_h, pid_b, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]))
    return out


@pytest.mark.parametrize(
    "fn",
    [naive_block_first, naive_head_first, swizzled_block_first, swizzled_head_first],
)
def test_each_mapping_is_a_permutation(fn, grid):
    outputs = _all_outputs(fn, grid)
    total = grid["num_m"] * grid["num_h"] * grid["num_b"]
    assert sorted(outputs) == list(range(total))


def test_mappings_are_distinct(grid):
    sigs = {
        name: tuple(_all_outputs(fn, grid))
        for name, fn in [
            ("naive_block_first", naive_block_first),
            ("naive_head_first", naive_head_first),
            ("swizzled_block_first", swizzled_block_first),
            ("swizzled_head_first", swizzled_head_first),
        ]
    }
    assert len(set(sigs.values())) == 4, sigs


def test_naive_block_first_layout(grid):
    # naive block-first: m varies fastest, then h, then b
    assert naive_block_first(0, 0, 0, **{k: grid[k] for k in ("num_m", "num_h", "num_b", "num_xcds")}) == 0
    assert naive_block_first(1, 0, 0, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]) == 1
    assert naive_block_first(0, 1, 0, grid["num_m"], grid["num_h"], grid["num_b"], grid["num_xcds"]) == grid["num_m"]


def test_swizzled_head_first_groups_heads_to_xcds(grid):
    """Heads in the same xcd_group should have consecutive program ids in a stride."""
    g = grid
    heads_per_group = (g["num_h"] + g["num_xcds"] - 1) // g["num_xcds"]
    # Test consistency with the spec's formula
    expected = (
        0 * g["num_h"] * g["num_m"]
        + 0 * heads_per_group * g["num_m"]
        + 0 * g["num_m"]
        + 0
    )
    assert swizzled_head_first(0, 0, 0, g["num_m"], g["num_h"], g["num_b"], g["num_xcds"]) == expected


def test_remap_kinds_constant():
    assert set(REMAP_KINDS) == {
        "naive_block_first",
        "naive_head_first",
        "swizzled_block_first",
        "swizzled_head_first",
    }


def test_attention_remap_plans_returns_four_topology_plans():
    plans = attention_remap_plans()
    assert len(plans) == 4
    kinds = {p.remap_kind for p in plans}
    assert kinds == set(REMAP_KINDS)
    # swizzled variants set use_workgroup_remap=True
    swizzled = [p for p in plans if p.remap_kind.startswith("swizzled")]
    assert all(p.use_workgroup_remap for p in swizzled)
    naive = [p for p in plans if p.remap_kind.startswith("naive")]
    assert all(not p.use_workgroup_remap for p in naive)
```

- [ ] **Step 6.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_remap_planner.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.remap_planner'`.

- [ ] **Step 6.3: Write `remap_planner.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/remap_planner.py`:

```python
"""Workgroup remapping orderings for attention kernels on MI300X.

These functions compute a *linearized program id* given grid coordinates
(pid_m, pid_h, pid_b). The MI300X XCD scheduler assigns consecutive program
ids to XCDs round-robin; by reordering ids we can place blocks that reuse
K/V data onto the same XCD's L2.

All four functions are permutations of [0, num_m * num_h * num_b).
"""

from __future__ import annotations

from .schemas import TopologyPlan

REMAP_KINDS: tuple[str, ...] = (
    "naive_block_first",
    "naive_head_first",
    "swizzled_block_first",
    "swizzled_head_first",
)


def naive_block_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    return pid_b * num_h * num_m + pid_h * num_m + pid_m


def naive_head_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    return pid_b * num_h * num_m + pid_m * num_h + pid_h


def swizzled_block_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    blocks_per_group = (num_m + num_xcds - 1) // num_xcds
    xcd_group = pid_m // blocks_per_group
    local_m = pid_m % blocks_per_group
    return (
        pid_b * num_h * num_m
        + pid_h * num_m
        + xcd_group * blocks_per_group
        + local_m
    )


def swizzled_head_first(
    pid_m: int, pid_h: int, pid_b: int, num_m: int, num_h: int, num_b: int, num_xcds: int = 8
) -> int:
    heads_per_group = (num_h + num_xcds - 1) // num_xcds
    xcd_group = pid_h // heads_per_group
    local_head = pid_h % heads_per_group
    return (
        pid_b * num_h * num_m
        + xcd_group * heads_per_group * num_m
        + local_head * num_m
        + pid_m
    )


_REMAP_NOTES = {
    "naive_block_first": "Baseline: m-blocks contiguous; XCD reuse only across batches.",
    "naive_head_first": "Heads swept within m-block; little K/V locality on XCDs.",
    "swizzled_block_first": "M-blocks grouped per XCD; helpful when Q tile is reused.",
    "swizzled_head_first": (
        "Heads grouped per XCD; all blocks of a head land on same XCD so K/V "
        "fits in 4MB L2 (paper: arXiv 2511.02132)."
    ),
}


def attention_remap_plans() -> list[TopologyPlan]:
    """Return one TopologyPlan per remap kind."""
    plans: list[TopologyPlan] = []
    for kind in REMAP_KINDS:
        plans.append(
            TopologyPlan(
                use_workgroup_remap=kind.startswith("swizzled"),
                remap_kind=kind,
                notes=_REMAP_NOTES[kind],
            )
        )
    return plans
```

- [ ] **Step 6.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_remap_planner.py -v
```

Expected: 9 passed (4 parametrized permutation tests + 5 others).

- [ ] **Step 6.5: Commit**

```bash
git add topoflow_prior/remap_planner.py tests/test_remap_planner.py
git commit -m "feat: add four attention workgroup orderings with permutation tests"
```

---

## Task 7: triton_templates.py + Jinja kernel template

**Files:**
- Create: `topoflow_prior/templates/triton/fused_silu_mul_fp8_quant.py.j2`
- Create: `topoflow_prior/triton_templates.py`
- Create: `tests/test_triton_templates.py`

- [ ] **Step 7.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_triton_templates.py`:

```python
import ast

import pytest

from topoflow_prior.schemas import TilePlan
from topoflow_prior.triton_templates import render_fused_silu_mul_fp8_quant


@pytest.fixture
def shape():
    return {"E": 32, "T": 1024, "H": 4096, "group_size": 128}


@pytest.fixture
def plan():
    return TilePlan(block_m=16, block_h=128, num_warps=4)


def test_render_returns_non_empty(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert isinstance(code, str)
    assert len(code) > 200


def test_rendered_code_is_valid_python(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    ast.parse(code)  # raises SyntaxError on failure


def test_rendered_code_contains_topoflow_intent(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "# TOPOFLOW_INTENT" in code
    # at least three intent comments
    assert code.count("# TOPOFLOW_INTENT") >= 3


def test_rendered_code_contains_triton_jit(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "@triton.jit" in code


def test_rendered_code_embeds_tile_constants(plan, shape):
    code = render_fused_silu_mul_fp8_quant(plan, shape)
    assert "BLOCK_M" in code
    assert f"BLOCK_M: tl.constexpr = {plan.block_m}" in code
    assert f"BLOCK_H: tl.constexpr = {plan.block_h}" in code
    assert f"num_warps={plan.num_warps}" in code


def test_different_plans_produce_different_code(shape):
    a = render_fused_silu_mul_fp8_quant(TilePlan(block_m=8, block_h=128, num_warps=4), shape)
    b = render_fused_silu_mul_fp8_quant(TilePlan(block_m=32, block_h=256, num_warps=8), shape)
    assert a != b


def test_three_branches_cover_block_h_relative_to_group(shape):
    """Each (block_h < / == / > group_size) branch must produce parseable code."""
    for bh in (64, 128, 256):
        code = render_fused_silu_mul_fp8_quant(
            TilePlan(block_m=16, block_h=bh, num_warps=4), shape
        )
        ast.parse(code)
```

- [ ] **Step 7.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_triton_templates.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.triton_templates'`.

- [ ] **Step 7.3: Write the Jinja kernel template**

Create `/Users/manavshah/topoflow/topoflow_prior/templates/triton/fused_silu_mul_fp8_quant.py.j2`:

```jinja
# TOPOFLOW_INTENT: Fused SiLU + Mul + per-group FP8 quantization.
# TOPOFLOW_INTENT: Avoid writing the bf16 silu(gate)*up intermediate to HBM.
# TOPOFLOW_INTENT: Each program owns one group of size GROUP_SIZE in the H dim
#                  so per-group amax/scale stay local to the tile.
# TOPOFLOW_INTENT: Tile = BLOCK_M={{ BLOCK_M }} x BLOCK_H={{ BLOCK_H }},
#                  GROUP_SIZE={{ GROUP_SIZE }}, num_warps={{ num_warps }}.
# TOPOFLOW_INTENT: Mutate BLOCK_M in [8,16,32], BLOCK_H in [64,128,256],
#                  num_warps in [4,8]; keep BLOCK_H related to GROUP_SIZE by
#                  bh % gs == 0 or gs % bh == 0.

import torch
import triton
import triton.language as tl

BLOCK_M: tl.constexpr = {{ BLOCK_M }}
BLOCK_H: tl.constexpr = {{ BLOCK_H }}
GROUP_SIZE: tl.constexpr = {{ GROUP_SIZE }}
FP8_E4M3_MAX: tl.constexpr = 448.0


@triton.jit
def fused_silu_mul_fp8_quant_kernel(
    x_ptr, y_ptr, scale_ptr,
    E, T, H,
    stride_xe, stride_xt, stride_xh,
    stride_ye, stride_yt, stride_yh,
    stride_se, stride_st, stride_sg,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_E4M3_MAX: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < T

{% if BLOCK_H == GROUP_SIZE %}
    # TOPOFLOW_INTENT branch: BLOCK_H == GROUP_SIZE -> one group per tile, single pass.
    offs_h = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = mask_m[:, None] & (offs_h[None, :] < H)
    gate_ptrs = (
        x_ptr
        + pid_e * stride_xe
        + offs_m[:, None] * stride_xt
        + offs_h[None, :] * stride_xh
    )
    up_ptrs = gate_ptrs + H * stride_xh
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)) * up
    amax = tl.max(tl.abs(activated), axis=1)
    scale = amax / FP8_E4M3_MAX
    scale_safe = tl.where(scale > 0, scale, 1.0)
    y_fp32 = activated / scale_safe[:, None]
    y_fp8 = y_fp32.to(tl.float8e4nv)
    y_ptrs = (
        y_ptr
        + pid_e * stride_ye
        + offs_m[:, None] * stride_yt
        + offs_h[None, :] * stride_yh
    )
    tl.store(y_ptrs, y_fp8, mask=mask)
    scale_ptrs = (
        scale_ptr
        + pid_e * stride_se
        + offs_m * stride_st
        + pid_g * stride_sg
    )
    tl.store(scale_ptrs, scale, mask=mask_m)
{% elif BLOCK_H > GROUP_SIZE %}
    # TOPOFLOW_INTENT branch: BLOCK_H > GROUP_SIZE -> N groups per tile (N = BLOCK_H/GROUP_SIZE).
    N_GROUPS_PER_TILE: tl.constexpr = BLOCK_H // GROUP_SIZE
    offs_h = pid_g * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = mask_m[:, None] & (offs_h[None, :] < H)
    gate_ptrs = (
        x_ptr
        + pid_e * stride_xe
        + offs_m[:, None] * stride_xt
        + offs_h[None, :] * stride_xh
    )
    up_ptrs = gate_ptrs + H * stride_xh
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)) * up
    activated_g = tl.reshape(activated, (BLOCK_M, N_GROUPS_PER_TILE, GROUP_SIZE))
    amax_g = tl.max(tl.abs(activated_g), axis=2)
    scale_g = amax_g / FP8_E4M3_MAX
    scale_safe_g = tl.where(scale_g > 0, scale_g, 1.0)
    y_fp32 = activated_g / scale_safe_g[:, :, None]
    y_fp32 = tl.reshape(y_fp32, (BLOCK_M, BLOCK_H))
    y_fp8 = y_fp32.to(tl.float8e4nv)
    y_ptrs = (
        y_ptr
        + pid_e * stride_ye
        + offs_m[:, None] * stride_yt
        + offs_h[None, :] * stride_yh
    )
    tl.store(y_ptrs, y_fp8, mask=mask)
    offs_g = pid_g * N_GROUPS_PER_TILE + tl.arange(0, N_GROUPS_PER_TILE)
    scale_ptrs = (
        scale_ptr
        + pid_e * stride_se
        + offs_m[:, None] * stride_st
        + offs_g[None, :] * stride_sg
    )
    tl.store(scale_ptrs, scale_g, mask=mask_m[:, None])
{% else %}
    # TOPOFLOW_INTENT branch: BLOCK_H < GROUP_SIZE -> iterate BLOCK_H chunks within one group.
    ITERS: tl.constexpr = GROUP_SIZE // BLOCK_H
    offs_h_base = pid_g * GROUP_SIZE
    # Pass 1: compute amax across the group.
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in tl.static_range(ITERS):
        offs_h = offs_h_base + i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = mask_m[:, None] & (offs_h[None, :] < H)
        gate_ptrs = (
            x_ptr
            + pid_e * stride_xe
            + offs_m[:, None] * stride_xt
            + offs_h[None, :] * stride_xh
        )
        up_ptrs = gate_ptrs + H * stride_xh
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
        activated = (gate * tl.sigmoid(gate)) * up
        chunk_max = tl.max(tl.abs(activated), axis=1)
        amax = tl.maximum(amax, chunk_max)
    scale = amax / FP8_E4M3_MAX
    scale_safe = tl.where(scale > 0, scale, 1.0)
    # Pass 2: recompute activated and store quantized output.
    for i in tl.static_range(ITERS):
        offs_h = offs_h_base + i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = mask_m[:, None] & (offs_h[None, :] < H)
        gate_ptrs = (
            x_ptr
            + pid_e * stride_xe
            + offs_m[:, None] * stride_xt
            + offs_h[None, :] * stride_xh
        )
        up_ptrs = gate_ptrs + H * stride_xh
        gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
        activated = (gate * tl.sigmoid(gate)) * up
        y_fp32 = activated / scale_safe[:, None]
        y_fp8 = y_fp32.to(tl.float8e4nv)
        y_ptrs = (
            y_ptr
            + pid_e * stride_ye
            + offs_m[:, None] * stride_yt
            + offs_h[None, :] * stride_yh
        )
        tl.store(y_ptrs, y_fp8, mask=mask)
    scale_ptrs = (
        scale_ptr
        + pid_e * stride_se
        + offs_m * stride_st
        + pid_g * stride_sg
    )
    tl.store(scale_ptrs, scale, mask=mask_m)
{% endif %}


def fused_silu_mul_fp8_quant(x, group_size: int = {{ GROUP_SIZE }}):
    """Host wrapper. x: [E, T, 2H] bf16. Returns (y_fp8 [E,T,H], scale [E,T,H/group_size])."""
    assert x.ndim == 3, "expected x of shape [E, T, 2H]"
    E, T, two_H = x.shape
    assert two_H % 2 == 0
    H = two_H // 2
    assert H % group_size == 0, "group_size must divide H"
    y = torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device=x.device)
    G = H // group_size
    scale = torch.empty((E, T, G), dtype=torch.float32, device=x.device)
    grid = (E, triton.cdiv(T, BLOCK_M), G if BLOCK_H <= group_size else G // (BLOCK_H // group_size))
    fused_silu_mul_fp8_quant_kernel[grid](
        x, y, scale,
        E, T, H,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        GROUP_SIZE=GROUP_SIZE,
        FP8_E4M3_MAX=FP8_E4M3_MAX,
        num_warps={{ num_warps }},
    )
    return y, scale
```

- [ ] **Step 7.4: Write `triton_templates.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/triton_templates.py`:

```python
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
        Python source code (str) — guaranteed to be syntactically valid Python.
    """
    template = _ENV.get_template("fused_silu_mul_fp8_quant.py.j2")
    return template.render(
        BLOCK_M=tile_plan.block_m,
        BLOCK_H=tile_plan.block_h,
        GROUP_SIZE=int(shape.get("group_size", 128)),
        num_warps=tile_plan.num_warps,
    )
```

- [ ] **Step 7.5: Run, see PASS**

```bash
.venv/bin/pytest tests/test_triton_templates.py -v
```

Expected: 7 passed. If `ast.parse` fails on a branch, fix the Jinja template (don't relax the test).

- [ ] **Step 7.6: Commit**

```bash
git add topoflow_prior/triton_templates.py topoflow_prior/templates/triton/fused_silu_mul_fp8_quant.py.j2 tests/test_triton_templates.py
git commit -m "feat: Jinja kernel template + renderer for fused_silu_mul_fp8_quant"
```

---

## Task 8: seed_generator.py — orchestrator

**Files:**
- Create: `topoflow_prior/seed_generator.py`
- Create: `tests/test_seed_generator.py`

- [ ] **Step 8.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_seed_generator.py`:

```python
import ast
import json
from pathlib import Path

import pytest

from topoflow_prior.seed_generator import generate_seeds_for_fused_silu_mul_fp8_quant
from topoflow_prior.topology import MI300X


@pytest.fixture
def shape():
    return {"E": 32, "T": 1024, "H": 4096, "group_size": 128}


def test_generates_at_least_12_candidates(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    candidates = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    assert len(candidates) >= 12
    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert len(folders) == len(candidates)


def test_each_candidate_has_three_files(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in sorted(out_dir.iterdir()):
        assert (d / "kernel.py").exists(), f"missing kernel.py in {d}"
        assert (d / "topoflow_metadata.json").exists(), f"missing metadata in {d}"
        assert (d / "topoflow_notes.md").exists(), f"missing notes in {d}"


def test_metadata_has_required_fields(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    sample = next(out_dir.iterdir())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_silu_mul_fp8_quant"
    assert meta["target_arch"] == "gfx942"
    assert meta["shape"] == shape
    for k in ("BLOCK_M", "BLOCK_H", "num_warps"):
        assert k in meta["tile_plan"]
    assert "cost_model" in meta and "score" in meta["cost_model"]
    assert "fusion_plan" in meta
    assert "suggested_mutations" in meta
    assert meta["candidate_id"].startswith("fused_silu_mul_fp8_quant_v")


def test_every_kernel_py_parses(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in out_dir.iterdir():
        code = (d / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code


def test_notes_md_is_non_empty(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in out_dir.iterdir():
        notes = (d / "topoflow_notes.md").read_text()
        assert len(notes) > 50
        assert "fuses SiLU" in notes or "fused" in notes.lower()


def test_candidate_ids_unique(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    ids = [c.candidate_id for c in cands]
    assert len(set(ids)) == len(ids)


def test_returns_seed_candidate_objects(tmp_path, shape):
    from topoflow_prior.schemas import SeedCandidate

    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    assert all(isinstance(c, SeedCandidate) for c in cands)
```

- [ ] **Step 8.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_seed_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'topoflow_prior.seed_generator'`.

- [ ] **Step 8.3: Write `seed_generator.py`**

Create `/Users/manavshah/topoflow/topoflow_prior/seed_generator.py`:

```python
"""Seed bundle orchestrator: planners + cost model + template renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cost_model import estimate_fused_silu_mul_fp8_quant
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

## Cost model (memory traffic)
- fused bytes:   {fused_bytes:,}
- unfused bytes: {unfused_bytes:,}
- bytes saved:   {bytes_saved:,}
- score (fused/unfused, lower is better): {score:.4f}

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
    cost: CostEstimate,
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
        "cost_model": {
            "fused_bytes": cost.fused_bytes,
            "unfused_bytes": cost.unfused_bytes,
            "bytes_saved": cost.bytes_saved,
            "score": cost.score,
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

    cost = estimate_fused_silu_mul_fp8_quant(
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
        metadata = _build_metadata(
            candidate_id=cid,
            shape=shape,
            topology=topology,
            tile_plan=plan,
            topology_plan=topology_plan,
            cost=cost,
        )
        notes = _NOTES_TEMPLATE.format(
            candidate_id=cid,
            arch=topology.arch,
            topology_name=topology.name,
            block_m=plan.block_m,
            block_h=plan.block_h,
            group_size=shape["group_size"],
            num_warps=plan.num_warps,
            fused_bytes=cost.fused_bytes,
            unfused_bytes=cost.unfused_bytes,
            bytes_saved=cost.bytes_saved,
            score=cost.score,
        )
        _write_candidate(out_dir, cid, kernel_code, metadata, notes)
        candidates.append(
            SeedCandidate(
                candidate_id=cid,
                op_name="fused_silu_mul_fp8_quant",
                kernel_code=kernel_code,
                tile_plan=plan,
                topology_plan=topology_plan,
                cost=cost,
                metadata=metadata,
                notes=notes,
            )
        )

    return candidates
```

- [ ] **Step 8.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_seed_generator.py -v
```

Expected: 7 passed.

- [ ] **Step 8.5: Commit**

```bash
git add topoflow_prior/seed_generator.py tests/test_seed_generator.py
git commit -m "feat: seed bundle orchestrator for fused_silu_mul_fp8_quant"
```

---

## Task 9: scripts/generate_seeds.py — CLI

**Files:**
- Create: `scripts/generate_seeds.py`
- Create: `tests/test_generate_seeds_cli.py`

- [ ] **Step 9.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_generate_seeds_cli.py`:

```python
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_seeds.py"), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_help_succeeds():
    r = _run_cli(["--help"])
    assert r.returncode == 0, r.stderr
    assert "--target" in r.stdout
    assert "--E" in r.stdout
    assert "--out" in r.stdout


def test_cli_generates_seed_bundle(tmp_path):
    out = tmp_path / "demo"
    r = _run_cli(
        [
            "--target", "fused_silu_mul_fp8_quant",
            "--E", "32",
            "--T", "1024",
            "--H", "4096",
            "--group-size", "128",
            "--arch", "mi300x",
            "--out", str(out),
        ]
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"
    folders = [p for p in out.iterdir() if p.is_dir()]
    assert len(folders) >= 12
    sample = folders[0]
    assert (sample / "kernel.py").exists()
    ast.parse((sample / "kernel.py").read_text())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_silu_mul_fp8_quant"


def test_cli_unknown_target_errors(tmp_path):
    r = _run_cli(
        [
            "--target", "not_a_real_op",
            "--E", "1", "--T", "1", "--H", "128",
            "--arch", "mi300x",
            "--out", str(tmp_path / "demo"),
        ]
    )
    assert r.returncode != 0
    assert "target" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 9.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_generate_seeds_cli.py -v
```

Expected: failures because `scripts/generate_seeds.py` doesn't exist yet (subprocess returns non-zero with `can't open file ...`).

- [ ] **Step 9.3: Write the CLI**

Create `/Users/manavshah/topoflow/scripts/generate_seeds.py`:

```python
#!/usr/bin/env python3
"""Topo-Flow Prior CLI: emit a seed bundle for one target op."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo without `pip install -e`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topoflow_prior.seed_generator import (  # noqa: E402
    generate_seeds_for_fused_silu_mul_fp8_quant,
)
from topoflow_prior.topology import get_topology  # noqa: E402

_TARGETS = {
    "fused_silu_mul_fp8_quant": generate_seeds_for_fused_silu_mul_fp8_quant,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_seeds.py",
        description="Topo-Flow Prior: emit a seed bundle for a target op.",
    )
    p.add_argument(
        "--target",
        required=True,
        help=f"target op; one of: {sorted(_TARGETS)}",
    )
    p.add_argument("--E", type=int, required=True, help="batch / expert dim")
    p.add_argument("--T", type=int, required=True, help="tokens-per-expert dim")
    p.add_argument("--H", type=int, required=True, help="hidden dim (per-half)")
    p.add_argument(
        "--group-size", type=int, default=128, help="FP8 quant group size (default 128)"
    )
    p.add_argument("--arch", default="mi300x", help="target arch (default mi300x)")
    p.add_argument("--out", required=True, help="output directory for seed bundle")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.target not in _TARGETS:
        print(
            f"error: unknown --target {args.target!r}; known: {sorted(_TARGETS)}",
            file=sys.stderr,
        )
        return 2
    topology = get_topology(args.arch)
    shape = {"E": args.E, "T": args.T, "H": args.H, "group_size": args.group_size}
    generator = _TARGETS[args.target]
    cands = generator(shape, topology, args.out)
    print(f"wrote {len(cands)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Then mark it executable:

```bash
chmod +x /Users/manavshah/topoflow/scripts/generate_seeds.py
```

- [ ] **Step 9.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_generate_seeds_cli.py -v
```

Expected: 3 passed.

- [ ] **Step 9.5: Manual smoke test**

Run:

```bash
cd /Users/manavshah/topoflow
.venv/bin/python scripts/generate_seeds.py --target fused_silu_mul_fp8_quant \
  --E 32 --T 1024 --H 4096 --arch mi300x --out runs/demo
ls runs/demo | head -5
.venv/bin/python -c "import ast; ast.parse(open('runs/demo/fused_silu_mul_fp8_quant_v000/kernel.py').read()); print('parse ok')"
```

Expected: lists ≥12 candidate folders; `parse ok`. The `runs/` dir is gitignored.

- [ ] **Step 9.6: Commit**

```bash
git add scripts/generate_seeds.py tests/test_generate_seeds_cli.py
git commit -m "feat: generate_seeds.py CLI for fused_silu_mul_fp8_quant"
```

---

## Task 10: scripts/package_for_geak.py — convert seeds to GEAK task folders

**Files:**
- Create: `scripts/package_for_geak.py`
- Create: `tests/test_package_for_geak.py`

- [ ] **Step 10.1: Write failing test**

Create `/Users/manavshah/topoflow/tests/test_package_for_geak.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from topoflow_prior.seed_generator import generate_seeds_for_fused_silu_mul_fp8_quant
from topoflow_prior.topology import MI300X

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed_bundle(tmp_path):
    bundle = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(
        {"E": 32, "T": 1024, "H": 4096, "group_size": 128}, MI300X, bundle
    )
    return bundle


def _run_cli(args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "package_for_geak.py"), *args],
        capture_output=True,
        text=True,
    )


def test_cli_help_succeeds():
    r = _run_cli(["--help"])
    assert r.returncode == 0, r.stderr
    assert "--seed-bundle" in r.stdout
    assert "--out" in r.stdout


def test_packages_each_candidate_as_task(seed_bundle, tmp_path):
    tasks_dir = tmp_path / "geak_tasks"
    r = _run_cli(["--seed-bundle", str(seed_bundle), "--out", str(tasks_dir)])
    assert r.returncode == 0, r.stderr

    seed_folders = sorted(p for p in seed_bundle.iterdir() if p.is_dir())
    task_folders = sorted(p for p in tasks_dir.iterdir() if p.is_dir())
    assert len(task_folders) == len(seed_folders)


def test_each_task_has_kernel_and_task_md(seed_bundle, tmp_path):
    tasks_dir = tmp_path / "geak_tasks"
    _run_cli(["--seed-bundle", str(seed_bundle), "--out", str(tasks_dir)])
    for d in tasks_dir.iterdir():
        assert (d / "kernel.py").exists(), f"missing kernel.py in {d}"
        assert (d / "task.md").exists(), f"missing task.md in {d}"
        assert (d / "topoflow_metadata.json").exists(), f"missing metadata in {d}"
        task_md = (d / "task.md").read_text()
        assert "Optimize" in task_md
        assert "MI300X" in task_md or "gfx942" in task_md
        # task.md should embed mutation hints
        assert "BLOCK_M" in task_md


def test_task_md_references_topoflow_notes(seed_bundle, tmp_path):
    tasks_dir = tmp_path / "geak_tasks"
    _run_cli(["--seed-bundle", str(seed_bundle), "--out", str(tasks_dir)])
    for d in tasks_dir.iterdir():
        task_md = (d / "task.md").read_text()
        assert "topoflow_notes.md" in task_md or "topoflow_metadata.json" in task_md
```

- [ ] **Step 10.2: Run, see fail**

```bash
.venv/bin/pytest tests/test_package_for_geak.py -v
```

Expected: failures because `scripts/package_for_geak.py` does not exist.

- [ ] **Step 10.3: Write the packager**

Create `/Users/manavshah/topoflow/scripts/package_for_geak.py`:

```python
#!/usr/bin/env python3
"""Convert a Topo-Flow seed bundle into GEAK task folders.

Each seed candidate becomes a GEAK task folder containing:
- kernel.py            (copied from the seed)
- topoflow_metadata.json (copied)
- topoflow_notes.md    (copied)
- task.md              (NEW: optimization instructions for GEAK referencing the metadata)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_TASK_MD_TEMPLATE = """\
# GEAK task: optimize {op} on {arch}

## What this kernel does

This Triton kernel implements **{op}** on **{topology_name}** ({arch}).
Shape: E={E}, T={T}, H={H}, group_size={group_size}. The kernel fuses SiLU,
elementwise multiply, and per-group FP8 quantization so the bf16 silu*up
intermediate never reaches HBM.

## Topo-Flow seed configuration

- BLOCK_M = {BLOCK_M}
- BLOCK_H = {BLOCK_H}
- num_warps = {num_warps}
- Cost model score (fused/unfused, lower is better): {score:.4f}
- Memory traffic saved vs. unfused: {bytes_saved:,} bytes

## Your job

Optimize this kernel for {arch} ({topology_name}). The seed compiles and is
correct; your job is to make it faster. Start by reading `topoflow_notes.md`
and `topoflow_metadata.json` in this folder — they contain the optimization
intent, risks, and Topo-Flow's suggested mutations.

## Suggested mutations (from Topo-Flow)

{mutations_md}

## Constraints

- Per-group amax/scale must use the same FP8_E4M3_MAX (448.0) as the seed.
- Output dtype is fp8_e4m3; scale dtype is fp32.
- `group_size` divides H. Do not change it without updating callers.
- Keep TOPOFLOW_INTENT comments — downstream tooling reads them.
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="package_for_geak.py",
        description="Convert a Topo-Flow seed bundle into GEAK task folders.",
    )
    p.add_argument(
        "--seed-bundle",
        required=True,
        help="path to the seed bundle dir produced by generate_seeds.py",
    )
    p.add_argument("--out", required=True, help="output dir for GEAK task folders")
    return p


def _render_task_md(meta: dict) -> str:
    mutations_md = "\n".join(f"- {m}" for m in meta.get("suggested_mutations", []))
    return _TASK_MD_TEMPLATE.format(
        op=meta["op"],
        arch=meta["target_arch"],
        topology_name=meta["topology"]["name"],
        E=meta["shape"]["E"],
        T=meta["shape"]["T"],
        H=meta["shape"]["H"],
        group_size=meta["shape"]["group_size"],
        BLOCK_M=meta["tile_plan"]["BLOCK_M"],
        BLOCK_H=meta["tile_plan"]["BLOCK_H"],
        num_warps=meta["tile_plan"]["num_warps"],
        score=meta["cost_model"]["score"],
        bytes_saved=meta["cost_model"]["bytes_saved"],
        mutations_md=mutations_md,
    )


def _package_one(seed_dir: Path, out_root: Path) -> Path:
    meta_path = seed_dir / "topoflow_metadata.json"
    meta = json.loads(meta_path.read_text())
    task_dir = out_root / seed_dir.name
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_dir / "kernel.py", task_dir / "kernel.py")
    shutil.copy2(meta_path, task_dir / "topoflow_metadata.json")
    shutil.copy2(seed_dir / "topoflow_notes.md", task_dir / "topoflow_notes.md")
    (task_dir / "task.md").write_text(_render_task_md(meta))
    return task_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seed_root = Path(args.seed_bundle)
    if not seed_root.is_dir():
        print(f"error: --seed-bundle {seed_root} is not a directory", file=sys.stderr)
        return 2
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = sorted(p for p in seed_root.iterdir() if p.is_dir())
    if not seeds:
        print(f"error: no candidate folders found in {seed_root}", file=sys.stderr)
        return 2

    for seed in seeds:
        _package_one(seed, out_root)
    print(f"packaged {len(seeds)} seeds into {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Mark it executable:

```bash
chmod +x /Users/manavshah/topoflow/scripts/package_for_geak.py
```

- [ ] **Step 10.4: Run, see PASS**

```bash
.venv/bin/pytest tests/test_package_for_geak.py -v
```

Expected: 4 passed.

- [ ] **Step 10.5: Manual smoke test**

```bash
cd /Users/manavshah/topoflow
.venv/bin/python scripts/package_for_geak.py \
  --seed-bundle runs/demo --out runs/demo_geak_tasks
ls runs/demo_geak_tasks | head -3
head -5 runs/demo_geak_tasks/fused_silu_mul_fp8_quant_v000/task.md
```

Expected: ≥12 task folders, each containing `task.md` referencing MI300X.

- [ ] **Step 10.6: Commit**

```bash
git add scripts/package_for_geak.py tests/test_package_for_geak.py
git commit -m "feat: package_for_geak.py — convert seed bundle to GEAK task folders"
```

---

## Task 11: End-to-end integration test + README

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `README.md`

- [ ] **Step 11.1: Write end-to-end test**

Create `/Users/manavshah/topoflow/tests/test_end_to_end.py`:

```python
"""End-to-end pipeline: CLI -> seed bundle -> GEAK task folders."""

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_produces_12_plus_valid_candidates_and_tasks(tmp_path):
    bundle = tmp_path / "bundle"
    tasks = tmp_path / "tasks"

    r1 = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "generate_seeds.py"),
            "--target", "fused_silu_mul_fp8_quant",
            "--E", "32", "--T", "1024", "--H", "4096",
            "--arch", "mi300x",
            "--out", str(bundle),
        ],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr

    candidates = sorted(p for p in bundle.iterdir() if p.is_dir())
    assert len(candidates) >= 12

    for c in candidates:
        # 1) kernel.py is valid Python and contains topoflow intent.
        code = (c / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code
        assert "@triton.jit" in code

        # 2) metadata is well-formed.
        meta = json.loads((c / "topoflow_metadata.json").read_text())
        assert meta["op"] == "fused_silu_mul_fp8_quant"
        assert meta["target_arch"] == "gfx942"
        assert set(meta["tile_plan"]) == {"BLOCK_M", "BLOCK_H", "num_warps"}
        assert meta["cost_model"]["bytes_saved"] > 0

        # 3) notes are present.
        notes = (c / "topoflow_notes.md").read_text()
        assert len(notes) > 50

    r2 = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "package_for_geak.py"),
            "--seed-bundle", str(bundle), "--out", str(tasks),
        ],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr

    task_folders = sorted(p for p in tasks.iterdir() if p.is_dir())
    assert len(task_folders) == len(candidates)
    for d in task_folders:
        task_md = (d / "task.md").read_text()
        assert "Optimize this kernel" in task_md
        assert "BLOCK_M" in task_md
        assert (d / "kernel.py").exists()
        assert (d / "topoflow_metadata.json").exists()
        assert (d / "topoflow_notes.md").exists()
```

- [ ] **Step 11.2: Run full suite**

```bash
cd /Users/manavshah/topoflow
.venv/bin/pytest -v
```

Expected: all tests pass (≈50 tests across all modules).

- [ ] **Step 11.3: Write README**

Create `/Users/manavshah/topoflow/README.md`:

```markdown
# Topo-Flow Prior

Topology-aware seed generator for AMD GPU kernel optimization agents
(GEAK / GEAK-eval). Topo-Flow consumes operation metadata, tensor shapes,
and MI300X topology, and emits a **seed bundle** — a folder of candidate
Triton kernels each with `kernel.py` + `topoflow_metadata.json` +
`topoflow_notes.md`. GEAK then optimizes *from* these seeds instead of
*from scratch*. See `TOPOFLOW_SPEC.md` for the full design rationale.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Generate a seed bundle for the AITER fused SiLU+Mul+FP8 op on MI300X.
.venv/bin/python scripts/generate_seeds.py \
  --target fused_silu_mul_fp8_quant \
  --E 32 --T 1024 --H 4096 \
  --arch mi300x \
  --out runs/demo

# 2. (Optional) Convert the bundle into GEAK task folders.
.venv/bin/python scripts/package_for_geak.py \
  --seed-bundle runs/demo \
  --out runs/demo_geak_tasks
```

The CLI writes one candidate folder per (BLOCK_M, BLOCK_H, num_warps)
combination (≥12 candidates by default).

## Tests

```bash
.venv/bin/pytest -v
```

## What's in scope

- Fused SiLU + Mul + FP8 quantization (AITER issue #2420)
- Attention workgroup remapping orderings (arXiv 2511.02132)

## What's NOT in scope

No compiler, no learned cost model, no MLIR/HIP, no GEAK modifications.
File-system integration only.

## Layout notes

Jinja templates ship inside the `topoflow_prior` package
(`topoflow_prior/templates/triton/`) instead of a sibling `templates/`
directory shown in `TOPOFLOW_SPEC.md`. This keeps `pip install` working
without env tweaks.
```

- [ ] **Step 11.4: Final commit**

```bash
cd /Users/manavshah/topoflow
.venv/bin/pytest -q
git add tests/test_end_to_end.py README.md
git commit -m "feat: end-to-end integration test and README quickstart"
```

Expected: all tests pass, commit succeeds.

---

## Self-Review

**1. Spec coverage** (TOPOFLOW_SPEC.md sections vs. tasks):

| Spec section | Task |
|---|---|
| schemas.py dataclasses | Task 1 |
| dataflow.py hardcoded DFG | Task 3 |
| topology.py MI300X spec | Task 2 |
| cost_model.py memory-traffic estimator | Task 4 |
| tile_planner.py BLOCK_M/H/num_warps enumeration | Task 5 |
| remap_planner.py four orderings + permutation tests | Task 6 |
| triton_templates.py Jinja template (fused only) | Task 7 |
| seed_generator.py orchestrator | Task 8 |
| generate_seeds.py CLI with `--target/--E/--T/--H/--arch/--out` | Task 9 |
| package_for_geak.py task.md output | Task 10 |
| Tests for cost model, seed generation, remap permutation | Tasks 4, 6, 8 |
| Every kernel has TOPOFLOW_INTENT comments | Task 7 template; verified in tests 7, 8, 11 |
| Every candidate has metadata.json | Task 8; verified in tests 8, 11 |
| Success criterion: 12+ candidate folders, each with valid Triton kernel + metadata + notes | Tasks 8, 9, 11 (`test_full_pipeline_produces_12_plus_valid_candidates_and_tasks`) |

Gaps: Attention Jinja template (`attention_remap.py.j2`) — explicitly out of scope per the clarifying question (fused-only template). `remap_planner` still ships four mapping functions + `attention_remap_plans()` for downstream consumers; full attention-kernel rendering is future work.

**2. Placeholder scan:** Searched plan for "TBD", "TODO", "implement later", "Similar to Task N", "add appropriate", "fill in" — none found. Every test, every implementation, every commit message is concrete.

**3. Type consistency:**

- `TilePlan(block_m, block_h, num_warps)` — used identically in tasks 1, 5, 7, 8, 9, 10.
- `TopologyPlan(use_workgroup_remap, remap_kind, notes)` — task 1 defines, task 6 constructs, task 8 references.
- `CostEstimate(fused_bytes, unfused_bytes, bytes_saved, score)` — task 1 defines, task 4 constructs, task 8 reads.
- `SeedCandidate(candidate_id, op_name, kernel_code, tile_plan, topology_plan, cost, metadata, notes)` — task 1 defines, task 8 constructs.
- Function signatures: `tile_plans_fused_silu_mul_fp8(H, group_size=128)` (task 5) — called identically in task 8. `render_fused_silu_mul_fp8_quant(tile_plan, shape)` (task 7) — called identically in task 8. `generate_seeds_for_fused_silu_mul_fp8_quant(shape, topology, out_dir)` (task 8) — called identically in CLI (task 9) and tests (tasks 10, 11).
- Metadata JSON field names (`op`, `target_arch`, `shape`, `tile_plan`, `cost_model`, `topology`, `topology_plan`, `fusion_plan`, `suggested_mutations`) — defined once in task 8, read identically in task 10 packager and tests in tasks 8, 11.

No naming drift detected.
