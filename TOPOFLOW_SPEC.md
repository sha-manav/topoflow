# Topo-Flow: Topology-Aware Seed Prior for AMD Kernel Search

## What this is

Topo-Flow is a seed generator that makes GEAK/OpenEvolve smarter at the start. It analyzes an operation's dataflow, tensor shapes, and MI300X topology, then emits strong initial Triton kernel variants with tile ranges, workgroup remappings, fusion plans, and cost estimates. GEAK then optimizes from these seeds instead of from scratch.

This is NOT a compiler, NOT a replacement for GEAK, NOT a learned model. It is a search prior — a structured way to give AMD kernel agents better starting points so they waste less search budget on physically implausible candidates.

## Why this works

GEAK-OpenEvolve reports 3.42x avg speedup on TritonBench-modified and 7.02x on ROCm-bench. SwizzlePerf gets 2.06x via L2-aware block swizzling. PRAGMA gets 4.5x via profiler-driven multi-agent feedback. None of them inject topology-derived structure into the initial candidates. They all start from generic code and search their way to good code. Topo-Flow starts the search closer to the answer.

The claim to prove: **GEAK + Topo-Flow seeds reaches equal or better performance with fewer evaluations/LLM calls than GEAK alone.** Even if the final kernel ties, reducing search cost is valuable.

## Two targets, not twenty

### Target 1: Fused SiLU + Mul + FP8 Quantization

From a real open AITER issue (github.com/ROCm/aiter/issues/2420): the desired CUDA path fuses SiLU activation, elementwise multiply, and per-group FP8 quantization into one kernel launch, avoiding three launches and two extra global-memory round trips. ROCm currently falls back to a slower Triton kernel.

Input: `x: [E, T, 2H], bf16` → split into gate/up → `silu(gate) * up` → per-group-128 FP8 quantize → output `y_fp8: [E, T, H], fp8` + `scale: [E, T, H/128], fp32`

Why this target: memory-movement sensitive, fusion-sensitive, directly relevant to vLLM/AITER production, more likely to show speedup than plain RMSNorm, less complex than full attention.

### Target 2: Attention Workgroup Remapping

From the NUMA attention paper (arXiv 2511.02132): Swizzled Head-first Mapping assigns all blocks of an attention head to the same XCD before moving to the next head, so workgroups that reuse K/V data hit local L2 cache. Reports up to 50% improvement and 80-97% L2 hit rates on MI300X.

This demonstrates the topology-specific part of Topo-Flow. Target 1 demonstrates memory-traffic/fusion awareness. Target 2 demonstrates chiplet/XCD-aware scheduling.

## Repos to build from

**GEAK** (github.com/AMD-AGI/GEAK): the agent/search layer. Fork, don't modify internals. File-system integration only.

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK && make install-dev
```

**GEAK-eval** (github.com/AMD-AIG-AIMA/GEAK-eval): correctness/performance evaluation. Do not build your own eval harness.

```bash
git clone https://github.com/AMD-AIG-AIMA/GEAK-eval
cd GEAK-eval && pip install -e .
```

**AITER** (github.com/ROCm/aiter): source of real targets and production baselines. Read-only reference.

```bash
git clone https://github.com/ROCm/aiter
```

## Architecture

Topo-Flow is a small Python package. It consumes operation metadata + shapes + topology, and emits a **Seed Bundle**: a folder of candidate kernels with metadata that GEAK can consume via filesystem.

```
topoflow-prior/
  pyproject.toml
  topoflow_prior/
    schemas.py          # dataclasses for everything
    dataflow.py         # hardcoded dataflow graphs per target op
    topology.py         # MI300X/MI355X topology specs
    cost_model.py       # simple memory-traffic estimator
    tile_planner.py     # emit tile ranges, not one tile
    remap_planner.py    # workgroup mapping variants (the topology novelty)
    fusion_planner.py   # decide what to fuse
    seed_generator.py   # orchestrates everything, produces seed bundles
    triton_templates.py # Jinja2-based kernel code generation
  templates/triton/
    fused_silu_mul_fp8_quant.py.j2
    attention_remap.py.j2
  scripts/
    generate_seeds.py       # CLI: produce seed bundle
    package_for_geak.py     # convert seeds to GEAK task folders
    compare_runs.py         # GEAK-alone vs GEAK+seeds
  tests/
    test_cost_model.py
    test_seed_generation.py
    test_remap.py
  experiments/
    fused_silu_mul_fp8_quant/
      shapes.yaml
    attention_swizzle/
      shapes.yaml
```

That's it. No learned models, no MLIR, no HIP assembly, no automatic graph extraction.

## Seed Bundle format

Each candidate is a folder:

```
candidate_000/
  kernel.py                 # Triton source with TOPOFLOW_INTENT comments
  topoflow_metadata.json    # machine-readable: shapes, tiles, cost score, fusion plan
  topoflow_notes.md         # human/LLM-readable: optimization intent, risks, suggested mutations
```

The metadata JSON contains:

```json
{
  "candidate_id": "fused_silu_mul_fp8_quant_v000",
  "op": "fused_silu_mul_fp8_quant",
  "target_arch": "gfx942",
  "shape": {"E": 32, "T": 1024, "H": 4096, "group_size": 128},
  "dtype": {"input": "bf16", "output": "fp8_e4m3", "scale": "fp32"},
  "tile_plan": {"BLOCK_M": 16, "BLOCK_H": 128, "num_warps": 4},
  "topology_plan": {"uses_workgroup_remap": false, "keep_group_quant_local": true},
  "cost_model": {
    "fused_bytes": 134217728,
    "unfused_bytes": 268435456,
    "bytes_saved": 134217728,
    "score": 0.42
  },
  "fusion_plan": ["silu", "mul", "group_amax", "scale", "fp8_quantize"],
  "suggested_mutations": ["BLOCK_M in [8,16,32]", "BLOCK_H in [128,256]", "num_warps in [4,8]"]
}
```

The notes.md tells GEAK/Claude what the kernel is trying to do and where to search:

```markdown
This candidate fuses SiLU, multiply, and per-group FP8 quantization.
Intent: avoid writing bf16 intermediate to HBM. Keep per-group quant local to BLOCK_H.
Risks: register pressure may reduce occupancy. Scale precision must match reference.
Try: BLOCK_M in [8,16,32], BLOCK_H in [128,256], num_warps in [4,8].
```

## Core modules

### schemas.py

```python
from dataclasses import dataclass, field
from typing import Literal, Any

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

### dataflow.py — hardcoded, one function per target

```python
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
            DataflowNode("amax", "reduce_max_abs", ["activated"], ["amax"], attrs={"group_size": 128}),
            DataflowNode("scale_compute", "div_by_fp8_max", ["amax"], ["scale"]),
            DataflowNode("quant", "fp8_quantize", ["activated", "scale"], ["y_fp8"]),
        ],
        fusion_opportunities=[["silu", "mul", "amax", "scale_compute", "quant"]],
    )
```

### topology.py

```python
MI300X = TopologySpec(name="MI300X", arch="gfx942", num_xcds=8, l2_per_xcd_mb=4.0, cu_per_xcd=38)
MI355X = TopologySpec(name="MI355X", arch="gfx950", num_xcds=8)  # placeholder
```

### cost_model.py

Simple memory traffic estimator. Does not need to be accurate — needs to rank variants better than random.

```python
def estimate_fused_silu_mul_fp8_quant(E, T, H, group_size=128):
    input_read = E * T * 2 * H * 2          # bf16
    output_write = E * T * H * 1             # fp8
    scale_write = E * T * (H // group_size) * 4  # fp32
    fused = input_read + output_write + scale_write
    unfused_intermediate = E * T * H * 2 * 2  # write + read bf16 intermediate
    unfused = fused + unfused_intermediate
    return CostEstimate(fused, unfused, unfused - fused, score=fused / unfused)
```

### tile_planner.py

Emit ranges. For fused SiLU+Mul+FP8:

```python
def tile_plans_fused_silu_mul_fp8(H, group_size=128):
    plans = []
    for bm in [8, 16, 32]:
        for bh in [64, 128, 256]:
            if bh % group_size == 0 or group_size % bh == 0:
                for nw in [4, 8]:
                    plans.append(TilePlan(block_m=bm, block_h=bh, num_warps=nw))
    return plans
```

### remap_planner.py — the topology novelty

Four workgroup orderings for attention:

```python
def swizzled_head_first(pid_m, pid_h, pid_b, num_m, num_h, num_b, num_xcds=8):
    heads_per_group = (num_h + num_xcds - 1) // num_xcds
    xcd_group = pid_h // heads_per_group
    local_head = pid_h % heads_per_group
    return pid_b * num_h * num_m + xcd_group * heads_per_group * num_m + local_head * num_m + pid_m
```

Generate all four variants (naive_block_first, naive_head_first, swizzled_block_first, swizzled_head_first) as seed candidates. Let GEAK pick the winner.

### seed_generator.py — orchestrator

Takes target + shapes + topology → produces seed bundle folder with N candidates spanning tile plans × remap variants × precision variants.

### triton_templates.py — Jinja2 kernel generation

Each generated kernel includes `# TOPOFLOW_INTENT` comments that tell GEAK/Claude what the kernel is trying to do and where to mutate.

## GEAK integration — filesystem only

### Phase 1 (v0): Loose coupling

```bash
# Generate seeds
python scripts/generate_seeds.py --target fused_silu_mul_fp8_quant \
  --E 32 --T 1024 --H 4096 --arch mi300x --out runs/seeds/

# Evaluate seeds directly
geak-eval -f runs/seeds/ -o topoflow_seed_eval -ds rocm

# Feed best seed to GEAK for further optimization
geak --kernel-url runs/seeds/candidate_003/kernel.py \
  --repo . \
  --task "Optimize this fused SiLU+Mul+FP8 Triton kernel for MI300X. \
          Read the topoflow_notes.md for optimization intent and constraints."
```

### Phase 2: Script that runs the full comparison

```bash
python scripts/compare_runs.py \
  --baseline-geak-log logs/geak_alone/ \
  --topoflow-geak-log logs/geak_with_seeds/ \
  --seeds-only-log logs/seeds_eval/
```

## Experiment design

Three runs, same LLM, same iteration budget, same hardware:

**A. GEAK alone**: from naive/baseline kernel. Measure best speedup, candidates evaluated, LLM calls, time to 95% of best.

**B. Topo-Flow seeds only**: no GEAK optimization. Measure best seed speedup, correctness rate, cost-model rank correlation with measured runtime.

**C. GEAK + Topo-Flow seeds**: GEAK initialized from best Topo-Flow seeds. Same metrics as A.

Main result: C reaches A's performance with fewer candidates/LLM calls. Secondary: B's best seed already beats A's starting point. Tertiary: cost model ranking correlates with measured performance (Spearman ρ > 0.5).

## Two-week build plan

**Day 1**: Repo skeleton. pyproject.toml, package structure, pytest setup, generate_seeds.py stub.

**Day 2**: schemas.py + dataflow.py + topology.py + cost_model.py. Tests pass.

**Day 3**: tile_planner.py + remap_planner.py. Tests for remap being a valid permutation.

**Day 4**: seed_generator.py. `generate_seeds.py --target fused_silu_mul_fp8_quant` produces candidate folders with metadata.json and notes.md (no kernel code yet).

**Day 5-6**: First Triton template for fused_silu_mul_fp8_quant. Correctness over speed. Generate 12+ candidates with actual kernel.py files.

**Day 7**: Correctness harness. PyTorch reference, torch.allclose with fp8 tolerances. Every candidate tested.

**Day 8**: GEAK-eval integration. Run seeds through geak-eval. If hardware unavailable, document dry-run path.

**Day 9-10**: package_for_geak.py. Each candidate becomes a GEAK task folder with task.md containing optimization instructions referencing Topo-Flow metadata.

**Day 11-12**: Attention remap prototype. Four mapping variants, tests proving permutation correctness and head grouping. Inject into a small Triton attention kernel.

**Day 13-14**: Run comparison experiment (A vs B vs C). Produce results table and write blog post.

## What NOT to build

No learned cost model. No GNN. No MLIR. No HIP assembly. No automatic PyTorch graph extraction. No multi-GPU. No full KernelBench integration. No modifications to GEAK internals. No claims about direct HBM-stack placement.

Every one of those is a tempting extension. Resist all of them until the two-target demo works end-to-end.

## Success criteria

The blog post should show:

1. Topo-Flow generates 12+ correct seed candidates for fused SiLU+Mul+FP8 on MI300X.
2. The best seed already outperforms naive/unfused baseline.
3. GEAK starting from Topo-Flow seeds reaches the same quality as GEAK-alone in fewer iterations.
4. For attention, swizzled_head_first mapping measurably improves L2 hit rate over naive ordering.
5. Cost-model ranking correlates with measured runtime.

If (3) fails but (1), (2), (4), (5) hold, the blog post still works — "Topo-Flow generates strong seeds and the topology-aware mappings show real L2 improvement" is a compelling demo. The GEAK sample-efficiency claim becomes future work.

## Claude Code prompt

Paste this at the root of topoflow-prior:

```
You are building Topo-Flow Prior, a topology-aware seed generator for AMD GPU kernel optimization agents.

Do NOT build a compiler. Build a small Python package that generates initial Triton kernel candidates and metadata for GEAK/GEAK-eval to consume via filesystem.

Two targets only:
1. Fused SiLU + Mul + FP8 quantization for batched MoE (AITER issue #2420)
2. Attention workgroup remapping using swizzled head-first mapping (arXiv 2511.02132)

The package consumes: operation metadata, tensor shapes, MI300X topology.
It emits: seed Triton kernels, tile ranges, workgroup remappings, cost estimates, structured notes.

Repo layout: see TOPOFLOW_SPEC.md in this directory.

Build order:
1. schemas.py — dataclasses for TensorSpec, DataflowGraph, TopologySpec, TilePlan, SeedCandidate, CostEstimate
2. dataflow.py — hardcoded dataflow graph for fused_silu_mul_fp8_quant
3. topology.py — MI300X spec (8 XCDs, 4MB L2/XCD, 38 CU/XCD)
4. cost_model.py — memory traffic estimator (fused vs unfused bytes)
5. tile_planner.py — enumerate BLOCK_M/BLOCK_H/num_warps candidates
6. remap_planner.py — four workgroup orderings for attention (naive/swizzled × block/head first)
7. seed_generator.py — orchestrate planners, produce candidate folders with kernel.py + metadata.json + notes.md
8. triton_templates.py — Jinja2 template for fused_silu_mul_fp8_quant kernel
9. generate_seeds.py CLI — `python scripts/generate_seeds.py --target fused_silu_mul_fp8_quant --E 32 --T 1024 --H 4096 --arch mi300x --out runs/demo`
10. package_for_geak.py — convert seed bundle to GEAK task folders with task.md

Constraints:
- Correctness over speed initially
- No GEAK modifications — file-system integration only
- Every candidate includes metadata.json explaining why it exists
- Every generated kernel includes TOPOFLOW_INTENT comments for GEAK/LLM to read
- Tests for cost model, seed generation, and remap permutation correctness

Success: generate_seeds.py produces 12+ candidate folders, each with a valid Triton kernel, metadata, and notes.
```
