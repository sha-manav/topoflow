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
combination (>=12 candidates by default; 18 for the standard shape).

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

The four attention workgroup orderings exposed by `remap_planner.py` use
correct topology-aware swizzles (program_id % num_xcds maps to XCD). The
spec's original formula was identity-equivalent to the naive ordering;
this implementation fixes that while preserving the four-variant API.
Requires num_h or num_m to be divisible by num_xcds for the swizzles.
