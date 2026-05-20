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
