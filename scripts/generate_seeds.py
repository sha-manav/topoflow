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
