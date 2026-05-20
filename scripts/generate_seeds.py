#!/usr/bin/env python3
"""Topo-Flow Prior CLI: emit a seed bundle for one target op.

Supported targets:
- ``fused_silu_mul_fp8_quant`` — AITER issue #2420, requires --E --T --H [--group-size].
- ``rmsnorm_residual``         — transformer epilogue, requires --M --N.
- ``bias_gelu_dropout``        — transformer FFN, requires --M --N [--dropout-p --seed].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo without `pip install -e`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topoflow_prior.seed_generator import (  # noqa: E402
    generate_seeds_for_bias_gelu_dropout,
    generate_seeds_for_fused_silu_mul_fp8_quant,
    generate_seeds_for_rmsnorm_residual,
)
from topoflow_prior.topology import get_topology  # noqa: E402


def _shape_fused_silu(args) -> dict:
    return {
        "E": args.E,
        "T": args.T,
        "H": args.H,
        "group_size": args.group_size,
    }


def _shape_rmsnorm_residual(args) -> dict:
    return {"M": args.M, "N": args.N}


def _shape_bias_gelu_dropout(args) -> dict:
    return {
        "M": args.M,
        "N": args.N,
        "dropout_p": args.dropout_p,
        "seed": args.seed,
    }


# target -> (generator, required-arg names, shape-builder)
_TARGETS = {
    "fused_silu_mul_fp8_quant": (
        generate_seeds_for_fused_silu_mul_fp8_quant,
        ("E", "T", "H"),
        _shape_fused_silu,
    ),
    "rmsnorm_residual": (
        generate_seeds_for_rmsnorm_residual,
        ("M", "N"),
        _shape_rmsnorm_residual,
    ),
    "bias_gelu_dropout": (
        generate_seeds_for_bias_gelu_dropout,
        ("M", "N"),
        _shape_bias_gelu_dropout,
    ),
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
    # fused_silu_mul_fp8_quant dims
    p.add_argument("--E", type=int, default=None, help="expert dim (fused_silu_mul_fp8_quant)")
    p.add_argument("--T", type=int, default=None, help="tokens-per-expert (fused_silu_mul_fp8_quant)")
    p.add_argument("--H", type=int, default=None, help="hidden-per-half (fused_silu_mul_fp8_quant)")
    p.add_argument(
        "--group-size", type=int, default=128, help="FP8 quant group size (default 128)"
    )
    # rmsnorm_residual / bias_gelu_dropout dims
    p.add_argument("--M", type=int, default=None, help="rows (rmsnorm_residual, bias_gelu_dropout)")
    p.add_argument("--N", type=int, default=None, help="cols (rmsnorm_residual, bias_gelu_dropout)")
    # bias_gelu_dropout extras
    p.add_argument(
        "--dropout-p", type=float, default=0.1,
        help="dropout probability (bias_gelu_dropout; default 0.1)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="dropout RNG seed (bias_gelu_dropout; default 0)",
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

    generator, required, shape_builder = _TARGETS[args.target]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        print(
            f"error: --target {args.target} requires args: "
            f"{', '.join('--' + n for n in missing)}",
            file=sys.stderr,
        )
        return 2

    topology = get_topology(args.arch)
    shape = shape_builder(args)
    cands = generator(shape, topology, args.out)
    print(f"wrote {len(cands)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
