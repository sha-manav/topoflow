#!/usr/bin/env python3
"""Compare GEAK-alone vs Topo-Flow-seeded eval runs.

Reads GEAK-eval JSON results from two directories, prints a per-kernel
comparison table (baseline_speedup vs topoflow_speedup + delta), and
optionally computes Spearman rank correlation between Topo-Flow cost-model
scores and measured runtimes when a seed bundle is supplied.

This is a stub that defines the interface; the JSON schema below mirrors a
plausible GEAK-eval output and will be tightened once we have real results.

Expected JSON shape (per file under --baseline-dir / --topoflow-dir):

    {
      "kernel_name": "fused_silu_mul_fp8_quant_v005",
      "speedup":     1.42,        # vs. some implicit reference; higher is better
      "runtime_ms":  0.123,
      "correctness": true
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_geak_eval_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load ``kernel_name -> metrics`` from GEAK-eval JSON files in ``results_dir``.

    Each ``*.json`` file is expected to be a JSON object with at least a
    ``kernel_name`` key. Other keys (``speedup``, ``runtime_ms``,
    ``correctness``) are passed through verbatim. Files that fail to parse or
    that lack ``kernel_name`` are skipped with a warning to stderr.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"warning: {path}: not valid JSON ({e})", file=sys.stderr)
            continue
        name = data.get("kernel_name")
        if not name:
            print(f"warning: {path}: missing 'kernel_name'", file=sys.stderr)
            continue
        out[name] = data
    return out


def load_topoflow_scores(seeds_dir: Path) -> dict[str, float]:
    """Read ``candidate_id -> cost_model.score`` from a Topo-Flow seed bundle."""
    out: dict[str, float] = {}
    for d in sorted(seeds_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "topoflow_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cid = meta.get("candidate_id") or d.name
        score = meta.get("cost_model", {}).get("score")
        if score is not None:
            out[cid] = float(score)
    return out


def _rank(values: list[float]) -> list[float]:
    """Average rank (1-indexed); ties get the mean of the ranks they would occupy."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation in [-1, 1]; handles ties via average ranks.

    Raises ValueError on length mismatch or n < 2. Returns 0.0 when either
    variable has zero rank variance (e.g. all values tied).
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: len(x)={len(x)} len(y)={len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError(f"need at least two pairs; got {n}")
    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rxi - mx) * (ryi - my) for rxi, ryi in zip(rx, ry))
    den_x = sum((rxi - mx) ** 2 for rxi in rx)
    den_y = sum((ryi - my) ** 2 for ryi in ry)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x ** 0.5 * den_y ** 0.5)


def print_comparison_table(
    baseline: dict[str, dict[str, Any]],
    topoflow: dict[str, dict[str, Any]],
) -> None:
    """Print one row per kernel: name, baseline speedup, topoflow speedup, delta."""
    all_names = sorted(set(baseline) | set(topoflow))
    header = f"{'kernel':<48} {'baseline':>10} {'topoflow':>10} {'delta':>10}"
    print(header)
    print("-" * len(header))
    for name in all_names:
        b = baseline.get(name, {}).get("speedup")
        t = topoflow.get(name, {}).get("speedup")
        b_cell = f"{b:>10.3f}" if isinstance(b, (int, float)) else f"{'-':>10}"
        t_cell = f"{t:>10.3f}" if isinstance(t, (int, float)) else f"{'-':>10}"
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            delta_cell = f"{t - b:>+10.3f}"
        else:
            delta_cell = f"{'-':>10}"
        print(f"{name:<48} {b_cell} {t_cell} {delta_cell}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Compare GEAK-alone vs Topo-Flow-seeded eval results.",
    )
    p.add_argument(
        "--baseline-dir",
        required=True,
        type=Path,
        help="directory of GEAK-eval *.json results for the baseline run",
    )
    p.add_argument(
        "--topoflow-dir",
        required=True,
        type=Path,
        help="directory of GEAK-eval *.json results for the Topo-Flow-seeded run",
    )
    p.add_argument(
        "--seeds-dir",
        type=Path,
        default=None,
        help=(
            "optional Topo-Flow seed bundle; enables Spearman correlation "
            "between cost-model score and measured runtime_ms"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.baseline_dir.is_dir():
        print(f"error: --baseline-dir {args.baseline_dir} is not a directory", file=sys.stderr)
        return 2
    if not args.topoflow_dir.is_dir():
        print(f"error: --topoflow-dir {args.topoflow_dir} is not a directory", file=sys.stderr)
        return 2
    baseline = load_geak_eval_results(args.baseline_dir)
    topoflow = load_geak_eval_results(args.topoflow_dir)
    print_comparison_table(baseline, topoflow)

    if args.seeds_dir is not None:
        if not args.seeds_dir.is_dir():
            print(f"error: --seeds-dir {args.seeds_dir} is not a directory", file=sys.stderr)
            return 2
        scores = load_topoflow_scores(args.seeds_dir)
        pairs = [
            (scores[name], topoflow[name]["runtime_ms"])
            for name in topoflow
            if name in scores
            and isinstance(topoflow[name].get("runtime_ms"), (int, float))
        ]
        print()
        if len(pairs) >= 2:
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            print(f"Spearman rho (cost_model.score vs runtime_ms): {rho:+.3f}")
            print(f"  ({len(pairs)} kernels matched between seeds-dir and topoflow-dir)")
        else:
            print(f"Spearman: need >=2 matched kernels (got {len(pairs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
