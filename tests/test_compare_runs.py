"""Tests for scripts/compare_runs.py — CLI and helper functions."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Import the script's helpers directly so we can unit-test pure functions.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import compare_runs as cr  # noqa: E402


def _run_cli(args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compare_runs.py"), *args],
        capture_output=True,
        text=True,
    )


def _write_result(d: Path, name: str, **fields):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({"kernel_name": name, **fields}))


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_spearman_perfect_monotone():
    assert cr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cr.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    # All-tied x => zero variance => 0.0 by convention.
    assert cr.spearman([1, 1, 1, 1], [10, 20, 30, 40]) == 0.0


def test_spearman_known_value_with_ties():
    # Two pairs tied on x, two pairs unique; check average-rank correctness.
    # x: [1, 2, 2, 3] -> ranks [1, 2.5, 2.5, 4]
    # y: [10, 20, 30, 40] -> ranks [1, 2, 3, 4]
    rho = cr.spearman([1, 2, 2, 3], [10, 20, 30, 40])
    # Pearson of (1,2.5,2.5,4) and (1,2,3,4):
    # mean_x=2.5, mean_y=2.5
    # num = -1.5*-1.5 + 0*-0.5 + 0*0.5 + 1.5*1.5 = 2.25 + 2.25 = 4.5
    # den_x = 2.25 + 0 + 0 + 2.25 = 4.5; den_y = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    expected = 4.5 / (4.5 ** 0.5 * 5.0 ** 0.5)
    assert rho == pytest.approx(expected)


def test_spearman_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        cr.spearman([1, 2, 3], [4, 5])


def test_spearman_too_few_pairs_raises():
    with pytest.raises(ValueError, match="at least two"):
        cr.spearman([1.0], [2.0])


def test_load_geak_eval_results_round_trip(tmp_path):
    _write_result(tmp_path, "k1", speedup=1.5, runtime_ms=0.5, correctness=True)
    _write_result(tmp_path, "k2", speedup=2.0, runtime_ms=0.25, correctness=True)
    results = cr.load_geak_eval_results(tmp_path)
    assert set(results) == {"k1", "k2"}
    assert results["k1"]["speedup"] == 1.5
    assert results["k2"]["runtime_ms"] == 0.25


def test_load_geak_eval_results_skips_malformed(tmp_path, capsys):
    _write_result(tmp_path, "good", speedup=1.0)
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "no_name.json").write_text(json.dumps({"speedup": 1.0}))
    results = cr.load_geak_eval_results(tmp_path)
    assert set(results) == {"good"}
    err = capsys.readouterr().err
    assert "broken.json" in err
    assert "no_name.json" in err


def test_load_topoflow_scores(tmp_path):
    for name, score in [("k0", 0.5), ("k1", 0.6)]:
        d = tmp_path / name
        d.mkdir()
        (d / "topoflow_metadata.json").write_text(
            json.dumps({"candidate_id": name, "cost_model": {"score": score}})
        )
    # Folder with no metadata is silently skipped.
    (tmp_path / "noise").mkdir()
    scores = cr.load_topoflow_scores(tmp_path)
    assert scores == {"k0": 0.5, "k1": 0.6}


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_help_succeeds():
    r = _run_cli(["--help"])
    assert r.returncode == 0, r.stderr
    assert "--baseline-dir" in r.stdout
    assert "--topoflow-dir" in r.stdout
    assert "--seeds-dir" in r.stdout


def test_cli_prints_comparison_table(tmp_path):
    baseline = tmp_path / "baseline"
    topoflow = tmp_path / "topoflow"
    _write_result(baseline, "k1", speedup=1.0, runtime_ms=2.0)
    _write_result(baseline, "k2", speedup=1.0, runtime_ms=4.0)
    _write_result(topoflow, "k1", speedup=1.5, runtime_ms=1.33)
    _write_result(topoflow, "k2", speedup=2.0, runtime_ms=2.0)
    r = _run_cli(["--baseline-dir", str(baseline), "--topoflow-dir", str(topoflow)])
    assert r.returncode == 0, r.stderr
    assert "k1" in r.stdout and "k2" in r.stdout
    assert "+0.500" in r.stdout  # delta for k1
    assert "+1.000" in r.stdout  # delta for k2


def test_cli_spearman_with_seeds_dir(tmp_path):
    baseline = tmp_path / "baseline"
    topoflow = tmp_path / "topoflow"
    seeds = tmp_path / "seeds"
    # Monotonically increasing score and runtime => perfect positive corr.
    for i, (score, rt) in enumerate(zip([0.5, 0.6, 0.7], [1.0, 1.5, 2.0])):
        name = f"k{i}"
        _write_result(baseline, name, speedup=1.0, runtime_ms=2.0)
        _write_result(topoflow, name, speedup=1.5, runtime_ms=rt)
        d = seeds / name
        d.mkdir(parents=True)
        (d / "topoflow_metadata.json").write_text(
            json.dumps({"candidate_id": name, "cost_model": {"score": score}})
        )
    r = _run_cli(
        [
            "--baseline-dir", str(baseline),
            "--topoflow-dir", str(topoflow),
            "--seeds-dir", str(seeds),
        ]
    )
    assert r.returncode == 0, r.stderr
    assert "Spearman" in r.stdout
    assert "+1.000" in r.stdout  # rho = +1.0


def test_cli_baseline_dir_not_a_directory(tmp_path):
    r = _run_cli(
        ["--baseline-dir", str(tmp_path / "nope"), "--topoflow-dir", str(tmp_path)]
    )
    assert r.returncode != 0
    assert "not a directory" in r.stderr
