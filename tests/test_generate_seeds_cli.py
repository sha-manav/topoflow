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


def test_cli_rmsnorm_residual_generates_seed_bundle(tmp_path):
    out = tmp_path / "rms"
    r = _run_cli(
        [
            "--target", "rmsnorm_residual",
            "--M", "2048",
            "--N", "4096",
            "--arch", "mi300x",
            "--out", str(out),
        ]
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"
    folders = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(folders) >= 8
    sample = folders[0]
    ast.parse((sample / "kernel.py").read_text())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_rmsnorm_residual"
    assert meta["shape"] == {"M": 2048, "N": 4096}


def test_cli_bias_gelu_dropout_generates_seed_bundle(tmp_path):
    out = tmp_path / "bgd"
    r = _run_cli(
        [
            "--target", "bias_gelu_dropout",
            "--M", "2048",
            "--N", "16384",
            "--dropout-p", "0.1",
            "--seed", "0",
            "--arch", "mi300x",
            "--out", str(out),
        ]
    )
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"
    folders = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(folders) >= 8
    sample = folders[0]
    ast.parse((sample / "kernel.py").read_text())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_bias_gelu_dropout"
    assert meta["shape"]["dropout_p"] == 0.1
    assert meta["shape"]["seed"] == 0


def test_cli_rmsnorm_missing_required_args_errors(tmp_path):
    r = _run_cli(
        [
            "--target", "rmsnorm_residual",
            "--M", "2048",  # missing --N
            "--arch", "mi300x",
            "--out", str(tmp_path / "out"),
        ]
    )
    assert r.returncode != 0
    assert "N" in r.stderr or "n" in r.stderr.lower()


def test_cli_help_lists_all_three_targets():
    r = _run_cli(["--help"])
    assert r.returncode == 0, r.stderr
    for name in ("fused_silu_mul_fp8_quant", "rmsnorm_residual", "bias_gelu_dropout"):
        assert name in r.stdout, f"target {name} not listed in --help"
