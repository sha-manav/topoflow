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
