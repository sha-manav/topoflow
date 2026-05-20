import json
import subprocess
import sys
from pathlib import Path

import pytest

from topoflow_prior.seed_generator import (
    generate_seeds_for_bias_gelu_dropout,
    generate_seeds_for_fused_silu_mul_fp8_quant,
    generate_seeds_for_rmsnorm_residual,
)
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
        assert "BLOCK_M" in task_md


def test_task_md_references_topoflow_notes(seed_bundle, tmp_path):
    tasks_dir = tmp_path / "geak_tasks"
    _run_cli(["--seed-bundle", str(seed_bundle), "--out", str(tasks_dir)])
    for d in tasks_dir.iterdir():
        task_md = (d / "task.md").read_text()
        assert "topoflow_notes.md" in task_md or "topoflow_metadata.json" in task_md


@pytest.fixture
def rmsnorm_bundle(tmp_path):
    bundle = tmp_path / "rms_seeds"
    generate_seeds_for_rmsnorm_residual({"M": 2048, "N": 4096}, MI300X, bundle)
    return bundle


@pytest.fixture
def bias_gelu_bundle(tmp_path):
    bundle = tmp_path / "bgd_seeds"
    generate_seeds_for_bias_gelu_dropout(
        {"M": 2048, "N": 16384, "dropout_p": 0.1, "seed": 0}, MI300X, bundle
    )
    return bundle


def test_packages_rmsnorm_residual_bundle(rmsnorm_bundle, tmp_path):
    tasks_dir = tmp_path / "rms_tasks"
    r = _run_cli(["--seed-bundle", str(rmsnorm_bundle), "--out", str(tasks_dir)])
    assert r.returncode == 0, r.stderr
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        task_md = (d / "task.md").read_text()
        assert "fused_rmsnorm_residual" in task_md
        assert "MI300X" in task_md or "gfx942" in task_md
        # tile_plan section uses BLOCK_N (not BLOCK_H) for this target
        assert "BLOCK_N" in task_md
        # generic shape rendering
        assert "M=2048" in task_md and "N=4096" in task_md


def test_packages_bias_gelu_dropout_bundle(bias_gelu_bundle, tmp_path):
    tasks_dir = tmp_path / "bgd_tasks"
    r = _run_cli(["--seed-bundle", str(bias_gelu_bundle), "--out", str(tasks_dir)])
    assert r.returncode == 0, r.stderr
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        task_md = (d / "task.md").read_text()
        assert "fused_bias_gelu_dropout" in task_md
        assert "BLOCK_N" in task_md
        # Generic shape rendering includes the extra kernel knobs.
        assert "dropout_p=0.1" in task_md
