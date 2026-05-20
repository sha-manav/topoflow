"""End-to-end pipeline: CLI -> seed bundle -> GEAK task folders."""

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_produces_12_plus_valid_candidates_and_tasks(tmp_path):
    bundle = tmp_path / "bundle"
    tasks = tmp_path / "tasks"

    r1 = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "generate_seeds.py"),
            "--target", "fused_silu_mul_fp8_quant",
            "--E", "32", "--T", "1024", "--H", "4096",
            "--arch", "mi300x",
            "--out", str(bundle),
        ],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr

    candidates = sorted(p for p in bundle.iterdir() if p.is_dir())
    assert len(candidates) >= 12

    for c in candidates:
        # 1) kernel.py is valid Python and contains topoflow intent.
        code = (c / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code
        assert "@triton.jit" in code

        # 2) metadata is well-formed.
        meta = json.loads((c / "topoflow_metadata.json").read_text())
        assert meta["op"] == "fused_silu_mul_fp8_quant"
        assert meta["target_arch"] == "gfx942"
        assert set(meta["tile_plan"]) == {"BLOCK_M", "BLOCK_H", "num_warps"}
        assert meta["cost_model"]["bytes_saved"] > 0

        # 3) notes are present.
        notes = (c / "topoflow_notes.md").read_text()
        assert len(notes) > 50

    r2 = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "package_for_geak.py"),
            "--seed-bundle", str(bundle), "--out", str(tasks),
        ],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr

    task_folders = sorted(p for p in tasks.iterdir() if p.is_dir())
    assert len(task_folders) == len(candidates)
    for d in task_folders:
        task_md = (d / "task.md").read_text()
        assert "Optimize this kernel" in task_md
        assert "BLOCK_M" in task_md
        assert (d / "kernel.py").exists()
        assert (d / "topoflow_metadata.json").exists()
        assert (d / "topoflow_notes.md").exists()
