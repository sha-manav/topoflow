import ast
import json
from pathlib import Path

import pytest

from topoflow_prior.schemas import SeedCandidate
from topoflow_prior.seed_generator import (
    generate_seeds_for_bias_gelu_dropout,
    generate_seeds_for_fused_silu_mul_fp8_quant,
    generate_seeds_for_rmsnorm_residual,
)
from topoflow_prior.topology import MI300X


@pytest.fixture
def shape():
    return {"E": 32, "T": 1024, "H": 4096, "group_size": 128}


def test_generates_at_least_12_candidates(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    candidates = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    assert len(candidates) >= 12
    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert len(folders) == len(candidates)


def test_each_candidate_has_three_files(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in sorted(out_dir.iterdir()):
        assert (d / "kernel.py").exists(), f"missing kernel.py in {d}"
        assert (d / "topoflow_metadata.json").exists(), f"missing metadata in {d}"
        assert (d / "topoflow_notes.md").exists(), f"missing notes in {d}"


def test_metadata_has_required_fields(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    sample = next(out_dir.iterdir())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_silu_mul_fp8_quant"
    assert meta["target_arch"] == "gfx942"
    assert meta["shape"] == shape
    for k in ("BLOCK_M", "BLOCK_H", "num_warps"):
        assert k in meta["tile_plan"]
    assert "cost_model" in meta and "score" in meta["cost_model"]
    assert "fusion_plan" in meta
    assert "suggested_mutations" in meta
    assert meta["candidate_id"].startswith("fused_silu_mul_fp8_quant_v")


def test_every_kernel_py_parses(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in out_dir.iterdir():
        code = (d / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code


def test_notes_md_is_non_empty(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    for d in out_dir.iterdir():
        notes = (d / "topoflow_notes.md").read_text()
        assert len(notes) > 50
        assert "fuses SiLU" in notes or "fused" in notes.lower()


def test_candidate_ids_unique(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    ids = [c.candidate_id for c in cands]
    assert len(set(ids)) == len(ids)


def test_returns_seed_candidate_objects(tmp_path, shape):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    assert all(isinstance(c, SeedCandidate) for c in cands)


# ---------------------------------------------------------------------------
# generate_seeds_for_rmsnorm_residual
# ---------------------------------------------------------------------------


def _assert_candidate_files(folder):
    assert (folder / "kernel.py").exists(), f"missing kernel.py in {folder}"
    assert (folder / "topoflow_metadata.json").exists(), f"missing metadata in {folder}"
    assert (folder / "topoflow_notes.md").exists(), f"missing notes in {folder}"


def test_rmsnorm_generates_at_least_8_candidates(tmp_path):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_rmsnorm_residual(
        {"M": 2048, "N": 4096}, MI300X, out_dir
    )
    assert len(cands) >= 8
    assert all(isinstance(c, SeedCandidate) for c in cands)
    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert len(folders) == len(cands)
    for d in folders:
        _assert_candidate_files(d)


def test_rmsnorm_metadata_has_required_fields(tmp_path):
    shape = {"M": 2048, "N": 4096}
    out_dir = tmp_path / "seeds"
    generate_seeds_for_rmsnorm_residual(shape, MI300X, out_dir)
    sample = next(out_dir.iterdir())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_rmsnorm_residual"
    assert meta["target_arch"] == "gfx942"
    assert meta["shape"] == shape
    assert "BLOCK_M" in meta["tile_plan"]
    assert "BLOCK_N" in meta["tile_plan"]  # RMSNorm uses BLOCK_N label
    assert meta["cost_model"]["score"] > 0
    assert meta["shape_cost"]["bytes_saved"] > 0
    assert meta["candidate_id"].startswith("rmsnorm_residual_v")


def test_rmsnorm_every_kernel_py_parses_and_has_intent(tmp_path):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_rmsnorm_residual({"M": 2048, "N": 4096}, MI300X, out_dir)
    for d in out_dir.iterdir():
        code = (d / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code
        assert "@triton.jit" in code


def test_rmsnorm_scores_not_all_identical(tmp_path):
    """The tile-cost should differentiate at least some plans."""
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_rmsnorm_residual({"M": 2048, "N": 4096}, MI300X, out_dir)
    scores = {c.metadata["cost_model"]["score"] for c in cands}
    assert len(scores) >= 2


# ---------------------------------------------------------------------------
# generate_seeds_for_bias_gelu_dropout
# ---------------------------------------------------------------------------


def test_bias_gelu_generates_at_least_8_candidates(tmp_path):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_bias_gelu_dropout(
        {"M": 2048, "N": 16384, "dropout_p": 0.1, "seed": 0}, MI300X, out_dir
    )
    assert len(cands) >= 8
    assert all(isinstance(c, SeedCandidate) for c in cands)
    folders = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert len(folders) == len(cands)
    for d in folders:
        _assert_candidate_files(d)


def test_bias_gelu_metadata_has_required_fields(tmp_path):
    shape = {"M": 2048, "N": 16384, "dropout_p": 0.1, "seed": 0}
    out_dir = tmp_path / "seeds"
    generate_seeds_for_bias_gelu_dropout(shape, MI300X, out_dir)
    sample = next(out_dir.iterdir())
    meta = json.loads((sample / "topoflow_metadata.json").read_text())
    assert meta["op"] == "fused_bias_gelu_dropout"
    assert meta["target_arch"] == "gfx942"
    assert meta["shape"] == shape
    assert "BLOCK_M" in meta["tile_plan"]
    assert "BLOCK_N" in meta["tile_plan"]
    assert meta["cost_model"]["score"] > 0
    assert meta["shape_cost"]["bytes_saved"] > 0
    assert meta["candidate_id"].startswith("bias_gelu_dropout_v")


def test_bias_gelu_every_kernel_py_parses_and_has_intent(tmp_path):
    out_dir = tmp_path / "seeds"
    generate_seeds_for_bias_gelu_dropout(
        {"M": 2048, "N": 16384, "dropout_p": 0.1, "seed": 0}, MI300X, out_dir
    )
    for d in out_dir.iterdir():
        code = (d / "kernel.py").read_text()
        ast.parse(code)
        assert "# TOPOFLOW_INTENT" in code
        assert "tl.rand(" in code
        assert "@triton.jit" in code


def test_bias_gelu_scores_not_all_identical(tmp_path):
    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_bias_gelu_dropout(
        {"M": 2048, "N": 16384, "dropout_p": 0.1, "seed": 0}, MI300X, out_dir
    )
    scores = {c.metadata["cost_model"]["score"] for c in cands}
    assert len(scores) >= 2


def test_rmsnorm_missing_shape_keys_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required keys"):
        generate_seeds_for_rmsnorm_residual({"M": 2048}, MI300X, tmp_path)


def test_bias_gelu_missing_shape_keys_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required keys"):
        generate_seeds_for_bias_gelu_dropout({"N": 4096}, MI300X, tmp_path)
