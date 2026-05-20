import ast
import json
from pathlib import Path

import pytest

from topoflow_prior.seed_generator import generate_seeds_for_fused_silu_mul_fp8_quant
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
    from topoflow_prior.schemas import SeedCandidate

    out_dir = tmp_path / "seeds"
    cands = generate_seeds_for_fused_silu_mul_fp8_quant(shape, MI300X, out_dir)
    assert all(isinstance(c, SeedCandidate) for c in cands)
