"""Sanity checks for experiment shape YAML configs.

We do not depend on PyYAML at runtime; these tests use text-level checks to
verify the configs exist, contain the expected keys, and don't drift.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"


def test_fused_silu_mul_fp8_quant_shapes_file_exists():
    path = EXP_ROOT / "fused_silu_mul_fp8_quant" / "shapes.yaml"
    assert path.exists(), f"missing config: {path}"


def test_attention_swizzle_shapes_file_exists():
    path = EXP_ROOT / "attention_swizzle" / "shapes.yaml"
    assert path.exists(), f"missing config: {path}"


def test_fused_shapes_yaml_has_required_keys():
    text = (EXP_ROOT / "fused_silu_mul_fp8_quant" / "shapes.yaml").read_text()
    assert "shapes:" in text
    # All five expected fused-target entries are present (E/T/H/group_size).
    for line in text.splitlines():
        if not line.strip().startswith("- {"):
            continue
        for k in ("E:", "T:", "H:", "group_size:"):
            assert k in line, f"missing {k} in entry: {line}"
    # Must declare at least five shapes (small/medium/large/llama-7/many-experts).
    entries = [l for l in text.splitlines() if l.strip().startswith("- {")]
    assert len(entries) >= 5, f"expected >=5 fused shapes, got {len(entries)}"


def test_attention_shapes_yaml_has_required_keys():
    text = (EXP_ROOT / "attention_swizzle" / "shapes.yaml").read_text()
    assert "shapes:" in text
    for line in text.splitlines():
        if not line.strip().startswith("- {"):
            continue
        for k in ("batch:", "seq_len:", "num_heads:", "head_dim:"):
            assert k in line, f"missing {k} in entry: {line}"
    entries = [l for l in text.splitlines() if l.strip().startswith("- {")]
    assert len(entries) >= 3, f"expected >=3 attention shapes, got {len(entries)}"


def test_attention_num_heads_divisible_by_num_xcds():
    """Swizzled-head-first requires num_heads % 8 == 0 (8 XCDs on MI300X)."""
    text = (EXP_ROOT / "attention_swizzle" / "shapes.yaml").read_text()
    for line in text.splitlines():
        m = re.search(r"num_heads:\s*(\d+)", line)
        if not m:
            continue
        n_heads = int(m.group(1))
        assert n_heads % 8 == 0, (
            f"num_heads={n_heads} is not divisible by num_xcds=8; "
            f"swizzled_head_first will reject this shape"
        )


def test_fused_h_divisible_by_group_size():
    """Cost model and tile planner both require H % group_size == 0."""
    text = (EXP_ROOT / "fused_silu_mul_fp8_quant" / "shapes.yaml").read_text()
    for line in text.splitlines():
        h_match = re.search(r"H:\s*(\d+)", line)
        gs_match = re.search(r"group_size:\s*(\d+)", line)
        if not (h_match and gs_match):
            continue
        H = int(h_match.group(1))
        gs = int(gs_match.group(1))
        assert H % gs == 0, f"H={H} not divisible by group_size={gs}"


def test_rmsnorm_residual_shapes_file_exists_and_well_formed():
    path = EXP_ROOT / "rmsnorm_residual" / "shapes.yaml"
    assert path.exists(), f"missing config: {path}"
    text = path.read_text()
    assert "shapes:" in text
    entries = [l for l in text.splitlines() if l.strip().startswith("- {")]
    assert len(entries) >= 4, f"expected >=4 rmsnorm shapes, got {len(entries)}"
    for line in entries:
        for k in ("M:", "N:"):
            assert k in line, f"missing {k} in entry: {line}"


def test_bias_gelu_dropout_shapes_file_exists_and_well_formed():
    path = EXP_ROOT / "bias_gelu_dropout" / "shapes.yaml"
    assert path.exists(), f"missing config: {path}"
    text = path.read_text()
    assert "shapes:" in text
    entries = [l for l in text.splitlines() if l.strip().startswith("- {")]
    assert len(entries) >= 3, f"expected >=3 bias_gelu shapes, got {len(entries)}"
    for line in entries:
        for k in ("M:", "N:", "dropout_p:"):
            assert k in line, f"missing {k} in entry: {line}"
