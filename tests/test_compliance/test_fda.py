"""Tests for FDA 21 CFR Part 11 audit trail generation."""

from __future__ import annotations

import json
from pathlib import Path

from mltk.compliance.fda import generate_fda_audit_trail

SAMPLE = [
    {"name": "data.schema", "passed": True, "severity": "info", "message": "ok"},
    {"name": "model.bias", "passed": False, "severity": "critical", "message": "gap 0.15"},
]


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "results.json"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return p


def test_creates_file(tmp_path):
    out = tmp_path / "audit.md"
    result = generate_fda_audit_trail(_write(tmp_path), output_path=str(out))
    assert result.exists()
    assert result.suffix == ".md"


def test_contains_sections(tmp_path):
    out = tmp_path / "audit.md"
    generate_fda_audit_trail(_write(tmp_path), output_path=str(out))
    content = out.read_text()
    assert "System Information" in content
    assert "Operator" in content
    assert "Test Evidence" in content
    assert "Digital Signature" in content
    assert "Regulatory Notes" in content


def test_contains_test_results(tmp_path):
    out = tmp_path / "audit.md"
    generate_fda_audit_trail(_write(tmp_path), output_path=str(out))
    content = out.read_text()
    assert "data.schema" in content
    assert "model.bias" in content
    assert "PASS" in content
    assert "FAIL" in content


def test_custom_operator(tmp_path):
    out = tmp_path / "audit.md"
    generate_fda_audit_trail(
        _write(tmp_path), operator="Jane Doe", output_path=str(out)
    )
    assert "Jane Doe" in out.read_text()


def test_custom_system_name(tmp_path):
    out = tmp_path / "audit.md"
    generate_fda_audit_trail(
        _write(tmp_path), system_name="ML Pipeline v3", output_path=str(out)
    )
    assert "ML Pipeline v3" in out.read_text()


def test_empty_results(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    out = tmp_path / "audit.md"
    result = generate_fda_audit_trail(str(p), output_path=str(out))
    assert result.exists()
    assert "0/0" in out.read_text()
