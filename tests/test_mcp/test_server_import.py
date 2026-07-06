"""Tests for the mltk_import MCP tool (tool #13).

Covers return-only behavior, opt-in file write, golden binding,
registration behind MLTK_DATASET_DIR, and error responses.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ._helpers import (
    assert_error,
    assert_has_workflow_hint,
    assert_ok,
    registered_tools,
)

REPO_ROOT = Path(__file__).parents[2]
TINY_CSV = REPO_ROOT / "tests" / "test_importer" / "fixtures" / "tiny.csv"


def _call_import(**kwargs):
    """Call mltk_import directly (its ``name`` param collides with call_tool)."""
    fn = registered_tools["mltk_import"]
    return json.loads(fn(**kwargs))


def test_import_return_only(tmp_path: Path) -> None:
    data = _call_import(source=str(TINY_CSV), name="tiny")
    assert_ok(data)
    assert_has_workflow_hint(data)
    assert data["task_type"] == "qa_rag"
    assert data["dataset_name"] == "tiny"
    assert data["sample_count"] == 5
    assert data["assertion_count"] >= 1
    assert data["file_written"] is None
    # The returned code is a valid, self-contained pytest module.
    ast.parse(data["generated_code"])
    assert "column | role | sample" in data["mapping_preview"]


def test_import_writes_file_when_output_path_set(tmp_path: Path) -> None:
    out = tmp_path / "test_tiny.py"
    data = _call_import(source=str(TINY_CSV), name="tiny", output_path=str(out))
    assert_ok(data)
    assert data["file_written"] == str(out)
    assert out.exists()
    ast.parse(out.read_text(encoding="utf-8"))


def test_import_with_golden_binding(tmp_path: Path) -> None:
    golden = tmp_path / "golden.csv"
    golden.write_text(
        "id,gold\n1,Paris\n3,Tokyo\n", encoding="utf-8"
    )
    data = _call_import(
        source=str(TINY_CSV),
        name="tiny",
        golden_path=str(golden),
        golden_target_column="gold",
        golden_key="id",
        judge=True,
    )
    assert_ok(data)
    assert "golden_binding" in data
    assert "matched" in data["golden_binding"]
    assert "bind_golden" in data["generated_code"]
    assert "test_judge_scored_samples" in data["generated_code"]
    ast.parse(data["generated_code"])


def test_import_golden_without_target_column_errors() -> None:
    data = _call_import(
        source=str(TINY_CSV), golden_path="whatever.csv"
    )
    assert_error(data)
    assert "golden_target_column" in data["error"]


def test_import_register_uses_dataset_dir_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLTK_DATASET_DIR", str(tmp_path))
    data = _call_import(source=str(TINY_CSV), name="tiny", register=True)
    assert_ok(data)
    assert data["registration"]["saved"] is True
    assert data["registration"]["quality_passed"] is True
    assert (tmp_path / "tiny" / "0.1.0" / "dataset.json").exists()


def test_import_register_blocked_by_quality_gate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLTK_DATASET_DIR", str(tmp_path))
    dupes = tmp_path / "dupes.csv"
    dupes.write_text(
        "question,answer\nsame?,a\nsame?,b\nsame?,c\nsame?,d\n",
        encoding="utf-8",
    )
    data = _call_import(source=str(dupes), name="dupes", register=True)
    assert_ok(data)  # the import itself succeeds; only the save is gated
    assert data["registration"]["saved"] is False
    assert data["registration"]["quality_passed"] is False
    assert not (tmp_path / "dupes").exists()


def test_import_missing_source_errors() -> None:
    data = _call_import(source="does/not/exist.csv")
    assert_error(data)
