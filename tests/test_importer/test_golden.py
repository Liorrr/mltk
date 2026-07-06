"""Tests for golden-set binding and its codegen emission."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mltk.eval._types import EvalSample
from mltk.eval.dataset import EvalDataset
from mltk.importer.classify import TaskType
from mltk.importer.codegen import generate_pytest
from mltk.importer.golden import (
    SCORING_EXACT,
    SCORING_JUDGE,
    GoldenBindingReport,
    GoldenSpec,
    bind_golden,
    load_golden,
)
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult


def _dataset(inputs, targets=None, metadata=None):
    """Build a small EvalDataset for binding tests."""
    targets = targets or [None] * len(inputs)
    metadata = metadata or [{} for _ in inputs]
    samples = [
        EvalSample(input=i, target=t, metadata=dict(m))
        for i, t, m in zip(inputs, targets, metadata, strict=True)
    ]
    return EvalDataset(name="ds", version="0.1.0", samples=samples)


# ------------------------------------------------------------------
# load_golden
# ------------------------------------------------------------------


def test_load_golden_csv(tmp_path: Path) -> None:
    path = tmp_path / "g.csv"
    path.write_text("id,gold\n1,A\n2,B\n", encoding="utf-8")
    rows = load_golden(path)
    assert rows == [{"id": "1", "gold": "A"}, {"id": "2", "gold": "B"}]


def test_load_golden_tsv(tmp_path: Path) -> None:
    path = tmp_path / "g.tsv"
    path.write_text("id\tgold\n1\tA\n", encoding="utf-8")
    assert load_golden(path) == [{"id": "1", "gold": "A"}]


def test_load_golden_json(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps([{"id": "1", "gold": "A"}]), encoding="utf-8")
    assert load_golden(path) == [{"id": "1", "gold": "A"}]


def test_load_golden_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "1", "gold": "A"}\n\n{"id": "2", "gold": "B"}\n', encoding="utf-8")
    assert load_golden(path) == [
        {"id": "1", "gold": "A"},
        {"id": "2", "gold": "B"},
    ]


def test_load_golden_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "nope.csv")


def test_load_golden_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "g.xml"
    path.write_text("<root/>", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported golden file format"):
        load_golden(path)


def test_load_golden_json_not_a_list(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"id": "1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="array of objects"):
        load_golden(path)


# ------------------------------------------------------------------
# bind_golden -- key join
# ------------------------------------------------------------------


def test_bind_key_join_fills_targets_and_marks_exact() -> None:
    ds = _dataset(
        ["q0", "q1", "q2"],
        metadata=[{"id": "0"}, {"id": "1"}, {"id": "2"}],
    )
    golden = [{"id": "0", "gold": "A0"}, {"id": "2", "gold": "A2"}]
    bound, report = bind_golden(ds, golden, target_column="gold", key="id")

    assert [s.target for s in bound.samples] == ["A0", None, "A2"]
    scoring = [s.metadata["scoring"] for s in bound.samples]
    assert scoring == [SCORING_EXACT, SCORING_JUDGE, SCORING_EXACT]
    assert report.matched == 2
    assert report.unmatched == [1]
    assert report.total == 3
    assert report.match_rate == pytest.approx(2 / 3)


def test_bind_key_column_differs_from_sample_key() -> None:
    ds = _dataset(["q0"], metadata=[{"id": "0"}])
    golden = [{"row_id": "0", "gold": "A0"}]
    bound, report = bind_golden(
        ds, golden, target_column="gold", key="id", golden_key="row_id"
    )
    assert bound.samples[0].target == "A0"
    assert report.matched == 1


def test_bind_key_input_matches_on_input_text() -> None:
    ds = _dataset(["what?", "who?"])
    golden = [{"input": "who?", "gold": "someone"}]
    bound, report = bind_golden(
        ds, golden, target_column="gold", key="input"
    )
    assert [s.target for s in bound.samples] == [None, "someone"]
    assert report.unmatched == [0]


def test_bind_golden_overrides_preexisting_target() -> None:
    ds = _dataset(["q0"], targets=["old"], metadata=[{"id": "0"}])
    golden = [{"id": "0", "gold": "new"}]
    bound, _ = bind_golden(ds, golden, target_column="gold", key="id")
    assert bound.samples[0].target == "new"


def test_bind_preexisting_target_kept_when_no_golden_match() -> None:
    # Sample has an in-dataset golden target but the golden file has no row
    # for it -- it stays "exact", not "judge".
    ds = _dataset(["q0"], targets=["kept"], metadata=[{"id": "0"}])
    golden = [{"id": "99", "gold": "unused"}]
    bound, report = bind_golden(ds, golden, target_column="gold", key="id")
    assert bound.samples[0].target == "kept"
    assert bound.samples[0].metadata["scoring"] == SCORING_EXACT
    assert report.matched == 0
    assert report.unmatched == []


# ------------------------------------------------------------------
# bind_golden -- row order
# ------------------------------------------------------------------


def test_bind_row_order() -> None:
    ds = _dataset(["q0", "q1", "q2"])
    golden = [{"gold": "A0"}, {"gold": "A1"}]
    bound, report = bind_golden(ds, golden, target_column="gold")
    assert [s.target for s in bound.samples] == ["A0", "A1", None]
    assert report.matched == 2
    assert report.unmatched == [2]
    assert report.key is None
    assert "row-order" in report.summary()


# ------------------------------------------------------------------
# bind_golden -- errors + immutability
# ------------------------------------------------------------------


def test_bind_missing_target_column_raises() -> None:
    ds = _dataset(["q0"])
    with pytest.raises(ValueError, match="target_column"):
        bind_golden(ds, [{"id": "0"}], target_column="gold", key="id")


def test_bind_missing_golden_key_raises() -> None:
    ds = _dataset(["q0"], metadata=[{"id": "0"}])
    with pytest.raises(ValueError, match="golden_key"):
        bind_golden(
            ds, [{"gold": "A"}], target_column="gold", key="id"
        )


def test_bind_does_not_mutate_original() -> None:
    ds = _dataset(["q0"], metadata=[{"id": "0"}])
    bind_golden(ds, [{"id": "0", "gold": "A"}], target_column="gold", key="id")
    assert ds.samples[0].target is None
    assert "scoring" not in ds.samples[0].metadata


def test_bind_recomputes_fingerprint() -> None:
    ds = _dataset(["q0"], metadata=[{"id": "0"}])
    bound, _ = bind_golden(
        ds, [{"id": "0", "gold": "A"}], target_column="gold", key="id"
    )
    # A concrete target changes hashed content, so the fingerprint changes.
    assert bound.fingerprint != ds.fingerprint
    assert bound.card is ds.card  # provenance preserved


def test_report_is_dataclass() -> None:
    report = GoldenBindingReport(total=0, matched=0, unmatched=[], key=None)
    assert report.match_rate == 0.0


# ------------------------------------------------------------------
# codegen emission with golden_spec
# ------------------------------------------------------------------


def _import_result():
    rows = [
        {"question": f"q{i}", "answer": f"a{i}", "id": str(i)}
        for i in range(3)
    ]
    return ImportResult(
        source="qa.csv",
        columns=["question", "answer", "id"],
        dtypes={"question": "string", "answer": "string", "id": "string"},
        rows=rows,
        mapping=ColumnMapping(
            roles={
                "question": ColumnRole.INPUT,
                "answer": ColumnRole.GOLDEN,
                "id": ColumnRole.METADATA,
            }
        ),
    )


def test_generate_pytest_golden_judge_emits_binding_and_judge() -> None:
    spec = GoldenSpec(
        path="golden.csv",
        target_column="gold",
        key="id",
        golden_key="row_id",
        judge=True,
    )
    code = generate_pytest(
        _import_result(),
        TaskType.GENERATION,
        dataset_name="qa",
        golden_spec=spec,
    )
    ast.parse(code)
    assert "from mltk.importer.golden import bind_golden, load_golden" in code
    assert "from mltk.domains.llm.judge import assert_llm_judge_score" in code
    assert "def judge_fn():" in code
    assert "def test_judge_scored_samples(" in code
    assert "target_column='gold'" in code
    assert "key='id'" in code
    assert "golden_key='row_id'" in code
    assert "MLTK_JUDGE_FN" in code


def test_generate_pytest_golden_without_judge_has_no_judge_test() -> None:
    spec = GoldenSpec(
        path="golden.csv", target_column="gold", key=None, judge=False
    )
    code = generate_pytest(
        _import_result(),
        TaskType.GENERATION,
        dataset_name="qa",
        golden_spec=spec,
    )
    ast.parse(code)
    assert "bind_golden" in code
    assert "key=None" in code
    assert "judge_fn" not in code
    assert "assert_llm_judge_score" not in code


def test_generate_pytest_no_golden_unchanged() -> None:
    code = generate_pytest(
        _import_result(), TaskType.GENERATION, dataset_name="qa"
    )
    ast.parse(code)
    assert "bind_golden" not in code
    assert "judge_fn" not in code
