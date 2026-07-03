"""Tests for mltk.importer.schema -- normalized import schema.

Covers ColumnRole, ColumnMapping, and ImportResult.to_eval_dataset(),
the contract that turns raw imported rows into a versioned EvalDataset.
"""

from __future__ import annotations

import pytest

from mltk.eval.dataset import EvalDataset
from mltk.importer.schema import ColumnMapping, ColumnRole, ImportResult

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "passage": "France is in Western Europe.",
            "category": "geography",
            "id": 1,
        },
        {
            "question": "What is 7 plus 5?",
            "answer": "12",
            "passage": "Addition combines two numbers.",
            "category": "math",
            "id": 2,
        },
    ]


def _dtypes() -> dict[str, str]:
    return {
        "question": "string",
        "answer": "string",
        "passage": "string",
        "category": "string",
        "id": "numeric",
    }


def _mapping(**role_overrides: ColumnRole) -> ColumnMapping:
    roles = {
        "question": ColumnRole.INPUT,
        "answer": ColumnRole.GOLDEN,
        "passage": ColumnRole.CONTEXT,
        "category": ColumnRole.LABEL,
        "id": ColumnRole.METADATA,
    }
    roles.update(role_overrides)
    return ColumnMapping(roles=roles, samples=_rows()[0])


def _import_result(mapping: ColumnMapping | None = None) -> ImportResult:
    return ImportResult(
        source="tests/fixtures/tiny.csv",
        columns=list(_dtypes().keys()),
        dtypes=_dtypes(),
        rows=_rows(),
        mapping=mapping if mapping is not None else _mapping(),
    )


# ===============================================================
# ColumnRole
# ===============================================================


class TestColumnRole:
    """ColumnRole: enum of normalized column purposes."""

    def test_all_values_present(self):
        # SCENARIO: enum defines all 7 documented roles
        # WHY: contract requires exactly these members
        # EXPECTED: all names/values exist
        assert ColumnRole.INPUT.value == "input"
        assert ColumnRole.GOLDEN.value == "golden"
        assert ColumnRole.CONTEXT.value == "context"
        assert ColumnRole.LABEL.value == "label"
        assert ColumnRole.METADATA.value == "metadata"
        assert ColumnRole.IGNORE.value == "ignore"
        assert ColumnRole.UNKNOWN.value == "unknown"

    def test_member_count(self):
        # SCENARIO: no extra roles snuck in
        # WHY: downstream code switches exhaustively on role
        # EXPECTED: exactly 7 members
        assert len(list(ColumnRole)) == 7

    def test_roles_are_distinct(self):
        # SCENARIO: no duplicate values
        # WHY: dict keys keyed by role must not collide
        # EXPECTED: unique value set
        values = [r.value for r in ColumnRole]
        assert len(values) == len(set(values))


# ===============================================================
# ColumnMapping
# ===============================================================


class TestColumnMappingPreview:
    """ColumnMapping.preview() -- human-readable rendering."""

    def test_preview_returns_string(self):
        # SCENARIO: preview a valid mapping
        # WHY: must be a printable summary
        # EXPECTED: non-empty str
        mapping = _mapping()
        preview = mapping.preview()
        assert isinstance(preview, str)
        assert len(preview) > 0

    def test_preview_lists_all_columns(self):
        # SCENARIO: preview mentions every mapped column
        # WHY: user must see the full mapping at a glance
        # EXPECTED: each column name appears in the output
        mapping = _mapping()
        preview = mapping.preview()
        for col in ("question", "answer", "passage", "category", "id"):
            assert col in preview

    def test_preview_flags_unknown_column(self):
        # SCENARIO: mapping has an UNKNOWN column
        # WHY: unknown columns need visible attention from the user
        # EXPECTED: preview highlights UNKNOWN somehow
        mapping = _mapping(category=ColumnRole.UNKNOWN)
        preview = mapping.preview()
        assert "category" in preview
        assert "unknown" in preview.lower() or "UNKNOWN" in preview


class TestColumnMappingOverride:
    """ColumnMapping.override() -- immutable role reassignment."""

    def test_override_returns_new_mapping(self):
        # SCENARIO: override a known column's role
        # WHY: contract says override returns a NEW ColumnMapping
        # EXPECTED: returned value is a distinct ColumnMapping
        mapping = _mapping()
        updated = mapping.override("category", ColumnRole.METADATA)
        assert isinstance(updated, ColumnMapping)
        assert updated is not mapping

    def test_override_does_not_mutate_original(self):
        # SCENARIO: override then inspect the original
        # WHY: mapping objects must be treated as immutable
        # EXPECTED: original roles dict unchanged
        mapping = _mapping()
        mapping.override("category", ColumnRole.METADATA)
        assert mapping.roles["category"] == ColumnRole.LABEL

    def test_override_applies_new_role(self):
        # SCENARIO: override changes the target column's role
        # WHY: core purpose of override()
        # EXPECTED: new mapping reflects requested role
        mapping = _mapping()
        updated = mapping.override("category", ColumnRole.METADATA)
        assert updated.roles["category"] == ColumnRole.METADATA

    def test_override_preserves_other_columns(self):
        # SCENARIO: override one column, others stay the same
        # WHY: override must be scoped to a single column
        # EXPECTED: unrelated roles are identical
        mapping = _mapping()
        updated = mapping.override("category", ColumnRole.METADATA)
        assert updated.roles["question"] == ColumnRole.INPUT
        assert updated.roles["answer"] == ColumnRole.GOLDEN
        assert updated.roles["passage"] == ColumnRole.CONTEXT
        assert updated.roles["id"] == ColumnRole.METADATA

    def test_override_unknown_column_raises(self):
        # SCENARIO: override a column that isn't in the mapping
        # WHY: must fail loudly instead of silently no-op'ing
        # EXPECTED: ValueError
        mapping = _mapping()
        with pytest.raises(ValueError, match="nonexistent_column"):
            mapping.override("nonexistent_column", ColumnRole.INPUT)


class TestColumnMappingColumnsWithRole:
    """ColumnMapping.columns_with_role() -- filter by role."""

    def test_filters_single_match(self):
        # SCENARIO: exactly one column has GOLDEN role
        # WHY: basic filter behavior
        # EXPECTED: list containing only that column
        mapping = _mapping()
        assert mapping.columns_with_role(ColumnRole.GOLDEN) == ["answer"]

    def test_filters_no_match_returns_empty(self):
        # SCENARIO: no column has IGNORE role
        # WHY: absence must yield empty list, not None/error
        # EXPECTED: empty list
        mapping = _mapping()
        assert mapping.columns_with_role(ColumnRole.IGNORE) == []

    def test_filters_multiple_matches_preserve_order(self):
        # SCENARIO: two columns share the same role
        # WHY: ordering must follow the roles dict, not be resorted
        # EXPECTED: both columns present, in insertion order
        roles = {
            "context_a": ColumnRole.CONTEXT,
            "input": ColumnRole.INPUT,
            "context_b": ColumnRole.CONTEXT,
        }
        mapping = ColumnMapping(roles=roles, samples={})
        result = mapping.columns_with_role(ColumnRole.CONTEXT)
        assert result == ["context_a", "context_b"]


class TestColumnMappingValidate:
    """ColumnMapping.validate() -- structural validity checks."""

    def test_valid_mapping_returns_empty_list(self):
        # SCENARIO: mapping has exactly one INPUT, no UNKNOWN
        # WHY: happy path must report zero problems
        # EXPECTED: empty list
        mapping = _mapping()
        assert mapping.validate() == []

    def test_zero_input_columns_flagged(self):
        # SCENARIO: no column mapped to INPUT
        # WHY: to_eval_dataset() cannot run without an INPUT column
        # EXPECTED: non-empty problem list
        mapping = _mapping(question=ColumnRole.METADATA)
        problems = mapping.validate()
        assert len(problems) > 0

    def test_unknown_columns_flagged(self):
        # SCENARIO: a column is left as UNKNOWN
        # WHY: user should be nudged to resolve ambiguous columns
        # EXPECTED: non-empty problem list
        mapping = _mapping(category=ColumnRole.UNKNOWN)
        problems = mapping.validate()
        assert len(problems) > 0

    def test_problems_are_strings(self):
        # SCENARIO: validate() on an invalid mapping
        # WHY: problems must be human-readable, not opaque codes
        # EXPECTED: every problem is a str
        mapping = _mapping(question=ColumnRole.METADATA)
        problems = mapping.validate()
        assert all(isinstance(p, str) for p in problems)


# ===============================================================
# ImportResult.to_eval_dataset
# ===============================================================


class TestImportResultToEvalDataset:
    """ImportResult.to_eval_dataset() -- rows -> EvalDataset."""

    def test_returns_eval_dataset(self):
        # SCENARIO: basic happy-path conversion
        # WHY: type contract
        # EXPECTED: EvalDataset instance
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert isinstance(ds, EvalDataset)

    def test_sample_count_matches_rows(self):
        # SCENARIO: 2 input rows
        # WHY: no rows dropped silently
        # EXPECTED: sample_count == 2
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.sample_count == 2

    def test_name_and_version_applied(self):
        # SCENARIO: explicit name + version passed
        # WHY: caller controls dataset identity
        # EXPECTED: fields set accordingly
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa", version="2.3.1")
        assert ds.name == "tiny-qa"
        assert ds.version == "2.3.1"

    def test_default_version(self):
        # SCENARIO: version omitted
        # WHY: contract default is "0.1.0"
        # EXPECTED: version == "0.1.0"
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.version == "0.1.0"

    def test_input_column_maps_to_sample_input(self):
        # SCENARIO: single INPUT column
        # WHY: core mapping contract
        # EXPECTED: EvalSample.input == question text
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].input == "What is the capital of France?"
        assert ds.samples[1].input == "What is 7 plus 5?"

    def test_golden_column_maps_to_sample_target(self):
        # SCENARIO: single GOLDEN column
        # WHY: core mapping contract
        # EXPECTED: EvalSample.target == answer text
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].target == "Paris"
        assert ds.samples[1].target == "12"

    def test_single_context_column_maps_to_metadata_str(self):
        # SCENARIO: single CONTEXT column
        # WHY: contract says single context -> plain str
        # EXPECTED: metadata["context"] is the passage string
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["context"] == (
            "France is in Western Europe."
        )
        assert isinstance(ds.samples[0].metadata["context"], str)

    def test_single_label_column_maps_to_metadata_scalar(self):
        # SCENARIO: single LABEL column
        # WHY: contract says single label -> scalar value
        # EXPECTED: metadata["label"] equals raw category value
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["label"] == "geography"
        assert ds.samples[1].metadata["label"] == "math"

    def test_metadata_column_lands_in_metadata_by_name(self):
        # SCENARIO: METADATA-role column ("id")
        # WHY: metadata columns keep their own name as key
        # EXPECTED: metadata["id"] == raw id value
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["id"] == 1
        assert ds.samples[1].metadata["id"] == 2

    def test_unknown_column_lands_in_metadata_by_name(self):
        # SCENARIO: a column left UNKNOWN
        # WHY: contract treats UNKNOWN like METADATA for output
        # EXPECTED: metadata[col] == raw value
        mapping = _mapping(id=ColumnRole.UNKNOWN)
        result = _import_result(mapping=mapping)
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["id"] == 1

    def test_ignore_column_dropped(self):
        # SCENARIO: a column marked IGNORE
        # WHY: user explicitly excludes noise columns
        # EXPECTED: key absent from metadata entirely
        mapping = _mapping(id=ColumnRole.IGNORE)
        result = _import_result(mapping=mapping)
        ds = result.to_eval_dataset(name="tiny-qa")
        assert "id" not in ds.samples[0].metadata

    def test_multiple_golden_columns_produce_references(self):
        # SCENARIO: two GOLDEN columns (answer, passage)
        # WHY: contract: first GOLDEN -> target, rest -> references list
        # EXPECTED: target from first, references from remaining
        mapping = _mapping(passage=ColumnRole.GOLDEN)
        result = _import_result(mapping=mapping)
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].target == "Paris"
        assert ds.samples[0].metadata["references"] == [
            "France is in Western Europe."
        ]
        assert isinstance(ds.samples[0].metadata["references"], list)

    def test_multiple_context_columns_produce_list(self):
        # SCENARIO: two CONTEXT columns (passage, category)
        # WHY: contract: multiple context -> list[str] in metadata
        # EXPECTED: metadata["context"] is a list of both values
        mapping = _mapping(category=ColumnRole.CONTEXT)
        result = _import_result(mapping=mapping)
        ds = result.to_eval_dataset(name="tiny-qa")
        ctx = ds.samples[0].metadata["context"]
        assert isinstance(ctx, list)
        assert "France is in Western Europe." in ctx
        assert "geography" in ctx

    def test_multiple_label_columns_produce_dict(self):
        # SCENARIO: two LABEL columns (category, id)
        # WHY: contract: multiple label -> dict in metadata
        # EXPECTED: metadata["label"] is a dict keyed by column name
        mapping = _mapping(id=ColumnRole.LABEL)
        result = _import_result(mapping=mapping)
        ds = result.to_eval_dataset(name="tiny-qa")
        label = ds.samples[0].metadata["label"]
        assert isinstance(label, dict)
        assert label["category"] == "geography"
        assert label["id"] == 1

    def test_zero_input_columns_raises(self):
        # SCENARIO: no column mapped to INPUT
        # WHY: cannot build EvalSample without an input
        # EXPECTED: ValueError
        mapping = _mapping(question=ColumnRole.METADATA)
        result = _import_result(mapping=mapping)
        with pytest.raises(ValueError):  # noqa: PT011 -- message text not part of contract
            result.to_eval_dataset(name="tiny-qa")

    def test_card_source_matches_result_source(self):
        # SCENARIO: ImportResult with a given source string
        # WHY: provenance tracking via DatasetCard
        # EXPECTED: ds.card.source == result.source
        result = _import_result()
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.card is not None
        assert ds.card.source == result.source

    def test_explicit_mapping_overrides_result_mapping(self):
        # SCENARIO: caller passes a different mapping than result.mapping
        # WHY: to_eval_dataset(mapping=...) lets callers override roles
        #   without mutating the stored ImportResult.mapping
        # EXPECTED: explicit mapping wins over result.mapping
        result = _import_result()
        alt_mapping = _mapping(passage=ColumnRole.IGNORE)
        ds = result.to_eval_dataset(name="tiny-qa", mapping=alt_mapping)
        assert "context" not in ds.samples[0].metadata

    def test_missing_golden_cell_is_none_not_string_none(self):
        # SCENARIO: a row's GOLDEN cell is missing (empty string)
        # WHY: a missing value must become None, never the literal
        #   string "None" from a bare str(None) call
        # EXPECTED: target is None
        rows = _rows()
        rows[0]["answer"] = ""
        result = _import_result()
        result.rows = rows
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].target is None

    def test_missing_input_cell_becomes_empty_string_not_string_none(self):
        # SCENARIO: a row's INPUT cell is missing
        # WHY: EvalSample.input is non-optional str, so a missing input
        #   must become "" -- never the literal string "None"
        # EXPECTED: input == ""
        rows = _rows()
        rows[0]["question"] = None
        result = _import_result()
        result.rows = rows
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].input == ""

    def test_missing_extra_golden_cell_is_none_in_references(self):
        # SCENARIO: two GOLDEN columns, the second (extra) cell is missing
        # WHY: references entries must mirror the primary target's
        #   missing-value handling, not silently become "None" strings
        # EXPECTED: metadata["references"] contains None for that cell
        rows = _rows()
        rows[0]["passage"] = None
        mapping = _mapping(passage=ColumnRole.GOLDEN)
        result = _import_result(mapping=mapping)
        result.rows = rows
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["references"] == [None]

    def test_missing_context_cell_is_none_not_string_none(self):
        # SCENARIO: single CONTEXT column, cell is missing
        # WHY: same "None" literal-string bug class as INPUT/references
        # EXPECTED: metadata["context"] is None
        rows = _rows()
        rows[0]["passage"] = None
        result = _import_result()
        result.rows = rows
        ds = result.to_eval_dataset(name="tiny-qa")
        assert ds.samples[0].metadata["context"] is None

    def test_no_input_column_zero_samples_still_raises(self):
        # SCENARIO: rows present but INPUT role missing entirely
        # WHY: even non-empty rows must fail without INPUT
        # EXPECTED: ValueError, not a dataset with empty inputs
        roles = {
            "question": ColumnRole.METADATA,
            "answer": ColumnRole.GOLDEN,
        }
        mapping = ColumnMapping(roles=roles, samples={})
        result = ImportResult(
            source="x.csv",
            columns=["question", "answer"],
            dtypes={"question": "string", "answer": "string"},
            rows=[{"question": "Q?", "answer": "A"}],
            mapping=mapping,
        )
        with pytest.raises(ValueError):  # noqa: PT011 -- message text not part of contract
            result.to_eval_dataset(name="no-input")
