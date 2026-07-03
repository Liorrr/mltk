"""Tests for mltk.importer.mapping -- deterministic column auto-mapping.

Exercises the name-based heuristic priority table documented in the
Sprint 1 contract: INPUT > CONTEXT > GOLDEN (string-only) > LABEL >
METADATA > UNKNOWN.
"""

from __future__ import annotations

import pytest

from mltk.importer.mapping import auto_map_columns
from mltk.importer.schema import ColumnMapping, ColumnRole

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _map(
    columns: list[str],
    dtypes: dict[str, str] | None = None,
    rows: list[dict] | None = None,
) -> ColumnMapping:
    if dtypes is None:
        dtypes = dict.fromkeys(columns, "string")
    if rows is None:
        rows = [{c: f"val-{c}" for c in columns}]
    return auto_map_columns(columns, dtypes, rows)


# ===============================================================
# Heuristic 1: INPUT
# ===============================================================


class TestInputHeuristic:
    """Columns named 'input'/'prompt'/'question'/'query'/'text' -> INPUT."""

    @pytest.mark.parametrize(
        "name",
        ["input", "prompt", "question", "query", "text"],
    )
    def test_exact_names_map_to_input(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.INPUT

    @pytest.mark.parametrize(
        "name",
        ["Input", "PROMPT", "Question", "QUERY", "Text"],
    )
    def test_case_insensitive(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.INPUT

    def test_substring_match(self):
        # SCENARIO: column name contains the keyword, not equal to it
        # WHY: heuristics are documented as "contains", not "equals"
        # EXPECTED: still classified as INPUT
        mapping = _map(["user_question_text"])
        assert mapping.roles["user_question_text"] == ColumnRole.INPUT

    def test_input_wins_over_context_when_both_match(self):
        # SCENARIO: column name matches both INPUT ('query') and
        #   CONTEXT ('context') keywords
        # WHY: rule 1 (INPUT) is resolved for all columns before rule 2
        #   (CONTEXT) is ever considered, so INPUT always wins on collision
        # EXPECTED: INPUT wins
        mapping = _map(["query_context"])
        assert mapping.roles["query_context"] == ColumnRole.INPUT


class TestTextFallbackHeuristic:
    """Rule 6: a lone 'text' column becomes INPUT only as a last resort."""

    def test_lone_text_column_becomes_input(self):
        # SCENARIO: 'text' is the only free-text candidate, no other
        #   column claimed INPUT via rule 1
        # WHY: contract rule 6 -- sole remaining free-text column wins
        # EXPECTED: INPUT
        mapping = _map(["text", "id"])
        assert mapping.roles["text"] == ColumnRole.INPUT

    def test_text_column_unknown_when_input_already_claimed(self):
        # SCENARIO: 'question' already claimed INPUT via rule 1; a
        #   separate 'raw_text' column also contains 'text'
        # WHY: rule 6 only fires when no column matched INPUT via rule 1
        # EXPECTED: 'raw_text' falls back to UNKNOWN, not INPUT
        mapping = _map(["question", "raw_text"])
        assert mapping.roles["question"] == ColumnRole.INPUT
        assert mapping.roles["raw_text"] == ColumnRole.UNKNOWN

    def test_multiple_text_candidates_become_unknown(self):
        # SCENARIO: two columns both contain 'text' and no INPUT match
        # WHY: rule 6 requires a single unambiguous candidate
        # EXPECTED: both fall back to UNKNOWN
        mapping = _map(["body_text", "raw_text"])
        assert mapping.roles["body_text"] == ColumnRole.UNKNOWN
        assert mapping.roles["raw_text"] == ColumnRole.UNKNOWN


# NOTE: an earlier draft of this suite flagged a suspected divergence
# between the Sprint 1 contract and this implementation over the 'text'
# keyword. Re-reading the contract's rule 1 resolved it: 'text' was
# always specified as conditional ("... when text is the only obvious
# free-text field and no more specific name matched"), matching exactly
# what auto_map_columns() implements as rule 6. No divergence -- see
# TestTextFallbackHeuristic above for the behavior this covers.


# ===============================================================
# Heuristic 2: CONTEXT
# ===============================================================


class TestContextHeuristic:
    """Columns containing context/passage/document/chunk/retrieved -> CONTEXT."""

    @pytest.mark.parametrize(
        "name",
        ["context", "passage", "document", "chunk", "retrieved"],
    )
    def test_exact_names_map_to_context(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.CONTEXT

    @pytest.mark.parametrize(
        "name",
        ["Context", "PASSAGE", "Document", "CHUNK", "Retrieved"],
    )
    def test_case_insensitive(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.CONTEXT

    def test_substring_match(self):
        # SCENARIO: 'retrieved_passages' contains 'retrieved' and 'passage'
        # WHY: contains-match, not exact
        # EXPECTED: CONTEXT
        mapping = _map(["retrieved_passages"])
        assert mapping.roles["retrieved_passages"] == ColumnRole.CONTEXT

    def test_context_wins_over_metadata_when_both_match(self):
        # SCENARIO: 'document_id' matches CONTEXT ('document') and
        #   METADATA ('id')
        # WHY: priority order puts CONTEXT (rule 2) before METADATA (rule 5)
        # EXPECTED: CONTEXT wins
        mapping = _map(["document_id"])
        assert mapping.roles["document_id"] == ColumnRole.CONTEXT


# ===============================================================
# Heuristic 3: GOLDEN (string dtype required)
# ===============================================================


class TestGoldenHeuristic:
    """Columns containing answer/target/expected/golden/reference/output/
    completion/response, AND dtype == 'string' -> GOLDEN."""

    @pytest.mark.parametrize(
        "name",
        [
            "answer",
            "target",
            "expected",
            "golden",
            "reference",
            "output",
            "completion",
            "response",
        ],
    )
    def test_exact_names_with_string_dtype_map_to_golden(self, name):
        mapping = _map([name], dtypes={name: "string"})
        assert mapping.roles[name] == ColumnRole.GOLDEN

    @pytest.mark.parametrize(
        "name",
        ["Answer", "TARGET", "Expected", "GOLDEN", "Reference"],
    )
    def test_case_insensitive(self, name):
        mapping = _map([name], dtypes={name: "string"})
        assert mapping.roles[name] == ColumnRole.GOLDEN

    def test_non_string_dtype_does_not_map_to_golden(self):
        # SCENARIO: column name matches GOLDEN keyword but dtype is numeric
        # WHY: contract requires dtype == "string" for this rule to fire
        # EXPECTED: not GOLDEN (falls through remaining rules -> UNKNOWN
        #   since it matches no other keyword)
        mapping = _map(["answer_score"], dtypes={"answer_score": "numeric"})
        assert mapping.roles["answer_score"] != ColumnRole.GOLDEN

    def test_non_string_dtype_falls_through_to_unknown(self):
        # SCENARIO: 'target' column with boolean dtype, no other keyword
        # WHY: golden rule is gated on string dtype; nothing else matches
        # EXPECTED: UNKNOWN
        mapping = _map(["target"], dtypes={"target": "boolean"})
        assert mapping.roles["target"] == ColumnRole.UNKNOWN

    def test_golden_wins_over_label_when_both_match(self):
        # SCENARIO: 'response_class' matches GOLDEN ('response') and
        #   LABEL ('class')
        # WHY: priority order puts GOLDEN (rule 3) before LABEL (rule 4)
        # EXPECTED: GOLDEN wins (dtype is string)
        mapping = _map(["response_class"], dtypes={"response_class": "string"})
        assert mapping.roles["response_class"] == ColumnRole.GOLDEN


# ===============================================================
# Heuristic 4: LABEL
# ===============================================================


class TestLabelHeuristic:
    """Columns containing label/class/category -> LABEL."""

    @pytest.mark.parametrize("name", ["label", "class", "category"])
    def test_exact_names_map_to_label(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.LABEL

    @pytest.mark.parametrize("name", ["Label", "CLASS", "Category"])
    def test_case_insensitive(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.LABEL

    def test_label_wins_over_metadata_when_both_match(self):
        # SCENARIO: 'class_source' matches LABEL ('class') and
        #   METADATA ('source')
        # WHY: priority order puts LABEL (rule 4) before METADATA (rule 5)
        # EXPECTED: LABEL wins
        mapping = _map(["class_source"])
        assert mapping.roles["class_source"] == ColumnRole.LABEL


# ===============================================================
# Heuristic 5: METADATA
# ===============================================================


class TestMetadataHeuristic:
    """Columns containing id/index/source/metadata/split/timestamp/date
    -> METADATA."""

    @pytest.mark.parametrize(
        "name",
        ["id", "index", "source", "metadata", "split", "timestamp", "date"],
    )
    def test_exact_names_map_to_metadata(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.METADATA

    @pytest.mark.parametrize(
        "name",
        ["ID", "Index", "SOURCE", "Metadata", "Split", "TIMESTAMP", "Date"],
    )
    def test_case_insensitive(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.METADATA

    def test_substring_match(self):
        # SCENARIO: 'created_date' contains 'date'
        # WHY: contains-match, not exact
        # EXPECTED: METADATA
        mapping = _map(["created_date"])
        assert mapping.roles["created_date"] == ColumnRole.METADATA


# ===============================================================
# Heuristic 6: UNKNOWN (fallback)
# ===============================================================


class TestUnknownFallback:
    """Columns matching no keyword -> UNKNOWN."""

    @pytest.mark.parametrize(
        "name", ["foo", "random_col", "xyz123", "notes"]
    )
    def test_unrecognized_names_map_to_unknown(self, name):
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.UNKNOWN

    def test_empty_column_name_maps_to_unknown(self):
        # SCENARIO: degenerate empty-string column name
        # WHY: must not crash, must not accidentally match a keyword
        # EXPECTED: UNKNOWN
        mapping = _map([""])
        assert mapping.roles[""] == ColumnRole.UNKNOWN


# ===============================================================
# Determinism + samples population
# ===============================================================


class TestAutoMapColumnsProperties:
    """Purity, determinism, and samples-dict population."""

    def test_deterministic_across_calls(self):
        # SCENARIO: call auto_map_columns twice on identical input
        # WHY: contract requires pure, deterministic mapping
        # EXPECTED: identical roles both times
        columns = ["question", "answer", "passage", "category", "id"]
        dtypes = {
            "question": "string",
            "answer": "string",
            "passage": "string",
            "category": "string",
            "id": "numeric",
        }
        rows = [
            {
                "question": "Q?",
                "answer": "A",
                "passage": "P",
                "category": "geography",
                "id": 1,
            }
        ]
        m1 = auto_map_columns(columns, dtypes, rows)
        m2 = auto_map_columns(columns, dtypes, rows)
        assert m1.roles == m2.roles

    def test_full_realistic_mapping(self):
        # SCENARIO: the tiny.csv fixture's column layout
        # WHY: end-to-end sanity check matching the loader fixtures
        # EXPECTED: matches documented heuristic priority
        columns = ["question", "answer", "passage", "category", "id"]
        dtypes = {
            "question": "string",
            "answer": "string",
            "passage": "string",
            "category": "string",
            "id": "numeric",
        }
        rows = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "passage": "France is in Western Europe.",
                "category": "geography",
                "id": 1,
            }
        ]
        mapping = auto_map_columns(columns, dtypes, rows)
        assert mapping.roles["question"] == ColumnRole.INPUT
        assert mapping.roles["answer"] == ColumnRole.GOLDEN
        assert mapping.roles["passage"] == ColumnRole.CONTEXT
        assert mapping.roles["category"] == ColumnRole.LABEL
        assert mapping.roles["id"] == ColumnRole.METADATA

    def test_samples_populated_from_first_row(self):
        # SCENARIO: rows has 2+ entries
        # WHY: contract says samples come from rows[0]
        # EXPECTED: mapping.samples matches the first row exactly
        columns = ["question", "answer"]
        dtypes = {"question": "string", "answer": "string"}
        rows = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        mapping = auto_map_columns(columns, dtypes, rows)
        assert mapping.samples == {"question": "Q1?", "answer": "A1"}

    def test_empty_rows_does_not_crash(self):
        # SCENARIO: rows list is empty
        # WHY: must handle datasets/schemas with zero preview rows
        # EXPECTED: samples == {}, roles still computed
        columns = ["question", "answer"]
        dtypes = {"question": "string", "answer": "string"}
        mapping = auto_map_columns(columns, dtypes, [])
        assert mapping.samples == {}
        assert mapping.roles["question"] == ColumnRole.INPUT
        assert mapping.roles["answer"] == ColumnRole.GOLDEN

    def test_returns_column_mapping_instance(self):
        # SCENARIO: basic call
        # WHY: type contract
        # EXPECTED: ColumnMapping instance
        mapping = _map(["question"])
        assert isinstance(mapping, ColumnMapping)

    def test_all_input_columns_present_in_roles(self):
        # SCENARIO: mixed set of columns
        # WHY: every input column must get a role, none dropped
        # EXPECTED: roles keys == input columns
        columns = ["question", "answer", "unknown_thing", "id"]
        dtypes = dict.fromkeys(columns, "string")
        mapping = _map(columns, dtypes=dtypes)
        assert set(mapping.roles.keys()) == set(columns)
