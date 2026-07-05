"""Tests for mltk.importer.mapping -- deterministic column auto-mapping.

Exercises the token-based heuristic priority table (heuristics v2):
METADATA-suffix (last token) > INPUT (string-only) > CONTEXT > GOLDEN
(string, or numeric when the column name IS the keyword) > LABEL >
METADATA (any token) > UNKNOWN. Matching is WHOLE-TOKEN, never a raw
substring check -- see ``mltk.importer.mapping._tokenize``.
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

    def test_compound_name_with_input_token_matches(self):
        # SCENARIO: column name contains the keyword as one of several
        #   underscore-separated tokens, not equal to the whole name
        # WHY: heuristics v2 tokenizes on '_' boundaries, so 'question'
        #   is matched as a whole token even inside a longer compound
        #   name (NOT a raw substring check -- see TestUnknownFallback
        #   for cases where a keyword appears as a substring but not a
        #   whole token, e.g. 'candidate' containing 'id')
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

    def test_compound_name_with_context_token_matches(self):
        # SCENARIO: 'retrieved_passages' tokenizes to ['retrieved',
        #   'passages'] -- 'retrieved' is an exact CONTEXT token even
        #   though 'passages' (plural) is not
        # WHY: whole-token match, not exact-column-name match
        # EXPECTED: CONTEXT
        mapping = _map(["retrieved_passages"])
        assert mapping.roles["retrieved_passages"] == ColumnRole.CONTEXT

    def test_metadata_suffix_wins_over_context_when_both_match(self):
        # SCENARIO: 'document_id' matches CONTEXT ('document' token) and
        #   the METADATA-suffix rule (last token 'id')
        # WHY: CHANGED under heuristics v2 -- the METADATA-suffix rule
        #   (rule a) is checked first, against the LAST token, before any
        #   other rule runs for any column. Under v1 this column was
        #   CONTEXT (substring match on 'document' won); v2 intentionally
        #   flips this so 'document_id' (and similarly 'question_id',
        #   'query_id') is treated as a bookkeeping id column, not the
        #   semantic field its non-suffix token suggests
        # EXPECTED: METADATA wins
        mapping = _map(["document_id"])
        assert mapping.roles["document_id"] == ColumnRole.METADATA


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

    def test_compound_name_with_metadata_suffix_matches(self):
        # SCENARIO: 'created_date' tokenizes to ['created', 'date'] --
        #   the LAST token 'date' hits the metadata-suffix rule (rule a)
        # WHY: whole-token match, not exact-column-name match
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

    @pytest.mark.parametrize("name", ["confidence", "video", "candidate", "valid"])
    def test_words_containing_id_as_substring_do_not_false_positive(self, name):
        # SCENARIO: 'confidence'/'video'/'candidate'/'valid' each contain
        #   the LETTERS 'id' as a raw substring but are single whole
        #   tokens that are none of the recognized keywords
        # WHY: heuristics v2 matches whole tokens (set membership), never
        #   a substring scan -- this is the regression test for the P2
        #   finding where the old substring-based METADATA rule would
        #   have misclassified these as METADATA via a bogus 'id' hit
        # EXPECTED: UNKNOWN
        mapping = _map([name])
        assert mapping.roles[name] == ColumnRole.UNKNOWN


# ===============================================================
# Heuristics v2 regressions -- METADATA-suffix rule, INPUT dtype gate,
# single-INPUT invariant, numeric-answer GOLDEN exception
# ===============================================================


class TestMetadataSuffixRule:
    """Rule (a): LAST token in the id/idx/index/uid/uuid/timestamp/date/
    split suffix set -> METADATA, checked before every other rule."""

    @pytest.mark.parametrize("name", ["question_id", "query_id"])
    def test_input_keyword_plus_id_suffix_is_metadata_not_input(self, name):
        # SCENARIO: P1 bug -- 'question_id'/'query_id' previously matched
        #   INPUT via substring, silently stealing the role from the real
        #   prompt column and dropping it to metadata
        # WHY: rule (a) fires on the last token 'id' before the INPUT
        #   rule is ever considered for this column
        # EXPECTED: METADATA
        mapping = _map([name], dtypes={name: "string"})
        assert mapping.roles[name] == ColumnRole.METADATA


class TestInputRequiresStringDtype:
    """Rule (b): INPUT requires dtype == 'string', not just a name match."""

    def test_numeric_input_tokens_is_not_input(self):
        # SCENARIO: P1 bug -- a numeric 'input_tokens' column (token
        #   count, not prompt text) previously matched INPUT via
        #   substring on 'input'
        # WHY: rule (b) now requires dtype == 'string'; numeric
        #   'input_tokens' matches no other rule either (no context/
        #   golden/label/metadata token) so it falls through to UNKNOWN
        # EXPECTED: UNKNOWN, never INPUT
        mapping = _map(["input_tokens"], dtypes={"input_tokens": "numeric"})
        assert mapping.roles["input_tokens"] == ColumnRole.UNKNOWN


class TestSingleInputInvariant:
    """Only the first INPUT candidate (by column order) keeps INPUT."""

    def test_multiple_input_candidates_first_wins_rest_unknown(self):
        # SCENARIO: P1 bug -- with two string columns both matching an
        #   INPUT keyword ('question' and 'prompt'), to_eval_dataset()
        #   used to pick the first in COLUMN-DICT order arbitrarily and
        #   silently drop the other's data
        # WHY: the single-INPUT invariant demotes every candidate after
        #   the first (in original column order) to UNKNOWN, so it is
        #   surfaced via preview()/validate() and passed through to
        #   metadata instead of silently vanishing
        # EXPECTED: 'question' (first) -> INPUT, 'prompt' (second) ->
        #   UNKNOWN
        columns = ["question", "prompt", "answer"]
        dtypes = {"question": "string", "prompt": "string", "answer": "string"}
        mapping = _map(columns, dtypes=dtypes)
        assert mapping.roles["question"] == ColumnRole.INPUT
        assert mapping.roles["prompt"] == ColumnRole.UNKNOWN
        assert mapping.roles["answer"] == ColumnRole.GOLDEN

    def test_reversed_order_still_picks_first_column(self):
        # SCENARIO: same ambiguity as above, but the INPUT-keyword
        #   column that should win is now second in column order
        # WHY: "first" must mean original column order, not name/dict
        #   ordering, or any dtype-based tiebreak
        # EXPECTED: 'prompt' (first) -> INPUT, 'question' (second) ->
        #   UNKNOWN
        columns = ["prompt", "question"]
        dtypes = {"prompt": "string", "question": "string"}
        mapping = _map(columns, dtypes=dtypes)
        assert mapping.roles["prompt"] == ColumnRole.INPUT
        assert mapping.roles["question"] == ColumnRole.UNKNOWN


class TestNumericAnswerGoldenException:
    """Rule (d): a numeric column named EXACTLY a golden keyword is
    still GOLDEN; a numeric compound name (e.g. 'output_tokens') is not."""

    def test_numeric_answer_column_maps_to_golden(self):
        # SCENARIO: P1 bug -- a math-QA dataset where pandas infers a
        #   numeric dtype for the 'answer' column (e.g. all-integer
        #   answers); the old dtype=='string' gate excluded it entirely,
        #   leaving target=None for every sample
        # WHY: rule (d)'s numeric exception fires only when the WHOLE
        #   lowered column name equals a golden keyword exactly
        # EXPECTED: GOLDEN
        mapping = _map(["answer"], dtypes={"answer": "numeric"})
        assert mapping.roles["answer"] == ColumnRole.GOLDEN

    def test_numeric_compound_golden_name_does_not_map_to_golden(self):
        # SCENARIO: 'output_tokens' is numeric and contains the GOLDEN
        #   token 'output', but the column name as a whole isn't the
        #   literal keyword 'output'
        # WHY: the numeric exception is deliberately narrow -- it must
        #   not swallow every numeric column whose name happens to
        #   contain a golden keyword as one of several tokens
        # EXPECTED: UNKNOWN, not GOLDEN
        mapping = _map(["output_tokens"], dtypes={"output_tokens": "numeric"})
        assert mapping.roles["output_tokens"] == ColumnRole.UNKNOWN


class TestTextFallbackNotBlockedByUnrelatedTextColumn:
    """A numeric column that merely contains the 'text' token must not
    prevent a lone string 'text' column from becoming INPUT."""

    def test_numeric_text_length_does_not_block_lone_text_input(self):
        # SCENARIO: 'text_length' (numeric, e.g. a precomputed character
        #   count) sits alongside a lone string 'text' column
        # WHY: the text-fallback candidate list requires dtype ==
        #   'string'; 'text_length' is numeric so it's excluded from
        #   consideration, leaving 'text' as the sole candidate
        # EXPECTED: 'text' -> INPUT, 'text_length' -> UNKNOWN
        columns = ["text", "text_length"]
        dtypes = {"text": "string", "text_length": "numeric"}
        mapping = _map(columns, dtypes=dtypes)
        assert mapping.roles["text"] == ColumnRole.INPUT
        assert mapping.roles["text_length"] == ColumnRole.UNKNOWN


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
