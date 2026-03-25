"""Tests for mltk.data.validation — format, set membership, conflicting labels.

Validation assertions target semantic correctness of data values:
- datetime_format: dates must parse with the agreed format string
- values_in_set: categoricals must use the agreed vocabulary
- no_conflicting_labels: identical feature rows must not map to different labels

These bugs are silent — a model will train without crashing while learning
corrupt patterns.
"""

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.validation import (
    assert_datetime_format,
    assert_no_conflicting_labels,
    assert_values_in_set,
)

# ---------------------------------------------------------------------------
# assert_datetime_format
# ---------------------------------------------------------------------------


class TestAssertDatetimeFormat:
    """Tests for assert_datetime_format — date string format validation."""

    def test_datetime_format_valid(self) -> None:
        """SCENARIO: All rows have ISO date strings in the expected format.
        WHY: Happy path — clean date column from a well-behaved data source.
        EXPECTED: pass=True, invalid_count == 0.
        """
        df = pd.DataFrame({"date": ["2024-01-01", "2024-06-15", "2025-12-31"]})
        result = assert_datetime_format(df, "date")
        assert result.passed is True
        assert result.details["invalid_count"] == 0

    def test_datetime_format_invalid(self) -> None:
        """SCENARIO: Some rows use DD/MM/YYYY instead of the expected YYYY-MM-DD.
        WHY: Mixed date formats in a production feed — catches a real pipeline bug.
        EXPECTED: MltkAssertionError raised, invalid_count > 0 in details.
        """
        df = pd.DataFrame({"date": ["2024-01-01", "15/06/2024", "not-a-date"]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_datetime_format(df, "date")
        result = exc.value.result
        assert result.details["invalid_count"] == 2

    def test_datetime_custom_format(self) -> None:
        """SCENARIO: Event log uses MM/DD/YYYY format; caller specifies that format.
        WHY: Default format is YYYY-MM-DD but real-world data varies widely.
        EXPECTED: pass=True when all values match the custom format string.
        """
        df = pd.DataFrame({"event_date": ["01/15/2024", "06/30/2024", "12/01/2025"]})
        result = assert_datetime_format(df, "event_date", fmt="%m/%d/%Y")
        assert result.passed is True

    def test_datetime_format_partial_invalid(self) -> None:
        """SCENARIO: One good date, one wrong-format, one garbage string.
        WHY: Partial invalidity should still be reported with exact count.
        EXPECTED: invalid_count == 2 (both bad rows), total_rows == 3.
        """
        df = pd.DataFrame({"ts": ["2024-03-01", "03-01-2024", "INVALID"]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_datetime_format(df, "ts")
        result = exc.value.result
        assert result.details["total_rows"] == 3
        assert result.details["invalid_count"] == 2

    def test_datetime_format_stored_in_details(self) -> None:
        """SCENARIO: Caller wants to know which format was checked after a failure.
        WHY: Audit trail — the TestResult should capture the format used.
        EXPECTED: details["format"] matches the format argument.
        """
        df = pd.DataFrame({"d": ["2024-01-01"]})
        result = assert_datetime_format(df, "d", fmt="%Y-%m-%d")
        assert result.details["fmt"] == "%Y-%m-%d"


# ---------------------------------------------------------------------------
# assert_values_in_set
# ---------------------------------------------------------------------------


class TestAssertValuesInSet:
    """Tests for assert_values_in_set — categorical vocabulary enforcement."""

    def test_values_in_set_pass(self) -> None:
        """SCENARIO: Status column only contains the three agreed states.
        WHY: Happy path — production data uses the exact agreed vocabulary.
        EXPECTED: pass=True, invalid_count == 0.
        """
        df = pd.DataFrame({
            "status": ["active", "inactive", "pending", "active", "pending"]
        })
        allowed = {"active", "inactive", "pending"}
        result = assert_values_in_set(df, "status", allowed_values=allowed)
        assert result.passed is True
        assert result.details["invalid_count"] == 0

    def test_values_in_set_fail(self) -> None:
        """SCENARIO: New country code 'AU' appeared in data but is not in the allowed set.
        WHY: A new upstream market was added without updating the schema contract.
        EXPECTED: MltkAssertionError raised, 'AU' in invalid_samples.
        """
        df = pd.DataFrame({"country": ["US", "GB", "AU", "DE", "AU"]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_values_in_set(df, "country", allowed_values=["US", "GB", "DE"])
        result = exc.value.result
        assert result.details["invalid_count"] == 2
        assert "AU" in result.details["invalid_samples"]

    def test_values_in_set_empty_df(self) -> None:
        """SCENARIO: Upstream query returned zero rows — empty DataFrame.
        WHY: Empty input should pass (zero violations), not raise a confusing error.
        EXPECTED: pass=True with invalid_count == 0.
        """
        df = pd.DataFrame({"category": []})
        result = assert_values_in_set(df, "category", allowed_values={"A", "B", "C"})
        assert result.passed is True
        assert result.details["invalid_count"] == 0

    def test_values_in_set_accepts_list(self) -> None:
        """SCENARIO: Caller passes a list instead of a set for allowed_values.
        WHY: The API should accept either type — callers shouldn't need to convert.
        EXPECTED: pass=True when all values match, using list input.
        """
        df = pd.DataFrame({"tier": ["free", "pro", "team", "free"]})
        result = assert_values_in_set(df, "tier", allowed_values=["free", "pro", "team"])
        assert result.passed is True

    def test_values_in_set_reports_allowed_count(self) -> None:
        """SCENARIO: Verify details capture allowed set size for traceability.
        WHY: Audit trail — how large was the allowed vocabulary at check time.
        EXPECTED: details["allowed_count"] matches size of allowed_values.
        """
        allowed = {"cat", "dog", "bird", "fish"}
        df = pd.DataFrame({"animal": ["cat", "dog"]})
        result = assert_values_in_set(df, "animal", allowed_values=allowed)
        assert result.details["allowed_count"] == 4


# ---------------------------------------------------------------------------
# assert_no_conflicting_labels
# ---------------------------------------------------------------------------


class TestAssertNoConflictingLabels:
    """Tests for assert_no_conflicting_labels — label consistency validation."""

    def test_no_conflicting_labels_clean(self) -> None:
        """SCENARIO: Each unique (age, region) combination maps to exactly one label.
        WHY: Happy path — a clean training dataset has consistent labeling.
        EXPECTED: pass=True, conflict_count == 0.
        """
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "region": ["north", "south", "north", "south"],
            "churn": [0, 1, 0, 1],
        })
        result = assert_no_conflicting_labels(df, feature_cols=["age", "region"], label_col="churn")
        assert result.passed is True
        assert result.details["conflict_count"] == 0

    def test_conflicting_labels_detected(self) -> None:
        """SCENARIO: Two rows have identical (age=30, region='north') but labels 0 and 1.
        WHY: This is an annotation error — same input, different output. Model cannot learn this.
        EXPECTED: MltkAssertionError raised, conflict_count == 1 in details.
        """
        df = pd.DataFrame({
            "age": [30, 30, 40],
            "region": ["north", "north", "south"],
            "churn": [0, 1, 0],  # Row 0 and 1 have identical features but different labels
        })
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_conflicting_labels(df, feature_cols=["age", "region"], label_col="churn")
        result = exc.value.result
        assert result.details["conflict_count"] == 1

    def test_conflicting_labels_multiple_conflicts(self) -> None:
        """SCENARIO: Multiple feature groups have conflicting labels — systematic annotation bug.
        WHY: Batch labeling errors often affect multiple groups at once.
        EXPECTED: MltkAssertionError raised, conflict_count == 2.
        """
        df = pd.DataFrame({
            "x": [1, 1, 2, 2, 3],
            "label": [0, 1, 0, 1, 0],  # Groups x=1 and x=2 both conflict
        })
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_conflicting_labels(df, feature_cols=["x"], label_col="label")
        result = exc.value.result
        assert result.details["conflict_count"] == 2

    def test_no_conflicting_labels_empty_df(self) -> None:
        """SCENARIO: Empty DataFrame passed in before any data is loaded.
        WHY: Edge case — should pass without crashing (zero groups, zero conflicts).
        EXPECTED: pass=True, conflict_count == 0.
        """
        df = pd.DataFrame({"feat": [], "label": []})
        result = assert_no_conflicting_labels(df, feature_cols=["feat"], label_col="label")
        assert result.passed is True
        assert result.details["conflict_count"] == 0

    def test_no_conflicting_labels_total_groups_in_details(self) -> None:
        """SCENARIO: Verify details capture total group count for audit.
        WHY: Knowing how many unique feature groups were checked is part of the audit trail.
        EXPECTED: total_groups == 3 for 3 distinct feature values.
        """
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C", "C"],
            "label": [0, 0, 1, 1, 0, 0],  # No conflicts
        })
        result = assert_no_conflicting_labels(df, feature_cols=["category"], label_col="label")
        assert result.passed is True
        assert result.details["total_groups"] == 3
