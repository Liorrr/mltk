"""Tests for mltk.testdefs — YAML test definitions engine.

Covers: YAML loading, env-var resolution, and the full run_test_suite
dispatcher for every supported assertion key.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mltk.testdefs import load_test_suite, run_test_suite
from mltk.testdefs.schema import TestDef, TestSuiteYaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, content: str) -> Path:
    """Write a YAML string to *path* and return the path."""
    path.write_text(content, encoding="utf-8")
    return path


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    """Write a DataFrame as CSV to *path* and return the path."""
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Load / parse tests
# ---------------------------------------------------------------------------

class TestLoadSuite:
    """Tests for load_test_suite — YAML parsing and validation."""

    def test_load_suite_valid(self, tmp_path: Path) -> None:
        # SCENARIO: A well-formed YAML file with two test entries
        # WHY: Core happy-path — ensure fields map correctly to dataclasses
        # EXPECTED: TestSuiteYaml with correct data_source and two TestDef objects
        csv_path = tmp_path / "data.csv"
        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv_path}
tests:
  - name: Check schema
    assertion: schema
    params:
      expected:
        id: int64
        score: float64

  - name: No nulls
    assertion: no_nulls
""",
        )

        suite = load_test_suite(yaml_path)

        assert isinstance(suite, TestSuiteYaml)
        assert suite.data_source == str(csv_path)
        assert len(suite.tests) == 2

        first = suite.tests[0]
        assert isinstance(first, TestDef)
        assert first.name == "Check schema"
        assert first.assertion == "schema"
        assert first.params["expected"] == {"id": "int64", "score": "float64"}

        second = suite.tests[1]
        assert second.assertion == "no_nulls"
        assert second.params == {}

    def test_load_suite_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # SCENARIO: data_source uses env:VAR_NAME syntax
        # WHY: Verify env-var interpolation resolves at parse time
        # EXPECTED: suite.data_source equals the value of the env variable
        resolved = str(tmp_path / "env_data.csv")
        monkeypatch.setenv("MLTK_TEST_DATA", resolved)

        yaml_path = _write_yaml(
            tmp_path / "suite_env.yaml",
            """\
data_source: env:MLTK_TEST_DATA
tests:
  - name: Row count
    assertion: row_count
    params:
      min_rows: 1
""",
        )

        suite = load_test_suite(yaml_path)
        assert suite.data_source == resolved

    def test_load_suite_missing_env_var_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # SCENARIO: env:VAR references an unset variable
        # WHY: Should give a clear KeyError rather than a cryptic file-not-found
        # EXPECTED: KeyError with the variable name in the message
        monkeypatch.delenv("MLTK_UNSET_VAR", raising=False)
        yaml_path = _write_yaml(
            tmp_path / "suite_bad_env.yaml",
            """\
data_source: env:MLTK_UNSET_VAR
tests: []
""",
        )

        with pytest.raises(KeyError, match="MLTK_UNSET_VAR"):
            load_test_suite(yaml_path)

    def test_load_suite_missing_file_raises(self, tmp_path: Path) -> None:
        # SCENARIO: The YAML file path does not exist
        # WHY: FileNotFoundError with a clear path is more helpful than a Python traceback
        # EXPECTED: FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_test_suite(tmp_path / "nonexistent.yaml")

    def test_load_suite_missing_data_source_raises(self, tmp_path: Path) -> None:
        # SCENARIO: YAML file omits the required 'data_source' key
        # WHY: Schema validation should catch this early with a descriptive error
        # EXPECTED: ValueError mentioning 'data_source'
        yaml_path = _write_yaml(
            tmp_path / "no_source.yaml",
            """\
tests:
  - name: Test
    assertion: no_nulls
""",
        )
        with pytest.raises(ValueError, match="data_source"):
            load_test_suite(yaml_path)

    def test_load_suite_default_name_when_missing(self, tmp_path: Path) -> None:
        # SCENARIO: A test entry has no 'name' field
        # WHY: The runner must still have a label; fall back to "test_<index>"
        # EXPECTED: TestDef.name is "test_0"
        csv_path = tmp_path / "data.csv"
        yaml_path = _write_yaml(
            tmp_path / "no_name.yaml",
            f"""\
data_source: {csv_path}
tests:
  - assertion: no_nulls
""",
        )
        suite = load_test_suite(yaml_path)
        assert suite.tests[0].name == "test_0"


# ---------------------------------------------------------------------------
# run_test_suite dispatcher tests
# ---------------------------------------------------------------------------

class TestRunSuite:
    """Integration tests for run_test_suite — end-to-end YAML → results."""

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    def test_run_schema_pass(self, tmp_path: Path) -> None:
        # SCENARIO: DataFrame matches the declared schema exactly
        # WHY: Verify that the 'schema' key dispatches to assert_schema and passes
        # EXPECTED: Single result with passed=True
        df = pd.DataFrame({"id": pd.array([1, 2, 3], dtype="int64"),
                           "score": pd.array([0.1, 0.5, 0.9], dtype="float64")})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Schema check
    assertion: schema
    params:
      expected:
        id: int64
        score: float64
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    def test_run_schema_fail(self, tmp_path: Path) -> None:
        # SCENARIO: DataFrame is missing a required column declared in expected schema
        # WHY: Confirms failure path propagates correctly through the dispatcher
        # EXPECTED: Single result with passed=False and a message about missing columns
        df = pd.DataFrame({"id": pd.array([1, 2, 3], dtype="int64")})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Schema must have score
    assertion: schema
    params:
      expected:
        id: int64
        score: float64
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is False
        assert "score" in results[0].message

    # ------------------------------------------------------------------
    # no_nulls
    # ------------------------------------------------------------------

    def test_run_no_nulls_pass(self, tmp_path: Path) -> None:
        # SCENARIO: DataFrame has no null values in any column
        # WHY: Verify 'no_nulls' key dispatches correctly and returns pass
        # EXPECTED: passed=True, no null counts in details
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: No nulls anywhere
    assertion: no_nulls
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    def test_run_no_nulls_subset_columns(self, tmp_path: Path) -> None:
        # SCENARIO: Check only specific columns; the unchecked column has nulls
        # WHY: Confirms the 'columns' param is forwarded correctly to assert_no_nulls
        # EXPECTED: passed=True (the checked column 'a' is clean)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, "y", None]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: No nulls in column a only
    assertion: no_nulls
    params:
      columns: [a]
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert results[0].passed is True

    # ------------------------------------------------------------------
    # range
    # ------------------------------------------------------------------

    def test_run_range_pass(self, tmp_path: Path) -> None:
        # SCENARIO: All values in 'score' are within [0.0, 1.0]
        # WHY: Verify 'range' key dispatches to assert_range with column + bounds
        # EXPECTED: passed=True
        df = pd.DataFrame({"score": pd.array([0.0, 0.5, 1.0], dtype="float64")})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Score must be probability
    assertion: range
    params:
      column: score
      min_val: 0.0
      max_val: 1.0
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert results[0].passed is True

    # ------------------------------------------------------------------
    # row_count
    # ------------------------------------------------------------------

    def test_run_row_count_pass(self, tmp_path: Path) -> None:
        # SCENARIO: DataFrame has 5 rows; min_rows=3 is satisfied
        # WHY: Verify 'row_count' key dispatches to assert_row_count
        # EXPECTED: passed=True, details contain actual row count
        df = pd.DataFrame({"x": range(5)})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: At least 3 rows
    assertion: row_count
    params:
      min_rows: 3
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert results[0].passed is True
        assert results[0].details.get("row_count") == 5

    # ------------------------------------------------------------------
    # multiple tests
    # ------------------------------------------------------------------

    def test_run_multiple_tests_all_pass(self, tmp_path: Path) -> None:
        # SCENARIO: Suite declares 4 different assertions; all should pass
        # WHY: Confirms the runner iterates over all TestDefs and returns one
        #      result per test, not short-circuiting on the first pass
        # EXPECTED: 4 results, all passed=True
        df = pd.DataFrame({
            "id": pd.array([1, 2, 3, 4, 5], dtype="int64"),
            "score": pd.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype="float64"),
            "label": pd.array([0, 1, 0, 1, 0], dtype="int64"),
        })
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Schema check
    assertion: schema
    params:
      expected:
        id: int64
        score: float64
        label: int64

  - name: No nulls
    assertion: no_nulls

  - name: Score in range
    assertion: range
    params:
      column: score
      min_val: 0.0
      max_val: 1.0

  - name: At least 5 rows
    assertion: row_count
    params:
      min_rows: 5
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 4
        for r in results:
            assert r.passed is True, f"Expected pass but failed: {r.name} — {r.message}"

    def test_run_multiple_tests_continues_after_fail(self, tmp_path: Path) -> None:
        # SCENARIO: First test fails (wrong schema); second test (no_nulls) should still run
        # WHY: run_test_suite must catch MltkAssertionError and continue, never abort
        # EXPECTED: 2 results; first passed=False, second passed=True
        df = pd.DataFrame({"name": ["alice", "bob"]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Schema fail
    assertion: schema
    params:
      expected:
        id: int64       # column does not exist — should fail

  - name: No nulls pass
    assertion: no_nulls
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 2
        assert results[0].passed is False
        assert results[1].passed is True

    # ------------------------------------------------------------------
    # unique
    # ------------------------------------------------------------------

    def test_run_unique_pass(self, tmp_path: Path) -> None:
        # SCENARIO: Column 'id' has all unique values
        # WHY: Verify 'unique' key with single 'column' param dispatches correctly
        # EXPECTED: passed=True
        df = pd.DataFrame({"id": [10, 20, 30, 40]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: IDs must be unique
    assertion: unique
    params:
      column: id
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert results[0].passed is True

    # ------------------------------------------------------------------
    # unknown assertion key
    # ------------------------------------------------------------------

    def test_run_unknown_assertion_returns_failed_result(self, tmp_path: Path) -> None:
        # SCENARIO: YAML declares assertion key 'does_not_exist'
        # WHY: Should return a failed result with a clear error, not raise an exception
        # EXPECTED: passed=False, message contains the unknown key name
        df = pd.DataFrame({"x": [1, 2, 3]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Unsupported assertion
    assertion: does_not_exist
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert results[0].passed is False
        assert "does_not_exist" in results[0].message

    # ------------------------------------------------------------------
    # column_mean (data statistics category)
    # ------------------------------------------------------------------

    def test_run_column_mean_pass(self, tmp_path: Path) -> None:
        # SCENARIO: Column mean of 'score' falls within declared bounds
        # WHY: Verify 'column_mean' dispatches to assert_column_mean correctly
        # EXPECTED: passed=True, details contain the actual mean
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0]})
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Mean score in range
    assertion: column_mean
    params:
      column: score
      min_val: 20.0
      max_val: 30.0
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    # ------------------------------------------------------------------
    # metric (model category)
    # ------------------------------------------------------------------

    def test_run_metric_accuracy_pass(self, tmp_path: Path) -> None:
        # SCENARIO: DataFrame has y_true and y_pred columns; accuracy is 100%
        # WHY: Verify 'metric' key reads columns from the CSV and passes
        # EXPECTED: passed=True with accuracy = 1.0
        df = pd.DataFrame({
            "y_true": [0, 1, 1, 0, 1],
            "y_pred": [0, 1, 1, 0, 1],
        })
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: Model accuracy
    assertion: metric
    params:
      y_true_col: y_true
      y_pred_col: y_pred
      metric: accuracy
      threshold: 0.9
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    # ------------------------------------------------------------------
    # no_overfitting (model category — scalar params)
    # ------------------------------------------------------------------

    def test_run_no_overfitting_pass(self, tmp_path: Path) -> None:
        # SCENARIO: Train/test scores have a small gap within tolerance
        # WHY: Verify 'no_overfitting' dispatches correctly with scalar params
        # EXPECTED: passed=True (gap of 0.03 is below default max_gap of 0.1)
        df = pd.DataFrame({"x": [1]})  # unused but required by runner
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: No overfitting
    assertion: no_overfitting
    params:
      train_score: 0.95
      test_score: 0.92
      max_gap: 0.1
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    # ------------------------------------------------------------------
    # no_degradation (monitor category)
    # ------------------------------------------------------------------

    def test_run_no_degradation_from_column(self, tmp_path: Path) -> None:
        # SCENARIO: A column tracks metric values over time; no significant decline
        # WHY: Verify 'no_degradation' reads history from a DataFrame column
        # EXPECTED: passed=True (values are stable around 0.90-0.93)
        df = pd.DataFrame({
            "daily_accuracy": [0.90, 0.91, 0.92, 0.91, 0.90, 0.91, 0.93, 0.92],
        })
        csv = _write_csv(tmp_path / "data.csv", df)

        yaml_path = _write_yaml(
            tmp_path / "suite.yaml",
            f"""\
data_source: {csv}
tests:
  - name: No accuracy degradation
    assertion: no_degradation
    params:
      column: daily_accuracy
      window: 4
      max_decline: 0.05
""",
        )
        suite = load_test_suite(yaml_path)
        results = run_test_suite(suite)

        assert len(results) == 1
        assert results[0].passed is True

    # ------------------------------------------------------------------
    # plugin registry — custom assertions via @register_assertion
    # ------------------------------------------------------------------

    def test_run_plugin_assertion_called(self, tmp_path: Path) -> None:
        # SCENARIO: A custom assertion is registered via @register_assertion
        #   and a YAML test def references it by name
        # WHY: Verifies the _dispatch fallback into the plugin registry works
        #   end-to-end: register -> YAML -> dispatch -> result
        # EXPECTED: The custom assertion runs, returns passed=True, and the
        #   sentinel detail confirms it was actually called
        from mltk.core.plugin import _ASSERTION_REGISTRY, register_assertion
        from mltk.core.result import Severity, TestResult

        key = "custom_yaml_test_check"
        _ASSERTION_REGISTRY.pop(key, None)

        call_log: list[dict] = []

        @register_assertion(key)
        def assert_custom_yaml_test_check(df=None, **kwargs):
            call_log.append({"df_shape": df.shape if df is not None else None,
                             "kwargs": kwargs})
            return TestResult(
                name=f"plugin.{key}",
                passed=True,
                severity=Severity.INFO,
                message="Custom plugin assertion passed",
                details={"plugin_called": True, "threshold": kwargs.get("threshold")},
            )

        try:
            df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            csv = _write_csv(tmp_path / "data.csv", df)

            yaml_path = _write_yaml(
                tmp_path / "suite.yaml",
                f"""\
data_source: {csv}
tests:
  - name: Custom plugin check
    assertion: {key}
    params:
      threshold: 0.42
""",
            )
            suite = load_test_suite(yaml_path)
            results = run_test_suite(suite)

            assert len(results) == 1
            assert results[0].passed is True
            assert results[0].details["plugin_called"] is True
            assert results[0].details["threshold"] == 0.42

            # Verify the function was actually called with the DataFrame
            assert len(call_log) == 1
            assert call_log[0]["df_shape"] == (3, 2)
            assert call_log[0]["kwargs"]["threshold"] == 0.42
        finally:
            _ASSERTION_REGISTRY.pop(key, None)

    def test_run_plugin_assertion_without_df_param(self, tmp_path: Path) -> None:
        # SCENARIO: A plugin assertion that does NOT accept a 'df' kwarg
        # WHY: The dispatcher tries df=df first, catches TypeError, retries
        #   with only user params — must handle both signatures gracefully
        # EXPECTED: The assertion still runs and returns a valid result
        from mltk.core.plugin import _ASSERTION_REGISTRY, register_assertion
        from mltk.core.result import Severity, TestResult

        key = "custom_no_df_check"
        _ASSERTION_REGISTRY.pop(key, None)

        @register_assertion(key)
        def assert_no_df(threshold=0.5):
            return TestResult(
                name=f"plugin.{key}",
                passed=threshold < 1.0,
                severity=Severity.INFO,
                message=f"Threshold {threshold} < 1.0",
                details={"no_df_plugin": True},
            )

        try:
            df = pd.DataFrame({"x": [1]})
            csv = _write_csv(tmp_path / "data.csv", df)

            yaml_path = _write_yaml(
                tmp_path / "suite.yaml",
                f"""\
data_source: {csv}
tests:
  - name: Plugin without df
    assertion: {key}
    params:
      threshold: 0.7
""",
            )
            suite = load_test_suite(yaml_path)
            results = run_test_suite(suite)

            assert len(results) == 1
            assert results[0].passed is True
            assert results[0].details["no_df_plugin"] is True
        finally:
            _ASSERTION_REGISTRY.pop(key, None)
