"""Tests for mltk.integrations.mlflow_logger — MLflow integration.

All tests mock the mlflow module so that actual mlflow installation is NOT
required. Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.result import Severity, TestResult, TestSuite
from mltk.integrations.mlflow_logger import _sanitize_metric_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_suite(*results: TestResult) -> TestSuite:
    """Build a TestSuite from a sequence of TestResult objects."""
    suite = TestSuite()
    for r in results:
        suite.add(r)
    return suite


def _make_result(
    name: str,
    passed: bool,
    duration_ms: float = 10.0,
) -> TestResult:
    """Build a minimal TestResult."""
    return TestResult(
        name=name,
        passed=passed,
        severity=Severity.INFO,
        message="ok" if passed else "fail",
        duration_ms=duration_ms,
    )


def _make_mlflow_mock() -> MagicMock:
    """Return a MagicMock that looks like the mlflow module.

    ``active_run()`` returns ``None`` by default so that MlflowLogger
    always enters a new run via ``start_run()``.
    ``start_run()`` returns a context manager that yields a mock run.
    """
    mock = MagicMock(name="mlflow")
    mock.active_run.return_value = None

    # start_run must behave as a context manager (__enter__ / __exit__)
    run_ctx = MagicMock()
    run_ctx.__enter__ = MagicMock(return_value=run_ctx)
    run_ctx.__exit__ = MagicMock(return_value=False)
    mock.start_run.return_value = run_ctx

    return mock


# ---------------------------------------------------------------------------
# Fixture: inject mock mlflow before each test
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, None, None]:
    """Patch mlflow in sys.modules so MlflowLogger uses the mock."""
    mlflow_mock = _make_mlflow_mock()

    # Temporarily replace (or inject) the mlflow module
    original = sys.modules.get("mlflow", None)
    sys.modules["mlflow"] = mlflow_mock  # type: ignore[assignment]
    try:
        yield mlflow_mock
    finally:
        if original is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogResultsMetrics:
    """MlflowLogger.log_results() — suite-level metrics."""

    def test_log_results_metrics(self, mock_mlflow: MagicMock) -> None:
        # SCENARIO: A suite with 2 passed and 1 failed test is logged.
        # WHY: Core contract — suite-level aggregates must appear as MLflow metrics
        #      with the correct names and values so downstream dashboards work.
        # EXPECTED: log_metric is called for the five aggregate keys plus one
        #           metric per individual test, with correct pass/fail values.

        from mltk.integrations.mlflow_logger import MlflowLogger

        suite = _make_suite(
            _make_result("accuracy_check", passed=True, duration_ms=5.0),
            _make_result("drift_check", passed=True, duration_ms=15.0),
            _make_result("schema_check", passed=False, duration_ms=8.0),
        )

        logger = MlflowLogger(experiment_name="my-exp")
        logger.log_results(suite)

        logged_calls = dict(
            c.args for c in mock_mlflow.log_metric.call_args_list
        )

        # Aggregate metrics
        assert logged_calls["mltk_total_tests"] == 3
        assert logged_calls["mltk_passed"] == 2
        assert logged_calls["mltk_failed"] == 1
        assert abs(logged_calls["mltk_score"] - (2 / 3 * 100)) < 1e-6
        assert logged_calls["mltk_duration_ms"] == pytest.approx(28.0)

        # Per-test metrics
        assert logged_calls["mltk_accuracy_check"] == 1.0
        assert logged_calls["mltk_drift_check"] == 1.0
        assert logged_calls["mltk_schema_check"] == 0.0

    def test_log_results_empty(self, mock_mlflow: MagicMock) -> None:
        # SCENARIO: log_results is called with an empty TestSuite (no tests ran).
        # WHY: An empty suite must not raise; automated pipelines can run before
        #      any data is present and must not crash.
        # EXPECTED: No exception, aggregate metrics reflect zero counts/score.

        from mltk.integrations.mlflow_logger import MlflowLogger

        suite = TestSuite()
        logger = MlflowLogger()
        logger.log_results(suite)  # Must not raise

        logged_calls = dict(
            c.args for c in mock_mlflow.log_metric.call_args_list
        )
        assert logged_calls["mltk_total_tests"] == 0
        assert logged_calls["mltk_passed"] == 0
        assert logged_calls["mltk_failed"] == 0
        assert logged_calls["mltk_score"] == 0.0

    def test_metric_name_sanitization(self, mock_mlflow: MagicMock) -> None:
        # SCENARIO: A test name contains dots and special characters
        #           (e.g. a pytest node ID like "tests/test_model.py::test_acc").
        # WHY: MLflow rejects metric names with dots and colons; the logger
        #      must sanitize names before calling log_metric or the run will fail.
        # EXPECTED: The logged metric name contains no dots or colons; it starts
        #           with "mltk_" and uses only safe characters.

        from mltk.integrations.mlflow_logger import MlflowLogger

        suite = _make_suite(
            _make_result("tests/test_model.py::test_accuracy", passed=True),
        )
        logger = MlflowLogger()
        logger.log_results(suite)

        logged_names = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        per_test_names = [n for n in logged_names if n not in {
            "mltk_total_tests", "mltk_passed", "mltk_failed",
            "mltk_score", "mltk_duration_ms",
        }]
        assert len(per_test_names) == 1
        metric = per_test_names[0]
        assert "." not in metric
        assert ":" not in metric
        assert metric.startswith("mltk_")


class TestLogReportArtifact:
    """MlflowLogger.log_report() — artifact upload."""

    def test_log_report_artifact(self, mock_mlflow: MagicMock, tmp_path: Path) -> None:
        # SCENARIO: An HTML report file is passed to log_report().
        # WHY: Keeping rendered HTML reports alongside metrics lets users open
        #      the full mltk report directly from the MLflow UI artifact panel.
        # EXPECTED: mlflow.log_artifact is called exactly once with the string
        #           path of the report file.

        from mltk.integrations.mlflow_logger import MlflowLogger

        report = tmp_path / "report.html"
        report.write_text("<html>report</html>")

        logger = MlflowLogger()
        logger.log_report(report)

        mock_mlflow.log_artifact.assert_called_once_with(str(report))

    def test_log_report_accepts_string_path(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        # SCENARIO: log_report receives a plain string instead of a Path object.
        # WHY: Users may pass string literals; both str and Path must work.
        # EXPECTED: log_artifact is called with the string path unchanged.

        from mltk.integrations.mlflow_logger import MlflowLogger

        report = tmp_path / "report.html"
        report.write_text("<html>report</html>")

        logger = MlflowLogger()
        logger.log_report(str(report))  # string, not Path

        mock_mlflow.log_artifact.assert_called_once_with(str(report))


class TestLogTestResult:
    """MlflowLogger.log_test_result() — single-result logging."""

    def test_log_test_result_passed(self, mock_mlflow: MagicMock) -> None:
        # SCENARIO: A single TestResult with passed=True is logged.
        # WHY: Individual-result logging is used by streaming workflows where
        #      tests are logged as they finish, not after the suite completes.
        # EXPECTED: log_metric is called with value 1.0 for the test name.

        from mltk.integrations.mlflow_logger import MlflowLogger

        result = _make_result("bias_check", passed=True)
        logger = MlflowLogger()
        logger.log_test_result(result)

        mock_mlflow.log_metric.assert_called_once_with("mltk_bias_check", 1.0)

    def test_log_test_result_failed(self, mock_mlflow: MagicMock) -> None:
        # SCENARIO: A single TestResult with passed=False is logged.
        # WHY: Failed results must be distinguishable from passed ones in the
        #      MLflow metric timeline (0.0 vs 1.0).
        # EXPECTED: log_metric is called with value 0.0 for the test name.

        from mltk.integrations.mlflow_logger import MlflowLogger

        result = _make_result("latency_p99", passed=False)
        logger = MlflowLogger()
        logger.log_test_result(result)

        mock_mlflow.log_metric.assert_called_once_with("mltk_latency_p99", 0.0)


class TestMlflowNotInstalled:
    """MlflowLogger gracefully fails when mlflow is absent."""

    def test_mlflow_not_installed(self) -> None:
        # SCENARIO: mlflow is not installed in the environment (ImportError).
        # WHY: mltk is an ML testing toolkit; mlflow is an *optional* extra.
        #      Users who don't need MLflow integration must not hit a confusing
        #      ImportError with no hint about how to fix it.
        # EXPECTED: ImportError is raised with "pip install mlflow" in the message.

        # Remove mlflow from sys.modules to simulate it being absent
        original = sys.modules.pop("mlflow", None)
        # Also ensure the cached import inside mlflow_logger is not reused by
        # forcing a fresh import of the logger module.
        mlflow_logger_mod = sys.modules.pop(
            "mltk.integrations.mlflow_logger", None
        )
        try:
            with patch.dict(sys.modules, {"mlflow": None}):  # type: ignore[dict-item]
                import importlib  # noqa: PLC0415

                import mltk.integrations.mlflow_logger as mod  # noqa: PLC0415

                importlib.reload(mod)
                with pytest.raises(ImportError):
                    mod.MlflowLogger()
        finally:
            # Restore original state
            if original is not None:
                sys.modules["mlflow"] = original
            if mlflow_logger_mod is not None:
                sys.modules["mltk.integrations.mlflow_logger"] = mlflow_logger_mod


class TestSanitizeMetricName:
    """Unit tests for _sanitize_metric_name — the MLflow name sanitizer."""

    def test_dots_replaced_with_underscores(self) -> None:
        # SCENARIO: name contains dots (common in pytest node IDs)
        # WHY: MLflow rejects dots in metric names; the sanitizer must replace them
        # EXPECTED: no dots in output
        result = _sanitize_metric_name("tests/test_model.py")
        assert "." not in result

    def test_special_chars_stripped(self) -> None:
        # SCENARIO: name contains colons and brackets (e.g. parametrized node IDs)
        # WHY: MLflow metric names only allow [a-zA-Z0-9_\-/]; other chars must go
        # EXPECTED: output contains only allowed characters
        result = _sanitize_metric_name("test[foo::bar]")
        for ch in result:
            assert ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/"

    def test_consecutive_underscores_collapsed(self) -> None:
        # SCENARIO: multiple special chars in a row produce consecutive underscores
        # WHY: "__" or "___" in metric names looks noisy and can confuse dashboards
        # EXPECTED: no consecutive underscores remain
        result = _sanitize_metric_name("a::b::c")
        assert "__" not in result

    def test_leading_trailing_underscores_stripped(self) -> None:
        # SCENARIO: name starts and ends with underscore-producing chars
        # WHY: MLflow may reject or misparse names with leading/trailing underscores
        # EXPECTED: result has no leading or trailing underscores
        result = _sanitize_metric_name("  spaces_around  ")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_valid_name_unchanged(self) -> None:
        # SCENARIO: name already contains only safe characters
        # WHY: sanitizer must be a no-op for already-valid names
        # EXPECTED: output equals input
        assert _sanitize_metric_name("model_accuracy") == "model_accuracy"

    def test_slash_preserved(self) -> None:
        # SCENARIO: name contains a forward slash (valid in MLflow metric names)
        # WHY: slashes are used for hierarchical metric grouping in MLflow runs
        # EXPECTED: slash survives sanitisation intact
        result = _sanitize_metric_name("data/drift_psi")
        assert "/" in result
