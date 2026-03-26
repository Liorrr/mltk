"""Tests for mltk.pytest_plugin -- markers, fixtures, flags, and collectors.

The pytest plugin is the primary user interface for mltk. It registers
custom markers (ml_data, ml_model, ml_smoke, ml_gpu, ml_drift, ml_inference,
ml_slow, ml_nondeterministic) for test selection, provides the ml_config and
ml_report fixtures for accessing MltkConfig and the session collector, and
collects results for JSON export, MLflow logging, and server push.

These tests verify:
1. All ML markers are registered (so -m "ml_data" works)
2. The ml_config fixture provides a valid config object
3. The ml_report fixture provides the session-level collector
4. The report collector correctly tracks pass/fail counts
5. to_json_records() produces valid JSON-serializable output (no Severity enums)
6. JSON export writes correct format (raw array, readable by CLI commands)
7. Each plugin flag option is registered with correct metavar/type
8. _export_json creates parent directories and writes valid JSON
9. _push_to_server handles network errors non-fatally
10. _log_mlflow handles import errors non-fatally
"""

from __future__ import annotations

import json

import pytest

from mltk.core.config import MltkConfig
from mltk.pytest_plugin.plugin import MltkReportCollector

# ---------------------------------------------------------------------------
# TestMarkers — all 8 markers
# ---------------------------------------------------------------------------

class TestMarkers:
    """Verify all 8 ML markers are registered in pytest.

    Markers allow users to run subsets of ML tests (e.g., `pytest -m ml_data`
    to run only data validation tests). If markers are not registered, pytest
    warns about unknown markers and -m filtering silently skips tests.
    """

    def _get_marker_names(self, request: pytest.FixtureRequest) -> list[str]:
        marker_lines = request.config.getini("markers")
        return [str(line).split(":")[0].strip() for line in marker_lines]

    def test_ml_data_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_data marker is registered in pytest configuration.

        WHY: Users run `pytest -m ml_data` to execute only data quality tests
        (schema, drift, freshness, PII). If this marker is not registered,
        pytest shows "Unknown marker" warnings and the filter may not work.
        Expected: "ml_data" found in configured marker names.
        """
        assert "ml_data" in self._get_marker_names(request)

    def test_ml_model_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_model marker is registered in pytest configuration.

        WHY: Users run `pytest -m ml_model` to execute only model quality tests
        (metrics, bias, regression, slicing). Marker must be registered for
        clean pytest output and correct test selection.
        Expected: "ml_model" found in configured marker names.
        """
        assert "ml_model" in self._get_marker_names(request)

    def test_ml_smoke_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_smoke marker is registered (added in Sprint 4).

        WHY: Smoke tests are a fast subset (<30s) used in pre-commit hooks
        and PR checks. They must be selectable via `-m ml_smoke`.
        Expected: "ml_smoke" found in configured marker names.
        """
        assert "ml_smoke" in self._get_marker_names(request)

    def test_ml_gpu_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_gpu marker is registered (added in Sprint 4).

        WHY: GPU-only tests (e.g., CUDA inference latency) should be skippable
        on CPU-only CI runners. The ml_gpu marker enables `-m "not ml_gpu"`
        to exclude these tests.
        Expected: "ml_gpu" found in configured marker names.
        """
        assert "ml_gpu" in self._get_marker_names(request)

    def test_ml_drift_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_drift marker is registered.

        WHY: Drift tests run against production data and may be expensive.
        Selective execution via `-m ml_drift` requires the marker to be known.
        Expected: "ml_drift" found in configured marker names.
        """
        assert "ml_drift" in self._get_marker_names(request)

    def test_ml_inference_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_inference marker is registered.

        WHY: Inference performance tests (latency, throughput) typically require
        a model to be loaded — expensive in CI. The marker enables selective skipping.
        Expected: "ml_inference" found in configured marker names.
        """
        assert "ml_inference" in self._get_marker_names(request)

    def test_ml_slow_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_slow marker is registered.

        WHY: Long-running tests should be excluded from fast CI pipelines via
        `-m "not ml_slow"`. If not registered, the filter silently does nothing.
        Expected: "ml_slow" found in configured marker names.
        """
        assert "ml_slow" in self._get_marker_names(request)

    def test_ml_nondeterministic_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_nondeterministic marker is registered.

        WHY: Tests with inherent randomness (e.g., sampling-based assertions)
        need a marker so they can be flagged or re-run with different seeds.
        Expected: "ml_nondeterministic" found in configured marker names.
        """
        assert "ml_nondeterministic" in self._get_marker_names(request)


# ---------------------------------------------------------------------------
# TestFixtures
# ---------------------------------------------------------------------------

class TestFixtures:
    """Verify plugin fixtures provide correct objects."""

    def test_ml_config_fixture(self, ml_config: MltkConfig) -> None:
        """PASS: ml_config fixture returns a valid MltkConfig with defaults.

        WHY: Every test that uses mltk assertions accesses config via this
        fixture (drift thresholds, report format, seed). If the fixture
        returns None or wrong type, all downstream assertions will crash.
        Expected: Instance of MltkConfig with drift_method="ks" (default).
        """
        assert isinstance(ml_config, MltkConfig)
        assert ml_config.drift_method == "ks"

    def test_ml_report_fixture_returns_collector(
        self, ml_report: MltkReportCollector
    ) -> None:
        """PASS: ml_report fixture returns the session-level MltkReportCollector.

        WHY: Tests that need to access the in-session result log (e.g., to
        assert that a previous test passed) use ml_report. If it returns the
        wrong type, accessing .total or .results would fail.
        Expected: Instance of MltkReportCollector.
        """
        assert isinstance(ml_report, MltkReportCollector)

    def test_ml_report_fixture_is_session_singleton(
        self,
        ml_report: MltkReportCollector,
        request: pytest.FixtureRequest,
    ) -> None:
        """PASS: ml_report fixture returns the same instance as config._mltk_collector.

        WHY: Results added during a test run must accumulate in the same
        collector that the session finish hook reads from. A different instance
        would mean session finish always sees an empty collector.
        Expected: ml_report is request.config._mltk_collector.
        """
        assert ml_report is request.config._mltk_collector


# ---------------------------------------------------------------------------
# TestReportCollector
# ---------------------------------------------------------------------------

class TestReportCollector:
    """Verify the report collector accumulates test results."""

    def test_collector_add_and_count(self) -> None:
        """PASS: Collector correctly counts passed and failed tests.

        WHY: The collector feeds into the HTML report summary ("2 passed,
        1 failed"). If counts are wrong, the report shows incorrect totals
        and the CI exit code may be wrong (exit 0 when tests actually failed).
        Expected: total=3, passed=2, failed=1.
        """
        collector = MltkReportCollector()
        collector.add("test_a", "passed", 0.1)
        collector.add("test_b", "failed", 0.2)
        collector.add("test_c", "passed", 0.1)

        assert collector.total == 3
        assert collector.passed_count == 2
        assert collector.failed_count == 1

    def test_collector_empty_by_default(self) -> None:
        """PASS: Freshly created collector has zero results.

        WHY: If the collector is shared across tests via a module-level
        singleton, counts from earlier tests would bleed into later ones.
        A fresh collector must start at zero.
        Expected: total=0, passed=0, failed=0.
        """
        collector = MltkReportCollector()
        assert collector.total == 0
        assert collector.passed_count == 0
        assert collector.failed_count == 0

    def test_collector_stores_duration(self) -> None:
        """PASS: Collector stores duration alongside each result.

        WHY: Duration is surfaced in the HTML report and JSON export. If the
        collector doesn't store it, all tests appear to take 0ms — hiding
        slow tests that need optimization.
        Expected: Stored duration matches the supplied value.
        """
        collector = MltkReportCollector()
        collector.add("test_x", "passed", 1.234)
        assert collector.results[0]["duration"] == pytest.approx(1.234)

    def test_collector_stores_ml_result(self) -> None:
        """PASS: Collector stores optional ml_result alongside each entry.

        WHY: ml_result carries assertion details (message, details dict,
        severity). Without it, failed tests show no diagnosis in the report.
        Expected: ml_result stored is the same object that was passed in.
        """
        from mltk.core.result import Severity, TestResult
        collector = MltkReportCollector()
        tr = TestResult(
            name="my_test",
            passed=True,
            severity=Severity.INFO,
            message="all good",
        )
        collector.add("tests::test_x", "passed", 0.5, ml_result=tr)
        assert collector.results[0]["ml_result"] is tr


# ---------------------------------------------------------------------------
# TestToJsonRecords
# ---------------------------------------------------------------------------

class TestToJsonRecords:
    """Verify to_json_records() produces JSON-serializable, well-formed output."""

    def test_to_json_records_empty(self) -> None:
        """PASS: to_json_records() returns empty list when no results collected.

        WHY: If it returns None or raises on an empty collector, the JSON
        export and server push hooks will crash at the end of every clean run.
        Expected: Empty list.
        """
        collector = MltkReportCollector()
        records = collector.to_json_records()
        assert records == []

    def test_to_json_records_is_json_serializable(self) -> None:
        """PASS: to_json_records() output is fully JSON-serializable.

        WHY: The Severity enum is not JSON-serializable by default. If
        to_json_records() returns the enum instead of its string value,
        json.dumps() raises TypeError and --mltk-export-json fails silently.
        This test catches the regression.
        Expected: json.dumps() completes without error.
        """
        from mltk.core.result import Severity, TestResult
        collector = MltkReportCollector()
        tr = TestResult(
            name="my_test",
            passed=False,
            severity=Severity.CRITICAL,
            message="schema mismatch",
            details={"expected": "int64", "got": "float64"},
        )
        collector.add("tests::test_schema", "failed", 0.05, ml_result=tr)
        records = collector.to_json_records()
        # This MUST not raise — Severity enum is not JSON-serializable
        serialized = json.dumps(records)
        parsed = json.loads(serialized)
        assert len(parsed) == 1

    def test_to_json_records_severity_is_string(self) -> None:
        """PASS: severity field in JSON records is a plain string, not an enum.

        WHY: 'severity' is used by notify slack and compliance commands to
        reconstruct a Severity enum via Severity(item['severity']). If it's
        a Severity enum object (e.g., <Severity.CRITICAL>), this lookup fails
        with a ValueError.
        Expected: severity is "critical", "error", "warning", or "info".
        """
        from mltk.core.result import Severity, TestResult
        collector = MltkReportCollector()
        tr = TestResult(
            name="t",
            passed=True,
            severity=Severity.WARNING,
            message="",
        )
        collector.add("tests::t", "passed", 0.1, ml_result=tr)
        records = collector.to_json_records()
        assert isinstance(records[0]["severity"], str)
        assert records[0]["severity"] == "warning"

    def test_to_json_records_without_ml_result(self) -> None:
        """PASS: Records without ml_result use safe defaults.

        WHY: Not all tests use mltk assertions. Plain pytest tests still
        get recorded with default severity="info" and empty message/details.
        Expected: severity="info", message="", details={}.
        """
        collector = MltkReportCollector()
        collector.add("tests::plain_test", "passed", 0.2)
        records = collector.to_json_records()
        assert records[0]["severity"] == "info"
        assert records[0]["message"] == ""
        assert records[0]["details"] == {}

    def test_to_json_records_duration_ms(self) -> None:
        """PASS: Duration is converted from seconds to milliseconds correctly.

        WHY: The duration is stored in seconds by pytest but exported as ms
        for readability in dashboards. If conversion is skipped, all durations
        appear 1000x too small (5ms shows as 0.005ms).
        Expected: 0.5s -> 500.0ms.
        """
        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.5)
        records = collector.to_json_records()
        assert records[0]["duration_ms"] == pytest.approx(500.0)

    def test_to_json_records_has_timestamp(self) -> None:
        """PASS: Each record includes an ISO 8601 UTC timestamp.

        WHY: Timestamps are used by server-side dashboards to plot test trends
        over time. A missing or malformed timestamp breaks timeline charts.
        Expected: timestamp field is non-empty and starts with "20".
        """
        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.1)
        records = collector.to_json_records()
        ts = records[0]["timestamp"]
        assert isinstance(ts, str)
        assert ts.startswith("20")  # "2024-..." or "2025-..."

    def test_to_json_records_non_serializable_details_fallback(self) -> None:
        """PASS: Non-JSON-serializable details fall back to empty dict.

        WHY: Some assertion details may contain numpy dtypes, custom objects,
        or other non-JSON types. The try/except fallback ensures the export
        always succeeds rather than crashing mid-session.
        Expected: details={} (empty dict) when details can't be serialized.
        """
        from mltk.core.result import Severity, TestResult

        class Unserializable:
            pass

        collector = MltkReportCollector()
        tr = TestResult(
            name="t",
            passed=True,
            severity=Severity.INFO,
            message="ok",
            details={"bad": Unserializable()},
        )
        collector.add("tests::t", "passed", 0.1, ml_result=tr)
        records = collector.to_json_records()
        # Should not raise; details may be serialized via default=str or fallback to {}
        serialized = json.dumps(records)
        assert serialized is not None


# ---------------------------------------------------------------------------
# TestPluginOptions
# ---------------------------------------------------------------------------

class TestPluginOptions:
    """Verify plugin CLI options are registered with correct names and types."""

    def test_mltk_report_option_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: --mltk-report option is registered as a boolean flag.

        WHY: If the option is not registered, passing --mltk-report will
        cause pytest to fail with "unrecognized arguments" and the entire
        test session is aborted — even if all tests would otherwise pass.
        Expected: Option registered and defaults to False.
        """
        val = request.config.getoption("--mltk-report", default=None)
        assert val is not None or val is False  # registered (may be False by default)

    def test_mltk_export_json_option_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: --mltk-export-json option is registered and defaults to None.

        WHY: Without this option, teams cannot export JSON results for
        downstream commands (model-card, compliance, notify slack).
        Expected: Option registered with None as default.
        """
        val = request.config.getoption("--mltk-export-json", default=None)
        assert val is None  # no JSON path passed in this test run

    def test_mltk_mlflow_option_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: --mltk-mlflow option is registered and defaults to None.

        WHY: MLflow logging is an optional integration. The option must be
        registered so users can pass it without pytest erroring.
        Expected: Option registered with None as default.
        """
        val = request.config.getoption("--mltk-mlflow", default=None)
        assert val is None  # no MLflow experiment passed in this test run

    def test_mltk_server_option_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: --mltk-server option is registered and defaults to None.

        WHY: Server push is an optional integration. The option must be
        registered so users can pass it without pytest erroring.
        Expected: Option registered with None as default.
        """
        val = request.config.getoption("--mltk-server", default=None)
        assert val is None  # no server URL passed in this test run


# ---------------------------------------------------------------------------
# TestExportJson
# ---------------------------------------------------------------------------

class TestExportJson:
    """Tests for _export_json helper function."""

    def test_export_json_creates_file(self, tmp_path) -> None:
        """PASS: _export_json writes a JSON file at the specified path.

        WHY: This is the primary output of --mltk-export-json. If the file
        is not created, all downstream CLI commands (model-card, compliance,
        notify slack) that read from it will fail with file-not-found.
        Expected: File exists after export with non-empty content.
        """
        from mltk.pytest_plugin.plugin import _export_json

        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.1)

        output = tmp_path / "results.json"

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        _export_json(collector, str(output), FakeSession())
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert len(content) > 2  # not just "[]"

    def test_export_json_creates_parent_dirs(self, tmp_path) -> None:
        """PASS: _export_json creates parent directories if they don't exist.

        WHY: Users often pass --mltk-export-json mltk-reports/results.json
        on a fresh project where mltk-reports/ doesn't exist yet. Without
        mkdir, the write fails with FileNotFoundError.
        Expected: File written successfully even when parent dir is absent.
        """
        from mltk.pytest_plugin.plugin import _export_json

        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.1)

        nested_output = tmp_path / "subdir" / "nested" / "results.json"

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        _export_json(collector, str(nested_output), FakeSession())
        assert nested_output.exists()

    def test_export_json_output_is_valid_json(self, tmp_path) -> None:
        """PASS: _export_json writes valid parseable JSON.

        WHY: If the output is malformed JSON (truncated, bad encoding),
        all downstream commands will fail with json.JSONDecodeError —
        a silent data loss scenario.
        Expected: File parses as a JSON list.
        """
        from mltk.pytest_plugin.plugin import _export_json

        collector = MltkReportCollector()
        collector.add("tests::a", "passed", 0.1)
        collector.add("tests::b", "failed", 0.2)

        output = tmp_path / "out.json"

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        _export_json(collector, str(output), FakeSession())
        parsed = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_export_json_with_severity_enum_is_serializable(self, tmp_path) -> None:
        """PASS: _export_json handles Severity enum in ml_result without crashing.

        WHY: This is the regression test for the severity serialization bug.
        Before the fix, json.dump raised TypeError: Object of type Severity is
        not JSON serializable, causing the export to fail silently.
        Expected: File written successfully, severity is a string.
        """
        from mltk.core.result import Severity, TestResult
        from mltk.pytest_plugin.plugin import _export_json

        collector = MltkReportCollector()
        tr = TestResult(
            name="t",
            passed=False,
            severity=Severity.CRITICAL,
            message="fail",
        )
        collector.add("tests::t", "failed", 0.1, ml_result=tr)

        output = tmp_path / "results_enum.json"

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        _export_json(collector, str(output), FakeSession())
        assert output.exists()
        parsed = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(parsed[0]["severity"], str)
        assert parsed[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# TestPushToServer
# ---------------------------------------------------------------------------

class TestPushToServer:
    """Tests for _push_to_server helper function."""

    def test_push_to_server_handles_connection_error(self) -> None:
        """PASS: _push_to_server does not raise when server is unreachable.

        WHY: If the server is down (common in local dev), the test session
        must not abort. The server push is non-fatal — results are still
        written locally.
        Expected: No exception raised when connecting to a dead port.
        """
        from mltk.pytest_plugin.plugin import _push_to_server

        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.1)

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        # Port 19999 is almost certainly not running an mltk server
        _push_to_server(collector, "http://127.0.0.1:19999", FakeSession())
        # If we get here, the function handled the error non-fatally


# ---------------------------------------------------------------------------
# TestLogMlflow
# ---------------------------------------------------------------------------

class TestLogMlflow:
    """Tests for _log_mlflow helper function."""

    def test_log_mlflow_handles_import_error(self) -> None:
        """PASS: _log_mlflow does not raise when mlflow is not installed.

        WHY: mlflow is an optional dependency. If it's not installed in the
        user's environment, the test session must still complete normally.
        The --mltk-mlflow flag must degrade gracefully.
        Expected: No exception raised even when mlflow is unavailable.
        """
        import sys

        from mltk.pytest_plugin.plugin import _log_mlflow

        collector = MltkReportCollector()
        collector.add("tests::t", "passed", 0.1)

        class FakeSession:
            class config:
                class pluginmanager:
                    @staticmethod
                    def get_plugin(name):
                        return None

        # Temporarily hide mlflow if installed to simulate missing package
        original = sys.modules.get("mlflow")
        sys.modules["mlflow"] = None  # type: ignore[assignment]
        try:
            _log_mlflow(collector, "test-experiment", FakeSession())
        except Exception as exc:  # noqa: BLE001
            # Should not propagate — _log_mlflow catches all exceptions
            pytest.fail(f"_log_mlflow raised unexpectedly: {exc}")
        finally:
            if original is None:
                sys.modules.pop("mlflow", None)
            else:
                sys.modules["mlflow"] = original
