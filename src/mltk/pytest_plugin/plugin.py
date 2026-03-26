"""pytest plugin for mltk. Auto-registered via pyproject.toml entry-points.

Provides ML-specific markers, fixtures, and the --mltk-report flag
for generating test summaries, plus --mltk-export-json for JSON export.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from mltk.core.config import MltkConfig
from mltk.core.result import TestResult


@dataclass
class MltkReportCollector:
    """Collects test results during a pytest session."""

    results: list[dict[str, object]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add(
        self, nodeid: str, outcome: str, duration: float,
        ml_result: TestResult | None = None,
    ) -> None:
        """Record a test result.

        Args:
            nodeid: pytest node ID (e.g., "tests/test_data.py::test_schema").
            outcome: "passed" or "failed".
            duration: Test duration in seconds.
            ml_result: Optional mltk TestResult with assertion details.
        """
        self.results.append({
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": duration,
            "ml_result": ml_result,
        })

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r["outcome"] == "passed")

    @property
    def failed_count(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.results if r["outcome"] == "failed")

    @property
    def total(self) -> int:
        """Total number of recorded test results."""
        return len(self.results)

    def to_json_records(self) -> list[dict[str, object]]:
        """Serialize collected results to a list of JSON-serializable dicts.

        Each record contains:
        - name: test node ID
        - passed: bool
        - severity: "error" | "warning" | "info" (from ml_result or default "info")
        - message: assertion message or empty string
        - details: assertion details dict or empty dict
        - duration_ms: test duration in milliseconds (float)
        - timestamp: ISO 8601 UTC timestamp string

        Returns:
            List of dicts suitable for json.dumps().
        """
        records = []
        for r in self.results:
            ml_result: TestResult | None = r.get("ml_result")  # type: ignore[assignment]
            duration_s = float(r.get("duration", 0.0))  # type: ignore[arg-type]

            severity = "info"
            message = ""
            details: dict[str, object] = {}

            if ml_result is not None:
                message = getattr(ml_result, "message", "") or ""
                raw_details = getattr(ml_result, "details", {}) or {}
                # Ensure details is JSON-serializable (convert non-basic types)
                try:
                    details = json.loads(json.dumps(raw_details, default=str))
                except Exception:  # noqa: BLE001
                    details = {}
                raw_severity = getattr(ml_result, "severity", "info") or "info"
                # Extract string value from Severity enum if needed
                if hasattr(raw_severity, "value"):
                    severity = raw_severity.value
                else:
                    severity = str(raw_severity)

            records.append({
                "name": str(r["nodeid"]),
                "passed": r["outcome"] == "passed",
                "severity": severity,
                "message": message,
                "details": details,
                "duration_ms": round(duration_s * 1000, 3),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        return records


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add --mltk-report, --mltk-export-json, --mltk-mlflow, and --mltk-server options."""
    parser.addoption(
        "--mltk-report",
        action="store_true",
        default=False,
        help="Generate mltk test summary report at end of session",
    )
    parser.addoption(
        "--mltk-export-json",
        action="store",
        default=None,
        metavar="PATH",
        help="Export mltk test results as JSON to the specified file path",
    )
    parser.addoption(
        "--mltk-mlflow",
        action="store",
        default=None,
        metavar="EXPERIMENT",
        help=(
            "Log mltk test results to MLflow under the given experiment name. "
            "Set MLFLOW_TRACKING_URI to point at a remote server, or omit for "
            "local ./mlruns storage. Requires: pip install mlflow"
        ),
    )
    parser.addoption(
        "--mltk-server",
        action="store",
        default=None,
        metavar="URL",
        help=(
            "Push test results to a running mltk server after the session "
            "(e.g., http://localhost:8080). Requires: pip install mltk[server]"
        ),
    )


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register mltk markers and initialize report collector."""
    # Register all ML markers
    markers = [
        "ml_data: data quality tests (schema, distribution, drift, PII)",
        "ml_model: model quality tests (metrics, bias, regression, slicing)",
        "ml_drift: drift detection tests",
        "ml_inference: inference performance tests (latency, throughput)",
        "ml_slow: long-running tests (skip in fast CI)",
        "ml_nondeterministic: tests with inherent randomness",
        "ml_smoke: fast smoke tests (<5 min, run on every PR)",
        "ml_gpu: tests requiring GPU hardware",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)

    # Initialize report collector
    config._mltk_collector = MltkReportCollector()


@pytest.fixture
def ml_config() -> MltkConfig:
    """Load MltkConfig from project configuration."""
    return MltkConfig.load()


@pytest.fixture
def ml_report(request) -> MltkReportCollector:  # type: ignore[no-untyped-def]
    """Access the session-level report collector."""
    return request.config._mltk_collector


def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Capture test results for mltk report."""
    if call.when != "call":
        return

    collector = getattr(item.config, "_mltk_collector", None)
    if collector is None:
        return

    outcome = "passed" if call.excinfo is None else "failed"
    duration = call.duration if hasattr(call, "duration") else 0.0

    # Extract MltkAssertionError result if available
    ml_result = None
    if call.excinfo is not None:
        exc = call.excinfo.value
        if hasattr(exc, "result"):
            ml_result = exc.result

    collector.add(
        nodeid=item.nodeid,
        outcome=outcome,
        duration=duration,
        ml_result=ml_result,
    )


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    """Generate mltk report and/or JSON export at end of session."""
    collector = getattr(session.config, "_mltk_collector", None)

    # --- JSON export (independent of --mltk-report) ---
    json_export_path = session.config.getoption("--mltk-export-json", default=None)
    if json_export_path and collector is not None:
        _export_json(collector, json_export_path, session)

    # --- MLflow logging (independent of --mltk-report) ---
    mlflow_experiment = session.config.getoption("--mltk-mlflow", default=None)
    if mlflow_experiment and collector is not None:
        _log_mlflow(collector, mlflow_experiment, session)

    # --- mltk server push (independent of --mltk-report) ---
    server_url = session.config.getoption("--mltk-server", default=None)
    if server_url and collector is not None:
        _push_to_server(collector, server_url, session)

    # --- HTML/terminal report ---
    if not session.config.getoption("--mltk-report", default=False):
        return

    if collector is None or collector.total == 0:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    writer = reporter._tw

    elapsed = time.time() - collector.start_time

    writer.sep("=", "MLTK Test Report")
    writer.line(f"Total: {collector.total} | "
                f"Passed: {collector.passed_count} | "
                f"Failed: {collector.failed_count} | "
                f"Time: {elapsed:.2f}s")
    writer.line("")

    # Group by module
    modules: dict[str, list[dict[str, object]]] = {}
    for r in collector.results:
        nodeid = str(r["nodeid"])
        parts = nodeid.split("::")
        module = parts[0] if parts else "unknown"
        modules.setdefault(module, []).append(r)

    for module, results in sorted(modules.items()):
        passed = sum(1 for r in results if r["outcome"] == "passed")
        total = len(results)
        status = "PASS" if passed == total else "FAIL"
        writer.line(f"  [{status}] {module}: {passed}/{total} passed")

    # Show failures with details
    failures = [r for r in collector.results if r["outcome"] == "failed"]
    if failures:
        writer.line("")
        writer.line("Failed assertions:")
        for r in failures:
            ml_result = r.get("ml_result")
            if ml_result and hasattr(ml_result, "message"):
                writer.line(f"  - {r['nodeid']}: {ml_result.message}")
            else:
                writer.line(f"  - {r['nodeid']}")

    writer.sep("=")

    # Generate HTML report
    try:
        from mltk.report.generator import generate_report

        report_path = generate_report(collector.results)
        writer.line(f"HTML report: {report_path}")
    except ImportError:
        pass  # plotly/jinja2 not installed — skip HTML report


def _log_mlflow(
    collector: MltkReportCollector,
    experiment_name: str,
    session: object,
) -> None:
    """Log collected test results to MLflow.

    Converts the flat collector records into a :class:`~mltk.core.result.TestSuite`
    and delegates to :class:`~mltk.integrations.mlflow_logger.MlflowLogger`.

    Args:
        collector: The report collector with accumulated results.
        experiment_name: MLflow experiment name (value of ``--mltk-mlflow``).
        session: pytest session (used for terminal reporter access).
    """
    from mltk.core.result import Severity, TestResult, TestSuite  # noqa: PLC0415
    from mltk.integrations.mlflow_logger import MlflowLogger  # noqa: PLC0415

    suite = TestSuite()
    for r in collector.results:
        ml_result = r.get("ml_result")
        duration_s = float(r.get("duration", 0.0))  # type: ignore[arg-type]

        if ml_result is not None:
            # Re-use the existing TestResult from the assertion
            suite.add(ml_result)
        else:
            # Synthesise a minimal TestResult from the pytest record
            suite.add(
                TestResult(
                    name=str(r["nodeid"]),
                    passed=r["outcome"] == "passed",
                    severity=Severity.INFO,
                    message="",
                    duration_ms=round(duration_s * 1000, 3),
                )
            )

    try:
        logger = MlflowLogger(experiment_name=experiment_name)
        logger.log_results(suite)
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — report the error but do not break the test session
        try:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")  # type: ignore[union-attr]
            if reporter is not None:
                reporter._tw.line(f"mltk MLflow logging failed: {exc}")
        except Exception:  # noqa: BLE001
            pass


def _push_to_server(
    collector: MltkReportCollector,
    server_url: str,
    session: object,
) -> None:
    """POST collected test results to a running mltk server instance.

    Sends results to ``{server_url}/api/runs`` using the standard
    :class:`SubmitRunRequest` payload (project + results list).

    Args:
        collector: The report collector with accumulated results.
        server_url: Base URL of the mltk server (e.g., "http://localhost:8080").
        session: pytest session (used for terminal reporter access).
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    records = collector.to_json_records()
    payload = json.dumps({"project": "default", "results": records}).encode("utf-8")
    endpoint = server_url.rstrip("/") + "/api/runs"

    try:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            response_body = resp.read().decode("utf-8")
            try:
                run_data = json.loads(response_body)
                run_id = run_data.get("run_id", "?")
            except json.JSONDecodeError:
                run_id = "?"

        try:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")  # type: ignore[union-attr]
            if reporter is not None:
                reporter._tw.line(
                    f"mltk server: results pushed to {server_url} (run_id={run_id})"
                )
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — report the error but do not break the test session
        try:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")  # type: ignore[union-attr]
            if reporter is not None:
                reporter._tw.line(f"mltk server push failed: {exc}")
        except Exception:  # noqa: BLE001
            pass


def _export_json(
    collector: MltkReportCollector,
    json_path: str,
    session: object,
) -> None:
    """Write collected test results to a JSON file.

    The output is a JSON array where each element represents one test with
    fields: name, passed, severity, message, details, duration_ms, timestamp.

    Args:
        collector: The report collector with accumulated results.
        json_path: Destination file path for the JSON output.
        session: pytest session (used for terminal reporter access).
    """
    import pathlib

    records = collector.to_json_records()
    output_path = pathlib.Path(json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    # Announce path via terminal reporter if available
    try:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")  # type: ignore[union-attr]
        if reporter is not None:
            reporter._tw.line(f"mltk JSON export: {output_path.resolve()}")
    except Exception:  # noqa: BLE001
        pass  # Non-fatal — file was already written
