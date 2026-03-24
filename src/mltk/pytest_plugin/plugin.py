"""pytest plugin for mltk. Auto-registered via pyproject.toml entry-points.

Provides ML-specific markers, fixtures, and the --mltk-report flag
for generating test summaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
        self.results.append({
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": duration,
            "ml_result": ml_result,
        })

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r["outcome"] == "passed")

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r["outcome"] == "failed")

    @property
    def total(self) -> int:
        return len(self.results)


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    """Add --mltk-report option."""
    parser.addoption(
        "--mltk-report",
        action="store_true",
        default=False,
        help="Generate mltk test summary report at end of session",
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
    """Generate mltk report at end of session if --mltk-report is set."""
    if not session.config.getoption("--mltk-report", default=False):
        return

    collector = getattr(session.config, "_mltk_collector", None)
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
