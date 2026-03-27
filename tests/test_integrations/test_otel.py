"""Tests for mltk.integrations.otel — OpenTelemetry integration.

Most tests exercise the **no-op** and **JSON export** paths so they work
WITHOUT opentelemetry installed.  This is intentional: mltk treats OTEL as
an optional extra, and the integration must never crash when OTEL is absent.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str = "accuracy",
    passed: bool = True,
    severity: str = "info",
    message: str = "ok",
    duration_ms: float = 10.0,
) -> dict[str, Any]:
    """Build a minimal result dict matching the format expected by MltkTracer."""
    return {
        "name": name,
        "passed": passed,
        "severity": severity,
        "message": message,
        "duration_ms": duration_ms,
    }


def _make_mixed_results() -> list[dict[str, Any]]:
    """Build a list of results with a mix of passed and failed."""
    return [
        _make_result("accuracy", passed=True, severity="critical", duration_ms=5.0),
        _make_result("drift_psi", passed=True, severity="warning", duration_ms=15.0),
        _make_result("latency_p99", passed=False, severity="critical",
                     message="450ms > 200ms threshold", duration_ms=8.0),
    ]


# ---------------------------------------------------------------------------
# Tests: No-op mode (OTEL not installed)
# ---------------------------------------------------------------------------

class TestNoOpMode:
    """MltkTracer behaves as a silent no-op when opentelemetry is absent."""

    def test_init_noop_mode(self) -> None:
        # SCENARIO: MltkTracer is instantiated without opentelemetry installed.
        # WHY: Users who do not need OTEL tracing must be able to import and
        #      instantiate MltkTracer without any error.  This is the "graceful
        #      degradation" contract.
        # EXPECTED: No exception raised; tracer is in no-op mode.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        # If OTEL happens to be installed in the test env, this still passes
        # because the test only checks "no crash".  The is_active property
        # reflects the actual state.
        assert isinstance(tracer, MltkTracer)

    def test_is_active_without_otel(self) -> None:
        # SCENARIO: Check is_active when opentelemetry is not installed.
        # WHY: Downstream code may branch on is_active to decide whether to
        #      configure OTEL-specific settings.  It must reliably reflect
        #      the real availability of the tracing backend.
        # EXPECTED: is_active is False when _OTEL_AVAILABLE is False.

        from unittest.mock import patch

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer.__new__(MltkTracer)
        tracer._tracer = None
        tracer._service_name = "mltk"
        tracer._endpoint = None

        # Force _OTEL_AVAILABLE to False
        with patch("mltk.integrations.otel._OTEL_AVAILABLE", False):
            assert tracer.is_active is False

    def test_trace_result_noop_no_crash(self) -> None:
        # SCENARIO: trace_result is called when OTEL is not available.
        # WHY: CI pipelines that call trace_result must not crash just because
        #      the OTEL collector is missing or the package is not installed.
        # EXPECTED: No exception; method returns immediately (no-op).

        from unittest.mock import patch

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer.__new__(MltkTracer)
        tracer._tracer = None
        tracer._service_name = "mltk"
        tracer._endpoint = None

        with patch("mltk.integrations.otel._OTEL_AVAILABLE", False):
            tracer.trace_result(_make_result())  # must not raise

    def test_trace_suite_noop_no_crash(self) -> None:
        # SCENARIO: trace_suite is called when OTEL is not available.
        # WHY: Same graceful-degradation contract as trace_result, but for
        #      the suite-level method.  An entire list of results must be
        #      safely ignored without errors.
        # EXPECTED: No exception; method returns immediately (no-op).

        from unittest.mock import patch

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer.__new__(MltkTracer)
        tracer._tracer = None
        tracer._service_name = "mltk"
        tracer._endpoint = None

        with patch("mltk.integrations.otel._OTEL_AVAILABLE", False):
            tracer.trace_suite(_make_mixed_results())  # must not raise


# ---------------------------------------------------------------------------
# Tests: JSON export (always works, no OTEL dependency)
# ---------------------------------------------------------------------------

class TestExportJson:
    """MltkTracer.export_json() — OTLP-compatible JSON output."""

    def test_export_json_writes_valid_json(self, tmp_path: Path) -> None:
        # SCENARIO: export_json is called with a list of results.
        # WHY: The primary value of export_json is producing a file that
        #      standard JSON parsers can load.  If the file is malformed,
        #      no downstream tool can use it.
        # EXPECTED: Output file exists, is valid JSON, and can be loaded.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "spans.json")
        tracer.export_json(_make_mixed_results(), out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "resourceSpans" in data

    def test_export_json_correct_span_structure(self, tmp_path: Path) -> None:
        # SCENARIO: Verify that each span in the export has the expected
        #           OTLP fields (name, kind, attributes, status).
        # WHY: OTLP ingest endpoints (Jaeger, Tempo) expect specific field
        #      names.  If we emit "spanName" instead of "name", the import
        #      will silently drop the data.
        # EXPECTED: Each span dict has name, kind, startTimeUnixNano,
        #           endTimeUnixNano, attributes (list), and status (dict).

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "spans.json")
        tracer.export_json([_make_result("accuracy", passed=True)], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1

        span = spans[0]
        assert span["name"] == "accuracy"
        assert span["kind"] == "SPAN_KIND_INTERNAL"
        assert "startTimeUnixNano" in span
        assert "endTimeUnixNano" in span
        assert isinstance(span["attributes"], list)
        assert isinstance(span["status"], dict)

    def test_export_json_empty_results(self, tmp_path: Path) -> None:
        # SCENARIO: export_json is called with an empty list of results.
        # WHY: A pipeline might export results before any tests have run
        #      (e.g., warm-up phase).  The method must produce a valid JSON
        #      file with an empty spans array, not crash.
        # EXPECTED: Valid JSON with zero spans.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "empty.json")
        tracer.export_json([], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert spans == []

    def test_export_json_failed_result_error_status(self, tmp_path: Path) -> None:
        # SCENARIO: A failed assertion is exported to JSON.
        # WHY: Observability platforms colour-code spans by status.  If a
        #      failed assertion shows STATUS_CODE_OK, the user will miss the
        #      failure in the Jaeger/Tempo UI.
        # EXPECTED: The span status.code is "STATUS_CODE_ERROR" and the
        #           status.message contains the failure reason.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "failed.json")
        failed = _make_result("latency", passed=False, message="too slow")
        tracer.export_json([failed], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span["status"]["code"] == "STATUS_CODE_ERROR"
        assert "too slow" in span["status"]["message"]

    def test_export_json_passed_result_ok_status(self, tmp_path: Path) -> None:
        # SCENARIO: A passed assertion is exported to JSON.
        # WHY: Counterpart to the error-status test — passed assertions must
        #      show STATUS_CODE_OK so they appear green in trace viewers.
        # EXPECTED: status.code is "STATUS_CODE_OK", message is empty.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "passed.json")
        tracer.export_json([_make_result("accuracy", passed=True)], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span["status"]["code"] == "STATUS_CODE_OK"
        assert span["status"]["message"] == ""

    def test_export_json_span_attributes(self, tmp_path: Path) -> None:
        # SCENARIO: Verify that span attributes contain the correct OTEL
        #           attribute keys and typed values.
        # WHY: OTLP attributes use a specific schema: each attribute is a
        #      dict with "key" and "value" where the value is typed
        #      (stringValue, boolValue, doubleValue).  If we use the wrong
        #      type, the collector may reject the data or lose precision.
        # EXPECTED: All five mltk.assertion.* attributes are present with
        #           correct keys and typed values.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "attrs.json")
        result = _make_result(
            "drift_psi", passed=False, severity="critical",
            message="PSI=0.35 > 0.25", duration_ms=42.5,
        )
        tracer.export_json([result], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {a["key"]: a["value"] for a in span["attributes"]}

        assert attrs["mltk.assertion.name"] == {"stringValue": "drift_psi"}
        assert attrs["mltk.assertion.passed"] == {"boolValue": False}
        assert attrs["mltk.assertion.severity"] == {"stringValue": "critical"}
        assert attrs["mltk.assertion.duration_ms"] == {"doubleValue": 42.5}
        assert attrs["mltk.assertion.message"] == {"stringValue": "PSI=0.35 > 0.25"}

    def test_export_json_duration_recorded(self, tmp_path: Path) -> None:
        # SCENARIO: A result with a specific duration_ms is exported.
        # WHY: Duration is the primary metric for identifying slow assertions.
        #      The nanosecond timestamps (startTimeUnixNano, endTimeUnixNano)
        #      must reflect the duration so that trace viewers show correct
        #      span widths on the timeline.
        # EXPECTED: endTimeUnixNano - startTimeUnixNano == duration_ms * 1e6
        #           (within tolerance for float arithmetic).

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "duration.json")
        tracer.export_json([_make_result(duration_ms=123.456)], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        span = data["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        actual_ns = span["endTimeUnixNano"] - span["startTimeUnixNano"]
        expected_ns = int(123.456 * 1_000_000)
        assert actual_ns == expected_ns

    def test_export_json_service_name(self, tmp_path: Path) -> None:
        # SCENARIO: MltkTracer is created with a custom service_name.
        # WHY: The service name appears in the OTLP resource attributes and
        #      is how Jaeger/Tempo groups traces by service.  A wrong service
        #      name means traces are mixed with unrelated services.
        # EXPECTED: The resource attribute "service.name" matches the
        #           service_name passed to the constructor.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer(service_name="my-model-tests")
        out = str(tmp_path / "svc.json")
        tracer.export_json([_make_result()], out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        resource_attrs = data["resourceSpans"][0]["resource"]["attributes"]
        svc_attr = next(a for a in resource_attrs if a["key"] == "service.name")
        assert svc_attr["value"]["stringValue"] == "my-model-tests"

    def test_export_json_returns_path(self, tmp_path: Path) -> None:
        # SCENARIO: export_json returns the resolved file path.
        # WHY: Callers (CI scripts, pytest fixtures) often need the path for
        #      artifact upload.  Returning it avoids redundant Path operations.
        # EXPECTED: Return value is a string path pointing to the written file.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "ret.json")
        returned = tracer.export_json([_make_result()], out)

        assert Path(returned).exists()
        assert returned == str(Path(out).resolve())

    def test_export_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        # SCENARIO: export_json is given a path whose parent directory does
        #           not yet exist.
        # WHY: In CI, output directories are often not pre-created.  The
        #      method should create intermediate directories automatically
        #      (like ``mkdir -p``), not fail with FileNotFoundError.
        # EXPECTED: File is created in a nested directory that did not exist.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "nested" / "deep" / "spans.json")
        tracer.export_json([_make_result()], out)

        assert Path(out).exists()

    def test_export_json_multiple_results(self, tmp_path: Path) -> None:
        # SCENARIO: export_json is called with multiple results (3 mixed).
        # WHY: The typical use case is exporting an entire suite of results.
        #      The number of spans in the output must match the number of
        #      results, and each span must have the correct name.
        # EXPECTED: 3 spans, names match the input result names.

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()
        out = str(tmp_path / "multi.json")
        results = _make_mixed_results()
        tracer.export_json(results, out)

        data = json.loads(Path(out).read_text(encoding="utf-8"))
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 3
        names = [s["name"] for s in spans]
        assert names == ["accuracy", "drift_psi", "latency_p99"]
