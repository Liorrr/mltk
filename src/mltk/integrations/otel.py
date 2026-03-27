"""OpenTelemetry integration — trace mltk assertion execution with OTEL spans.

WHY OpenTelemetry for ML testing:

    When you run 150+ assertions in a CI pipeline, you need to know:
    - Which assertions are slow? (latency bottleneck)
    - Which assertions fail most often? (reliability signal)
    - How does assertion performance change over time? (regression detection)

    OpenTelemetry (OTEL) is the industry standard for distributed tracing.
    By wrapping each assertion in an OTEL span, you get:
    - Per-assertion timing in Jaeger/Zipkin/Grafana Tempo
    - Failure attributes searchable in your observability platform
    - Correlation between assertion failures and deployment events
    - Cost attribution (which test suites consume the most resources)

    This module gracefully degrades: if opentelemetry is not installed,
    every method is a no-op with zero overhead and zero import errors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Graceful import — set _OTEL_AVAILABLE flag
# ---------------------------------------------------------------------------
# We try to import the opentelemetry tracing API at module level.  If it is
# not installed, we set a flag so that every public method becomes a silent
# no-op.  This is the standard pattern for optional-dependency integrations:
# the user can always ``from mltk.integrations.otel import MltkTracer``
# without worrying about whether opentelemetry is on the path.

_OTEL_AVAILABLE: bool

try:
    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Span JSON helpers (always available, no OTEL dependency)
# ---------------------------------------------------------------------------

def _result_to_span_dict(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a single test-result dict into an OTEL-compatible span dict.

    The returned dict mirrors the OTLP JSON span format so that the output
    of :meth:`MltkTracer.export_json` can be ingested by Jaeger, Grafana
    Tempo, or any tool that reads OTLP JSON.

    WHY a custom serializer instead of using the OTEL SDK's built-in JSON
    exporter?  Because ``export_json`` must work **without** opentelemetry
    installed.  Pure dict -> JSON conversion means zero external dependencies.

    Args:
        result: A dict with keys ``name``, ``passed``, ``severity``,
            ``message``, and optionally ``duration_ms``.

    Returns:
        A dict representing a single OTLP-style span.
    """
    passed = result.get("passed", True)
    duration_ms = result.get("duration_ms", 0.0)

    # Convert duration_ms to nanoseconds (OTLP convention)
    duration_ns = int(duration_ms * 1_000_000)
    start_ns = int(time.time() * 1_000_000_000)
    end_ns = start_ns + duration_ns

    return {
        "name": result.get("name", "unknown"),
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "attributes": [
            {"key": "mltk.assertion.name", "value": {"stringValue": result.get("name", "")}},
            {"key": "mltk.assertion.passed", "value": {"boolValue": passed}},
            {
                "key": "mltk.assertion.severity",
                "value": {"stringValue": result.get("severity", "info")},
            },
            {"key": "mltk.assertion.duration_ms", "value": {"doubleValue": duration_ms}},
            {"key": "mltk.assertion.message", "value": {"stringValue": result.get("message", "")}},
        ],
        "status": {
            "code": "STATUS_CODE_OK" if passed else "STATUS_CODE_ERROR",
            "message": "" if passed else result.get("message", "assertion failed"),
        },
    }


# ---------------------------------------------------------------------------
# MltkTracer
# ---------------------------------------------------------------------------

class MltkTracer:
    """Trace mltk assertion execution with OpenTelemetry spans.

    WHY OpenTelemetry for ML testing:

    When you run 150+ assertions in a CI pipeline, you need to know:

    - Which assertions are slow? (latency bottleneck)
    - Which assertions fail most often? (reliability signal)
    - How does assertion performance change over time? (regression detection)

    OpenTelemetry (OTEL) is the industry standard for distributed tracing.
    By wrapping each assertion in an OTEL span, you get:

    - Per-assertion timing in Jaeger/Zipkin/Grafana Tempo
    - Failure attributes searchable in your observability platform
    - Correlation between assertion failures and deployment events
    - Cost attribution (which test suites consume the most resources)

    The tracer works in two modes:

    1. **Real mode**: When ``opentelemetry-api`` is installed, creates actual
       spans sent to your OTLP collector (Jaeger, Tempo, etc.).
    2. **No-op mode**: When opentelemetry is not installed, all methods are
       no-ops (zero overhead, zero errors).  This means you can always import
       ``MltkTracer`` regardless of whether OTEL is installed.

    Example (real mode)::

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer(service_name="my-model-tests",
                            endpoint="http://localhost:4317")
        tracer.trace_result({"name": "accuracy", "passed": True,
                             "severity": "critical", "message": "0.95 >= 0.90",
                             "duration_ms": 42.5})

    Example (no-op mode — opentelemetry not installed)::

        from mltk.integrations.otel import MltkTracer

        tracer = MltkTracer()          # no error
        tracer.trace_result({...})      # silent no-op
        assert not tracer.is_active     # confirms no-op mode

    Example (JSON export — always works)::

        tracer = MltkTracer()
        tracer.export_json(results, "/tmp/spans.json")
        # File can be imported into Jaeger or analyzed with scripts
    """

    def __init__(
        self,
        service_name: str = "mltk",
        endpoint: str | None = None,
    ) -> None:
        """Initialize the tracer.

        In **real mode** (opentelemetry installed), this sets up a
        ``TracerProvider`` with an OTLP gRPC exporter pointing at the
        given endpoint.

        In **no-op mode** (opentelemetry not installed), this stores
        the configuration but does nothing else.  No error is raised.

        Args:
            service_name: OTEL service name (appears in Jaeger/Tempo UI as
                the service that produced the spans).  Defaults to ``"mltk"``.
            endpoint: OTLP collector gRPC endpoint (e.g.,
                ``"http://localhost:4317"``).  If ``None``, the SDK falls
                back to the ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment
                variable, then to ``http://localhost:4317``.
        """
        self._service_name = service_name
        self._endpoint = endpoint
        self._tracer: Any = None

        if _OTEL_AVAILABLE:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)

            exporter_kwargs: dict[str, Any] = {}
            if endpoint is not None:
                exporter_kwargs["endpoint"] = endpoint
                exporter_kwargs["insecure"] = True

            exporter = OTLPSpanExporter(**exporter_kwargs)
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            # Register as the global provider so that downstream code
            # using ``trace.get_tracer()`` sees the same provider.
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("mltk")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether real OTEL tracing is available (vs no-op mode).

        Returns ``True`` when the ``opentelemetry`` packages are installed
        and the tracer was successfully initialised.  Returns ``False``
        otherwise — in which case every tracing method is a silent no-op.
        """
        return _OTEL_AVAILABLE and self._tracer is not None

    # ------------------------------------------------------------------
    # Tracing API
    # ------------------------------------------------------------------

    def trace_result(self, result: dict[str, Any]) -> None:
        """Record a single test result as an OTEL span.

        Creates a span whose name is the assertion name (e.g.,
        ``"model.metric.accuracy"``) and whose attributes capture the
        pass/fail status, severity, duration, and human-readable message.

        In **no-op mode** this method returns immediately.

        WHY per-result spans: Each assertion becomes independently
        searchable in your observability platform.  You can query
        ``mltk.assertion.passed = false`` to find all failing assertions
        across all pipeline runs, or sort by ``mltk.assertion.duration_ms``
        to find the slowest ones.

        Args:
            result: A dict with the following keys:

                - ``name`` (str): Assertion name.
                - ``passed`` (bool): Whether the assertion passed.
                - ``severity`` (str): One of ``"critical"``, ``"warning"``,
                  ``"info"``.
                - ``message`` (str): Human-readable result message.
                - ``duration_ms`` (float, optional): Execution time in ms.
        """
        if not self.is_active:
            return

        name = result.get("name", "unknown")
        passed = result.get("passed", True)
        severity = result.get("severity", "info")
        duration_ms = result.get("duration_ms", 0.0)
        message = result.get("message", "")

        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute("mltk.assertion.name", name)
            span.set_attribute("mltk.assertion.passed", passed)
            span.set_attribute("mltk.assertion.severity", severity)
            span.set_attribute("mltk.assertion.duration_ms", duration_ms)
            span.set_attribute("mltk.assertion.message", message)

            if passed:
                span.set_status(StatusCode.OK)
            else:
                span.set_status(StatusCode.ERROR, message)

    def trace_suite(self, results: list[dict[str, Any]]) -> None:
        """Record an entire test suite as a parent span with child spans.

        Creates a parent span named ``"mltk.test_suite"`` that contains
        aggregate attributes (total, passed, failed counts) and one child
        span per individual result.

        WHY a parent span: Distributed-tracing UIs (Jaeger, Tempo) display
        spans in a tree.  The parent span gives you the "big picture" of a
        test run at a glance, and you can expand into child spans to see
        which specific assertions were slow or failing.

        In **no-op mode** this method returns immediately.

        Args:
            results: A list of result dicts (same format as
                :meth:`trace_result`).
        """
        if not self.is_active:
            return

        passed_count = sum(1 for r in results if r.get("passed", True))
        failed_count = len(results) - passed_count

        with self._tracer.start_as_current_span("mltk.test_suite") as parent:
            parent.set_attribute("mltk.suite.total", len(results))
            parent.set_attribute("mltk.suite.passed", passed_count)
            parent.set_attribute("mltk.suite.failed", failed_count)

            if failed_count > 0:
                parent.set_status(StatusCode.ERROR, f"{failed_count} assertions failed")
            else:
                parent.set_status(StatusCode.OK)

            for result in results:
                self.trace_result(result)

    # ------------------------------------------------------------------
    # JSON export (always available, no OTEL dependency)
    # ------------------------------------------------------------------

    def export_json(self, results: list[dict[str, Any]], output_path: str) -> str:
        """Export results as OTEL-compatible JSON (for offline analysis).

        WHY JSON export: Not every environment has an OTEL collector running.
        CI runners, local dev machines, and air-gapped environments may not
        have Jaeger or Tempo available.  This method serialises the results
        into the OTLP JSON span format so they can be:

        - Imported later into Jaeger (``jaeger-query`` supports JSON ingest)
        - Loaded into Grafana Tempo with ``tempo-cli``
        - Analysed with Python scripts (``json.load`` + pandas)
        - Archived alongside test artifacts in CI

        The JSON structure follows the OTLP ``ResourceSpans`` schema so that
        standard tooling can parse it without custom adapters.

        **This method works WITHOUT opentelemetry installed** because it
        performs pure dict-to-JSON serialisation using only the standard
        library.

        Args:
            results: A list of result dicts (same format as
                :meth:`trace_result`).
            output_path: File path where the JSON will be written.

        Returns:
            The absolute path of the written file (same as *output_path*
            after resolution).
        """
        spans = [_result_to_span_dict(r) for r in results]

        # Build the OTLP ResourceSpans envelope
        document = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            },
                        ],
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "mltk"},
                            "spans": spans,
                        },
                    ],
                },
            ],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return str(out.resolve())
