"""Prometheus metrics for mltk server.

Enable with: ``pip install mltk[metrics]``.

Metrics are exposed at ``/metrics`` in Prometheus exposition format when the
FastAPI app mounts :func:`metrics_response`. If ``prometheus_client`` is not
installed, the endpoint returns HTTP 404 with an install hint instead of
raising ``ImportError`` at import time.

Three metrics are defined:

* ``mltk_assertions_total`` (Counter) -- labels: ``status`` (passed/failed),
  ``category`` (e.g. data/model/llm/container).
* ``mltk_assertion_duration_seconds`` (Histogram) -- labels: ``category``.
* ``mltk_container_scan_vulnerabilities_total`` (Counter) -- labels:
  ``severity`` (CRITICAL/HIGH/MEDIUM/LOW).

The recording helpers (:func:`record_assertion`, :func:`record_container_scan`)
are no-ops when ``prometheus_client`` is unavailable so callers do not need to
guard every call site.
"""
from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    PROMETHEUS_AVAILABLE = False

__all__ = [
    "ASSERTION_COUNTER",
    "ASSERTION_DURATION",
    "CONTAINER_VULN_COUNTER",
    "PROMETHEUS_AVAILABLE",
    "metrics_response",
    "record_assertion",
    "record_container_scan",
]


if PROMETHEUS_AVAILABLE:
    ASSERTION_COUNTER: Counter = Counter(
        "mltk_assertions_total",
        "Total mltk assertions run",
        ["status", "category"],
    )
    ASSERTION_DURATION: Histogram = Histogram(
        "mltk_assertion_duration_seconds",
        "Duration of mltk assertions",
        ["category"],
        buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0],
    )
    CONTAINER_VULN_COUNTER: Counter = Counter(
        "mltk_container_scan_vulnerabilities_total",
        "Container image vulnerabilities found by Trivy",
        ["severity"],
    )
else:
    ASSERTION_COUNTER = None  # type: ignore[assignment]
    ASSERTION_DURATION = None  # type: ignore[assignment]
    CONTAINER_VULN_COUNTER = None  # type: ignore[assignment]


def record_assertion(*, category: str, passed: bool, duration_s: float) -> None:
    """Record a single assertion result.

    No-op if ``prometheus_client`` is not installed so callers never need to
    branch on availability.

    Args:
        category: Assertion family (e.g. ``"data"``, ``"model"``, ``"llm"``).
        passed: ``True`` if the assertion passed, ``False`` otherwise.
        duration_s: Wall-clock duration of the assertion in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    status = "passed" if passed else "failed"
    ASSERTION_COUNTER.labels(status=status, category=category).inc()
    ASSERTION_DURATION.labels(category=category).observe(duration_s)


def record_container_scan(
    *,
    critical: int = 0,
    high: int = 0,
    medium: int = 0,
    low: int = 0,
) -> None:
    """Record container image vulnerability counts from a Trivy scan.

    No-op if ``prometheus_client`` is not installed. Only severities with a
    positive count are incremented to avoid creating empty label series.

    Args:
        critical: Number of CRITICAL vulnerabilities.
        high: Number of HIGH vulnerabilities.
        medium: Number of MEDIUM vulnerabilities.
        low: Number of LOW vulnerabilities.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    for severity, count in (
        ("CRITICAL", critical),
        ("HIGH", high),
        ("MEDIUM", medium),
        ("LOW", low),
    ):
        if count > 0:
            CONTAINER_VULN_COUNTER.labels(severity=severity).inc(count)


def metrics_response() -> tuple[bytes, str] | None:
    """Generate a Prometheus exposition-format response body.

    Returns:
        A ``(body, content_type)`` tuple when ``prometheus_client`` is
        installed, or ``None`` otherwise. Callers should translate ``None``
        into an HTTP 404 so clients get a clear "metrics disabled" signal
        instead of a 500.
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    return generate_latest(), CONTENT_TYPE_LATEST
