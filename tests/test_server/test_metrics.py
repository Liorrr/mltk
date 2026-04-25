"""Tests for ``mltk.server.metrics`` -- Prometheus exposition endpoint.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>

Coverage:
    1. ``metrics_response`` returns None when prometheus_client is missing.
    2. ``metrics_response`` returns (bytes, str) when prometheus_client is
       installed.
    3. ``record_assertion`` is a no-op (no exception) when unavailable.
    4. ``record_assertion`` increments the status+category labels when
       available.
    5. ``record_container_scan`` increments severity labels only for
       positive counts.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltk.server import metrics as metrics_module

# ---------------------------------------------------------------------------
# Tests: metrics_response
# ---------------------------------------------------------------------------


class TestMetricsResponse:
    # SCENARIO: prometheus_client is not installed
    # WHY: the /metrics endpoint must degrade gracefully so the rest of the
    #      server is usable even without the optional extra.
    # EXPECTED: metrics_response() returns None so the route can translate
    #           it into a 404 with an install hint.
    def test_metrics_response_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", False)

        result = metrics_module.metrics_response()

        assert result is None

    # SCENARIO: prometheus_client is installed and flag is True
    # WHY: serving /metrics is the whole point of this module -- if the
    #      payload is empty or the content-type wrong, Prometheus will
    #      silently drop the scrape.
    # EXPECTED: returns a (bytes, str) tuple where the content type matches
    #           what was patched in.
    def test_metrics_response_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_body = b"# HELP mltk_assertions_total\n"
        fake_ct = "text/plain; version=0.0.4; charset=utf-8"

        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", True)
        monkeypatch.setattr(
            metrics_module,
            "generate_latest",
            lambda: fake_body,
            raising=False,
        )
        monkeypatch.setattr(
            metrics_module,
            "CONTENT_TYPE_LATEST",
            fake_ct,
            raising=False,
        )

        result = metrics_module.metrics_response()

        assert result is not None
        body, content_type = result
        assert body == fake_body
        assert content_type == fake_ct
        assert isinstance(body, bytes)
        assert isinstance(content_type, str)


# ---------------------------------------------------------------------------
# Tests: record_assertion
# ---------------------------------------------------------------------------


class TestRecordAssertion:
    # SCENARIO: prometheus_client is not installed
    # WHY: call sites should not have to guard every record_assertion call;
    #      forcing them to would defeat the graceful-degradation design.
    # EXPECTED: the function returns without raising.
    def test_record_assertion_noop_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", False)

        metrics_module.record_assertion(
            category="data", passed=True, duration_s=0.1
        )

    # SCENARIO: prometheus_client is installed, a passing data assertion runs
    # WHY: the Counter must be incremented exactly once per assertion so the
    #      mltk_assertions_total series reflects reality.
    # EXPECTED: both counter and histogram labels() are called with the
    #           expected kwargs, and inc()/observe() run once each.
    def test_record_assertion_increments_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_counter = MagicMock()
        fake_histogram = MagicMock()

        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", True)
        monkeypatch.setattr(metrics_module, "ASSERTION_COUNTER", fake_counter)
        monkeypatch.setattr(
            metrics_module, "ASSERTION_DURATION", fake_histogram
        )

        metrics_module.record_assertion(
            category="data", passed=True, duration_s=0.1
        )

        fake_counter.labels.assert_called_once_with(
            status="passed", category="data"
        )
        fake_counter.labels.return_value.inc.assert_called_once_with()
        fake_histogram.labels.assert_called_once_with(category="data")
        fake_histogram.labels.return_value.observe.assert_called_once_with(0.1)

    # SCENARIO: a failing assertion is recorded
    # WHY: the status label distinguishes passed vs. failed runs; getting
    #      this wrong inverts dashboards and alerts.
    # EXPECTED: the counter is labelled with status="failed".
    def test_record_assertion_failed_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_counter = MagicMock()
        fake_histogram = MagicMock()

        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", True)
        monkeypatch.setattr(metrics_module, "ASSERTION_COUNTER", fake_counter)
        monkeypatch.setattr(
            metrics_module, "ASSERTION_DURATION", fake_histogram
        )

        metrics_module.record_assertion(
            category="model", passed=False, duration_s=0.25
        )

        fake_counter.labels.assert_called_once_with(
            status="failed", category="model"
        )


# ---------------------------------------------------------------------------
# Tests: record_container_scan
# ---------------------------------------------------------------------------


class TestRecordContainerScan:
    # SCENARIO: prometheus_client is not installed
    # WHY: container scans must work without the metrics extra.
    # EXPECTED: the function returns without raising.
    def test_record_container_scan_noop_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", False)

        metrics_module.record_container_scan(critical=5, high=10)

    # SCENARIO: a scan finds CRITICAL=1 and HIGH=2 vulnerabilities
    # WHY: we want one counter series per severity so dashboards can filter;
    #      zero-valued severities must not create empty series.
    # EXPECTED: labels() is called for CRITICAL and HIGH only, with the
    #           correct increment values.
    def test_record_container_scan_increments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_counter = MagicMock()

        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", True)
        monkeypatch.setattr(
            metrics_module, "CONTAINER_VULN_COUNTER", fake_counter
        )

        metrics_module.record_container_scan(critical=1, high=2)

        severities_called = [
            call.kwargs["severity"]
            for call in fake_counter.labels.call_args_list
        ]
        assert severities_called == ["CRITICAL", "HIGH"]
        inc_calls = fake_counter.labels.return_value.inc.call_args_list
        assert [c.args[0] for c in inc_calls] == [1, 2]

    # SCENARIO: a scan finds zero vulnerabilities at every severity
    # WHY: prometheus_client creates a label child on first .labels() call;
    #      calling it with zeros would publish empty series that clutter
    #      dashboards.
    # EXPECTED: labels() is never called.
    def test_record_container_scan_skips_zero_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_counter = MagicMock()

        monkeypatch.setattr(metrics_module, "PROMETHEUS_AVAILABLE", True)
        monkeypatch.setattr(
            metrics_module, "CONTAINER_VULN_COUNTER", fake_counter
        )

        metrics_module.record_container_scan()

        fake_counter.labels.assert_not_called()
