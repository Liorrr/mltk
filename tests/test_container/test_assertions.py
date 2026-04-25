"""Tests for :mod:`mltk.container.assertions` and :class:`ContainerScanner`."""

from __future__ import annotations

import pytest

from mltk.container import (
    ContainerScanner,
    assert_container_vulnerabilities,
    assert_no_secrets_in_image,
)
from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity


def test_assert_vulnerabilities_fails_on_critical(
    mock_trivy_adapter,
) -> None:
    """One CRITICAL CVE with ``max_critical=0`` raises MltkAssertionError."""
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_container_vulnerabilities(
            "alpine:3.18",
            max_critical=0,
            max_high=0,
            adapter=mock_trivy_adapter,
        )

    result = exc_info.value.result
    assert not result.passed
    assert result.severity == Severity.CRITICAL
    assert result.details["critical_count"] == 1
    assert result.details["high_count"] == 2
    assert result.details["medium_count"] == 1
    cve_ids = {c["id"] for c in result.details["cves"]}
    assert "CVE-2024-1234" in cve_ids


def test_assert_vulnerabilities_passes_when_under_threshold(
    mock_trivy_adapter,
) -> None:
    """High thresholds let the fixture pass without raising."""
    result = assert_container_vulnerabilities(
        "alpine:3.18",
        max_critical=5,
        max_high=5,
        adapter=mock_trivy_adapter,
    )
    assert result.passed
    assert result.details["critical_count"] == 1
    assert result.details["high_count"] == 2
    assert result.duration_ms >= 0


def test_assert_vulnerabilities_passes_when_empty(
    make_empty_adapter,
) -> None:
    """Empty report passes with strict thresholds."""
    adapter = make_empty_adapter()
    result = assert_container_vulnerabilities(
        "empty:latest",
        max_critical=0,
        max_high=0,
        adapter=adapter,
    )
    assert result.passed
    assert result.details["critical_count"] == 0
    assert result.details["cves"] == []


def test_assert_vulnerabilities_severity_floor_filters_medium(
    mock_trivy_adapter,
) -> None:
    """``severity_floor='HIGH'`` excludes the MEDIUM CVE from counts."""
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_container_vulnerabilities(
            "alpine:3.18",
            max_critical=0,
            max_high=0,
            severity_floor="HIGH",
            adapter=mock_trivy_adapter,
        )
    result = exc_info.value.result
    assert result.details["medium_count"] == 0
    cve_severities = {c["severity"] for c in result.details["cves"]}
    assert "MEDIUM" not in cve_severities


def test_assert_no_secrets_fails_when_secret_present(
    mock_trivy_adapter,
) -> None:
    """Fixture contains one secret, so the assertion raises."""
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_no_secrets_in_image(
            "alpine:3.18", adapter=mock_trivy_adapter,
        )
    result = exc_info.value.result
    assert not result.passed
    assert result.details["secret_count"] == 1
    assert result.details["secrets"][0]["rule_id"] == "aws-access-key-id"


def test_assert_no_secrets_passes_when_clean(make_empty_adapter) -> None:
    """Empty report has no secrets and the assertion returns passed."""
    adapter = make_empty_adapter()
    result = assert_no_secrets_in_image("clean:latest", adapter=adapter)
    assert result.passed
    assert result.details["secret_count"] == 0


def test_scanner_emits_findings_per_severity(mock_trivy_adapter) -> None:
    """ContainerScanner groups vulns by severity -- one finding per group."""
    scanner = ContainerScanner(adapter=mock_trivy_adapter)
    findings = scanner.scan("alpine:3.18")

    assert len(findings) == 3
    severities = [f.result.details["severity"] for f in findings]
    assert severities == ["CRITICAL", "HIGH", "MEDIUM"]

    critical_finding = findings[0]
    assert critical_finding.scanner_name == "container"
    assert critical_finding.assertion_fn is assert_container_vulnerabilities
    assert critical_finding.assertion_args == ("alpine:3.18",)
    assert critical_finding.assertion_kwargs == {
        "max_critical": 0,
        "max_high": 0,
        "severity_floor": "MEDIUM",
    }
    assert critical_finding.result.details["count"] == 1
    assert not critical_finding.result.passed


def test_scanner_finding_has_fix_suggestions(mock_trivy_adapter) -> None:
    """Findings with fixable CVEs include a configuration fix suggestion."""
    scanner = ContainerScanner(adapter=mock_trivy_adapter)
    findings = scanner.scan("alpine:3.18")

    critical = findings[0]
    assert critical.suggested_fixes
    categories = {fix.category for fix in critical.suggested_fixes}
    assert "config" in categories or "process" in categories
    assert critical.suggested_test.startswith("def test_container_vulnerabilities")


def test_scanner_respects_severity_floor(mock_trivy_adapter) -> None:
    """``severity_floor='HIGH'`` drops the MEDIUM finding."""
    scanner = ContainerScanner(adapter=mock_trivy_adapter)
    findings = scanner.scan("alpine:3.18", severity_floor="HIGH")
    severities = [f.result.details["severity"] for f in findings]
    assert "MEDIUM" not in severities
    assert set(severities) == {"CRITICAL", "HIGH"}


def test_scanner_high_threshold_allows_pass(mock_trivy_adapter) -> None:
    """With permissive thresholds the CRITICAL finding still reports but passes."""
    scanner = ContainerScanner(adapter=mock_trivy_adapter)
    findings = scanner.scan(
        "alpine:3.18", max_critical=5, max_high=5,
    )
    critical = [f for f in findings if f.result.details["severity"] == "CRITICAL"][0]
    assert critical.result.passed
