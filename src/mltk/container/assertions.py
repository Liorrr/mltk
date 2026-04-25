"""pytest-native assertions for container image security.

These functions wrap :class:`~mltk.container.trivy_adapter.TrivyAdapter`
so callers can drop a single ``assert_container_vulnerabilities`` or
``assert_no_secrets_in_image`` call into a pytest test -- no scanner
plumbing, no report parsing.

Both assertions obey the standard mltk contract:
  * Return a :class:`~mltk.core.result.TestResult` on pass or
    non-critical failure.
  * Raise :class:`~mltk.core.assertion.MltkAssertionError` when the
    assertion fails with CRITICAL severity.
  * Carry structured ``details`` (CVE list, secret list, counts) for
    downstream reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mltk.container.trivy_adapter import TrivyAdapter
from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity

if TYPE_CHECKING:
    from mltk.core.result import TestResult

__all__ = ["assert_container_vulnerabilities", "assert_no_secrets_in_image"]


_SEVERITY_FLOORS = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")


def _severity_index(severity: str) -> int:
    """Return ordinal index for a severity label (lower == worse)."""
    label = severity.upper()
    if label in _SEVERITY_FLOORS:
        return _SEVERITY_FLOORS.index(label)
    return len(_SEVERITY_FLOORS)


@timed_assertion
def assert_container_vulnerabilities(
    image: str,
    *,
    max_critical: int = 0,
    max_high: int = 0,
    severity_floor: str = "MEDIUM",
    adapter: TrivyAdapter | None = None,
) -> TestResult:
    """Assert a container image has no CVEs above the configured threshold.

    Args:
        image: Container image reference (e.g. ``"alpine:3.18"``).
        max_critical: Maximum allowed CRITICAL severity CVEs.
        max_high: Maximum allowed HIGH severity CVEs.
        severity_floor: Minimum severity to count, one of
            ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
            Anything less severe is ignored.
        adapter: Optional pre-configured :class:`TrivyAdapter`.
            Primarily used by tests to inject a mock.

    Returns:
        :class:`TestResult` with ``details`` carrying per-severity
        counts and the full CVE list.

    Raises:
        MltkAssertionError: When the CVE counts exceed the configured
            thresholds (severity ``CRITICAL``).
    """
    adapter = adapter or TrivyAdapter()
    report = adapter.scan_image(image)

    floor_index = _severity_index(severity_floor)
    cves = []
    severity_counts: dict[str, int] = dict.fromkeys(_SEVERITY_FLOORS, 0)
    for vuln in report.all_vulnerabilities():
        if _severity_index(vuln.severity) > floor_index:
            continue
        severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
        cves.append(
            {
                "id": vuln.id,
                "severity": vuln.severity,
                "pkg_name": vuln.pkg_name,
                "installed_version": vuln.installed_version,
                "fixed_in": vuln.fixed_version,
                "title": vuln.title,
            }
        )

    critical_count = severity_counts.get("CRITICAL", 0)
    high_count = severity_counts.get("HIGH", 0)
    medium_count = severity_counts.get("MEDIUM", 0)

    passed = critical_count <= max_critical and high_count <= max_high

    if passed:
        message = (
            f"{image}: {critical_count} CRITICAL / {high_count} HIGH / "
            f"{medium_count} MEDIUM vulnerabilities (within thresholds)"
        )
    else:
        message = (
            f"{image}: {critical_count} CRITICAL (max {max_critical}) / "
            f"{high_count} HIGH (max {max_high}) vulnerabilities exceed thresholds"
        )

    return assert_true(
        condition=passed,
        name="container.vulnerabilities",
        message=message,
        severity=Severity.CRITICAL,
        image=image,
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        cves=cves,
        max_critical=max_critical,
        max_high=max_high,
        severity_floor=severity_floor,
    )


@timed_assertion
def assert_no_secrets_in_image(
    image: str,
    *,
    adapter: TrivyAdapter | None = None,
) -> TestResult:
    """Assert a container image contains no exposed secrets.

    Args:
        image: Container image reference (e.g. ``"myapp:latest"``).
        adapter: Optional pre-configured :class:`TrivyAdapter`.
            Primarily used by tests to inject a mock.

    Returns:
        :class:`TestResult` with ``details`` carrying the list of any
        secret findings Trivy reported.

    Raises:
        MltkAssertionError: When Trivy reports one or more secrets
            (severity ``CRITICAL``).
    """
    adapter = adapter or TrivyAdapter()
    report = adapter.scan_image(image, scanners=["secret"])

    secrets = report.all_secrets()
    secret_details = [
        {
            "rule_id": s.rule_id,
            "category": s.category,
            "severity": s.severity,
            "title": s.title,
            "target": s.target,
        }
        for s in secrets
    ]

    passed = len(secrets) == 0

    if passed:
        message = f"{image}: no exposed secrets detected"
    else:
        message = f"{image}: {len(secrets)} exposed secret(s) detected"

    return assert_true(
        condition=passed,
        name="container.secrets",
        message=message,
        severity=Severity.CRITICAL,
        image=image,
        secret_count=len(secrets),
        secrets=secret_details,
    )
