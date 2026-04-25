"""ContainerScanner -- Trivy-backed container image security scanner.

Unlike the data/model scanners under ``mltk.scan.scanners``, this
scanner does not subclass :class:`mltk.scan.scanners.base.Scanner`:
container image analysis has no model, no X, no y, so the
:class:`~mltk.scan.config.ScanContext` contract does not apply.

Instead, :class:`ContainerScanner` exposes a sibling ``scan(image_ref)``
method that returns the same :class:`~mltk.scan.finding.ScanFinding`
objects so that report, suite, and CLI consumers can ingest results
uniformly.

Findings are grouped by severity: one finding per severity bucket
that contains at least one CVE, so a report with 1 CRITICAL + 2
HIGH produces two findings rather than three.
"""

from __future__ import annotations

from mltk.container.assertions import assert_container_vulnerabilities
from mltk.container.trivy_adapter import (
    TrivyAdapter,
    TrivyReport,
    TrivyVulnerability,
)
from mltk.core.result import Severity, TestResult
from mltk.scan.finding import FixSuggestion, ScanFinding

__all__ = ["ContainerScanner"]


_SEVERITY_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")


class ContainerScanner:
    """Scan a container image and emit mltk :class:`ScanFinding` objects.

    Attributes:
        name: Scanner identifier (``"container"``).
        category: Scanner category (``"security"``).
    """

    name = "container"
    category = "security"

    def __init__(self, adapter: TrivyAdapter | None = None) -> None:
        """Create a scanner.

        Args:
            adapter: Optional pre-configured :class:`TrivyAdapter`.
                When ``None``, a default adapter (which auto-discovers
                the Trivy binary) is constructed on demand.
        """
        self._adapter = adapter

    @property
    def adapter(self) -> TrivyAdapter:
        """Return the adapter, creating a default one if needed."""
        if self._adapter is None:
            self._adapter = TrivyAdapter()
        return self._adapter

    def scan(self, image_ref: str, **cfg: object) -> list[ScanFinding]:
        """Scan an image and return one finding per populated severity group.

        Args:
            image_ref: Container image reference (e.g. ``"alpine:3.18"``).
            **cfg: Optional scanner configuration. Recognised keys:
                ``max_critical`` (int, default 0),
                ``max_high`` (int, default 0),
                ``severity_floor`` (str, default ``"MEDIUM"``).

        Returns:
            List of :class:`ScanFinding` objects, one per severity
            group that contains at least one CVE above the floor.
        """
        max_critical = int(cfg.get("max_critical", 0))  # type: ignore[arg-type]
        max_high = int(cfg.get("max_high", 0))  # type: ignore[arg-type]
        severity_floor = str(cfg.get("severity_floor", "MEDIUM"))

        report = self.adapter.scan_image(image_ref)
        return self._report_to_findings(
            report,
            image_ref,
            max_critical=max_critical,
            max_high=max_high,
            severity_floor=severity_floor,
        )

    def _report_to_findings(
        self,
        report: TrivyReport,
        image_ref: str,
        *,
        max_critical: int,
        max_high: int,
        severity_floor: str,
    ) -> list[ScanFinding]:
        """Group vulnerabilities by severity and emit one finding each."""
        findings: list[ScanFinding] = []
        floor_index = _severity_index(severity_floor)

        grouped: dict[str, list[TrivyVulnerability]] = {}
        for vuln in report.all_vulnerabilities():
            if _severity_index(vuln.severity) > floor_index:
                continue
            grouped.setdefault(vuln.severity, []).append(vuln)

        for severity_label in _SEVERITY_ORDER:
            vulns = grouped.get(severity_label)
            if not vulns:
                continue
            findings.append(
                self._build_finding(
                    image_ref=image_ref,
                    severity_label=severity_label,
                    vulns=vulns,
                    max_critical=max_critical,
                    max_high=max_high,
                    severity_floor=severity_floor,
                )
            )

        return findings

    def _build_finding(
        self,
        *,
        image_ref: str,
        severity_label: str,
        vulns: list[TrivyVulnerability],
        max_critical: int,
        max_high: int,
        severity_floor: str = "MEDIUM",
    ) -> ScanFinding:
        """Construct a :class:`ScanFinding` for one severity group."""
        cve_details = [
            {
                "id": v.id,
                "severity": v.severity,
                "pkg_name": v.pkg_name,
                "installed_version": v.installed_version,
                "fixed_in": v.fixed_version,
                "title": v.title,
            }
            for v in vulns
        ]

        if severity_label == "CRITICAL":
            passed = len(vulns) <= max_critical
            severity = Severity.CRITICAL
        elif severity_label == "HIGH":
            passed = len(vulns) <= max_high
            severity = Severity.CRITICAL if not _allowed(len(vulns), max_high) else Severity.WARNING
        else:
            passed = True
            severity = Severity.WARNING

        message = (
            f"{len(vulns)} {severity_label} vulnerabilit"
            f"{'y' if len(vulns) == 1 else 'ies'} found in {image_ref}"
        )

        result = TestResult(
            name=f"scan.container.{severity_label.lower()}",
            passed=passed,
            severity=severity,
            message=message,
            details={
                "image": image_ref,
                "severity": severity_label,
                "count": len(vulns),
                "cves": cve_details,
            },
        )

        return ScanFinding(
            result=result,
            assertion_fn=assert_container_vulnerabilities,
            assertion_args=(image_ref,),
            assertion_kwargs={
                "max_critical": max_critical,
                "max_high": max_high,
                "severity_floor": severity_floor,
            },
            suggested_test=_gen_test_snippet(
                image_ref, max_critical, max_high, severity_floor,
            ),
            suggested_fixes=_gen_fix_suggestions(severity_label, vulns),
            scanner_name=self.name,
        )


def _severity_index(severity: str) -> int:
    """Return the ordinal index for a severity label (lower == worse)."""
    label = severity.upper()
    if label in _SEVERITY_ORDER:
        return _SEVERITY_ORDER.index(label)
    return len(_SEVERITY_ORDER)


def _allowed(count: int, maximum: int) -> bool:
    """True if ``count`` is within the allowed maximum."""
    return count <= maximum


def _gen_test_snippet(
    image_ref: str,
    max_critical: int,
    max_high: int,
    severity_floor: str = "MEDIUM",
) -> str:
    """Generate a runnable pytest snippet for the finding."""
    floor_arg = (
        f"        severity_floor={severity_floor!r},\n"
        if severity_floor != "MEDIUM"
        else ""
    )
    return (
        "def test_container_vulnerabilities():\n"
        f"    \"\"\"Image '{image_ref}' must stay within CVE thresholds.\"\"\"\n"
        "    from mltk.container import assert_container_vulnerabilities\n"
        "    assert_container_vulnerabilities(\n"
        f"        '{image_ref}',\n"
        f"        max_critical={max_critical},\n"
        f"        max_high={max_high},\n"
        f"{floor_arg}"
        "    )\n"
    )


def _gen_fix_suggestions(
    severity_label: str,
    vulns: list[TrivyVulnerability],
) -> list[FixSuggestion]:
    """Generate fix suggestions for a severity group."""
    fixable = [v for v in vulns if v.fixed_version]
    suggestions: list[FixSuggestion] = []

    if fixable:
        pkgs = ", ".join(sorted({v.pkg_name for v in fixable if v.pkg_name})[:5])
        suggestions.append(
            FixSuggestion(
                category="config",
                title=f"Update packages with known fixes ({severity_label})",
                description=(
                    f"{len(fixable)} {severity_label} vulnerabilities have "
                    f"upstream fixes available. Update: {pkgs or 'affected packages'}."
                ),
                confidence="high",
                code_snippet=(
                    "# In your Dockerfile, refresh packages:\n"
                    "RUN apk update && apk upgrade  # alpine\n"
                    "# or\n"
                    "RUN apt-get update && apt-get upgrade -y  # debian/ubuntu\n"
                ),
            )
        )

    suggestions.append(
        FixSuggestion(
            category="process",
            title="Rebuild on a newer base image",
            description=(
                "Pin the base image to a newer tag or digest that ships with "
                "patched packages, and rebuild the container."
            ),
            confidence="medium",
        )
    )

    return suggestions
