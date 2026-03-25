"""Test result types for mltk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Severity(Enum):
    """Severity level for test results."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TestResult:
    """Result of a single mltk assertion."""

    name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        status_color = "#22c55e" if self.passed else "#ef4444"
        status_label = "PASS" if self.passed else "FAIL"
        badge = (
            f'<span style="background:{status_color};color:#fff;'
            f'font-weight:700;padding:2px 10px;border-radius:4px;'
            f'font-size:0.8em;letter-spacing:0.05em;">{status_label}</span>'
        )
        details_html = ""
        if self.details:
            rows = "".join(
                f'<tr>'
                f'<td style="padding:3px 10px 3px 0;color:#a78bfa;font-weight:600;'
                f'white-space:nowrap;">{k}</td>'
                f'<td style="padding:3px 0;color:#e2e8f0;">{v}</td>'
                f'</tr>'
                for k, v in self.details.items()
            )
            details_html = (
                f'<table style="margin-top:8px;border-collapse:collapse;'
                f'font-size:0.85em;">{rows}</table>'
            )
        return (
            f'<div style="background:#1e1e2e;color:#e2e8f0;padding:12px 16px;'
            f'border-radius:8px;border-left:4px solid #7c3aed;'
            f'font-family:monospace;margin:4px 0;">'
            f'{badge} '
            f'<strong style="color:#c4b5fd;font-size:1em;">{self.name}</strong>'
            f'<div style="margin-top:6px;color:#cbd5e1;">{self.message}</div>'
            f'<div style="margin-top:4px;font-size:0.8em;color:#94a3b8;">'
            f'<span style="margin-right:16px;">duration: '
            f'<span style="color:#7c3aed;">{self.duration_ms:.2f} ms</span></span>'
            f'<span>severity: '
            f'<span style="color:#7c3aed;">{self.severity.value}</span></span>'
            f'</div>'
            f'{details_html}'
            f'</div>'
        )


@dataclass
class TestSuite:
    """Collection of test results from a run."""

    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        """Add a test result to the suite."""
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """True if all critical tests passed."""
        return all(r.passed for r in self.results if r.severity == Severity.CRITICAL)

    @property
    def total(self) -> int:
        """Total number of test results in the suite."""
        return len(self.results)

    @property
    def passed_count(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of tests that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def score(self) -> float:
        """Score as percentage (0-100)."""
        if not self.results:
            return 0.0
        return (self.passed_count / self.total) * 100

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        pct = f"{self.score:.1f}"
        header_color = "#22c55e" if self.passed else "#ef4444"
        pass_badge = (
            f'<span style="background:#22c55e;color:#fff;font-weight:700;'
            f'padding:2px 10px;border-radius:4px;font-size:0.8em;">'
            f'{self.passed_count} passed</span>'
        )
        fail_badge = (
            f'<span style="background:#ef4444;color:#fff;font-weight:700;'
            f'padding:2px 10px;border-radius:4px;font-size:0.8em;margin-left:6px;">'
            f'{self.failed_count} failed</span>'
        )
        total_duration = sum(r.duration_ms for r in self.results)
        rows = ""
        for r in self.results:
            row_bg = "#1a2e1a" if r.passed else "#2e1a1a"
            s_color = "#22c55e" if r.passed else "#ef4444"
            s_label = "PASS" if r.passed else "FAIL"
            s_badge = (
                f'<span style="background:{s_color};color:#fff;font-weight:700;'
                f'padding:1px 8px;border-radius:3px;font-size:0.75em;">{s_label}</span>'
            )
            rows += (
                f'<tr style="background:{row_bg};">'
                f'<td style="padding:6px 12px;color:#c4b5fd;font-weight:600;'
                f'white-space:nowrap;">{r.name}</td>'
                f'<td style="padding:6px 12px;text-align:center;">{s_badge}</td>'
                f'<td style="padding:6px 12px;color:#cbd5e1;">{r.message}</td>'
                f'<td style="padding:6px 12px;color:#94a3b8;text-align:right;'
                f'white-space:nowrap;">{r.duration_ms:.2f} ms</td>'
                f'</tr>'
            )
        return (
            f'<div style="background:#1e1e2e;color:#e2e8f0;padding:14px 18px;'
            f'border-radius:8px;border-left:4px solid {header_color};'
            f'font-family:monospace;margin:6px 0;">'
            f'<div style="font-size:1.1em;font-weight:700;color:{header_color};'
            f'margin-bottom:8px;">'
            f'{self.passed_count}/{self.total} passed ({pct}%)'
            f'</div>'
            f'<div style="margin-bottom:10px;">{pass_badge}{fail_badge}'
            f'<span style="margin-left:14px;font-size:0.8em;color:#94a3b8;">'
            f'total: {total_duration:.2f} ms</span></div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.85em;">'
            f'<thead><tr style="border-bottom:1px solid #3d3d5c;">'
            f'<th style="padding:6px 12px;text-align:left;color:#7c3aed;">Test</th>'
            f'<th style="padding:6px 12px;text-align:center;color:#7c3aed;">Status</th>'
            f'<th style="padding:6px 12px;text-align:left;color:#7c3aed;">Message</th>'
            f'<th style="padding:6px 12px;text-align:right;color:#7c3aed;">Duration</th>'
            f'</tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table>'
            f'</div>'
        )
