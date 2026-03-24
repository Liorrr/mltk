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
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def score(self) -> float:
        """Score as percentage (0-100)."""
        if not self.results:
            return 0.0
        return (self.passed_count / self.total) * 100
