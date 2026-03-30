"""Drift scanner -- detect distribution shifts within the dataset.

Feature distributions that change over time (or between dataset
halves) signal data pipeline issues, seasonality, or concept
drift. DriftScanner splits each numeric column into a reference
half and a current half, then runs a statistical test to detect
significant distribution changes.

How it works:
  1. For each numeric column, split into first-half (reference)
     and second-half (current).
  2. Call ``assert_no_drift`` with the KS test.
  3. Each failure becomes a ScanFinding with the exact
     assertion call for reproduction.

No model is needed -- this is a data-only scanner.
"""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.data.drift import assert_no_drift
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["DriftScanner"]


class DriftScanner(Scanner):
    """Detect distribution drift in numeric features.

    Splits each numeric column into first-half (reference) and
    second-half (current) and runs a KS test to detect
    significant distribution changes. This self-split approach
    works even without separate reference data.

    Requires:
        X
    """

    name = "drift"
    category = "data_quality"
    requires: set[str] = {"X"}

    _DEFAULT_METHOD: str = "ks"

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run drift analysis and return findings.

        Args:
            ctx: Scan context with features X and config.

        Returns:
            List of ScanFinding for every column with
            detected drift.
        """
        findings: list[ScanFinding] = []
        method = self._get_method(ctx)

        for col in ctx.numeric_columns:
            if col not in ctx.X.columns:
                continue
            finding = self._check_column(
                ctx, col, method,
            )
            if finding is not None:
                findings.append(finding)
        return findings

    # ----------------------------------------------------------
    # per-column check
    # ----------------------------------------------------------

    def _check_column(
        self,
        ctx: ScanContext,
        col: str,
        method: str,
    ) -> ScanFinding | None:
        """Test one column for drift. Return finding or None."""
        series = ctx.X[col].dropna()
        if len(series) < 2:
            return None

        mid = len(series) // 2
        reference = series.iloc[:mid]
        current = series.iloc[mid:]

        try:
            assert_no_drift(
                reference,
                current,
                method=method,
            )
        except MltkAssertionError as exc:
            result = TestResult(
                name=f"scan.drift.{col}",
                passed=False,
                severity=Severity.WARNING,
                message=(
                    f"Distribution drift detected "
                    f"in '{col}' ({method} test)"
                ),
                details={
                    "column": col,
                    "method": method,
                    "reference_size": len(reference),
                    "current_size": len(current),
                    **exc.result.details,
                },
            )
            return ScanFinding(
                result=result,
                assertion_fn=assert_no_drift,
                assertion_args=(
                    reference.copy(),
                    current.copy(),
                ),
                assertion_kwargs={
                    "method": method,
                },
                suggested_test=(
                    self._gen_test(col, method)
                ),
                scanner_name=self.name,
            )
        return None

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _get_method(
        self, ctx: ScanContext,
    ) -> str:
        """Read method from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return str(
            scanner_cfg.get(
                "method", self._DEFAULT_METHOD,
            ),
        )

    @staticmethod
    def _gen_test(col: str, method: str) -> str:
        """Generate a pytest snippet for drift check."""
        return (
            f"def test_no_drift_{col}():\n"
            f"    \"\"\"Column '{col}' must not"
            f" drift.\"\"\"\n"
            f"    mid = len(X_test) // 2\n"
            f"    reference = "
            f"X_test['{col}'].iloc[:mid]\n"
            f"    current = "
            f"X_test['{col}'].iloc[mid:]\n"
            f"    assert_no_drift(\n"
            f"        reference, current,\n"
            f"        method='{method}',\n"
            f"    )\n"
        )
