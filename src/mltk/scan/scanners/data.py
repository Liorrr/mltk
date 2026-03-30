"""Data scanner -- check data quality before model evaluation.

Bad data invalidates every downstream check. DataScanner runs
first in the pipeline to catch nulls and PII before any model
scanner wastes compute on corrupted inputs.

How it works:
  1. For each column, call ``assert_no_nulls`` to detect
     missing values.
  2. For each string/object column, call ``assert_no_pii``
     to detect personally identifiable information.
  3. Each failure becomes a ScanFinding with the exact
     assertion call for reproduction.

No model is needed -- this is a data-only scanner.
"""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.data.pii import assert_no_pii
from mltk.data.schema import assert_no_nulls
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["DataScanner"]


class DataScanner(Scanner):
    """Detect data quality issues: nulls and PII.

    Iterates every column checking for null values, then
    scans string columns for personally identifiable
    information. Runs before model scanners because bad
    data invalidates all downstream checks.

    Requires:
        X
    """

    name = "data"
    category = "data_quality"
    requires: set[str] = {"X"}

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run data quality checks and return findings.

        Args:
            ctx: Scan context with features X and config.

        Returns:
            List of ScanFinding for every data quality
            issue discovered.
        """
        findings: list[ScanFinding] = []
        findings.extend(self._check_nulls(ctx))
        findings.extend(self._check_pii(ctx))
        return findings

    # ----------------------------------------------------------
    # null checks
    # ----------------------------------------------------------

    def _check_nulls(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Check each column for null values."""
        findings: list[ScanFinding] = []
        for col in ctx.X.columns:
            if ctx.X[col].isnull().sum() == 0:
                continue
            try:
                assert_no_nulls(
                    ctx.X,
                    columns=[col],
                )
            except MltkAssertionError as exc:
                null_count = int(
                    ctx.X[col].isnull().sum(),
                )
                result = TestResult(
                    name=f"scan.data.nulls.{col}",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Column '{col}' has "
                        f"{null_count} null value(s)"
                    ),
                    details={
                        "column": col,
                        "null_count": null_count,
                        **exc.result.details,
                    },
                )
                findings.append(ScanFinding(
                    result=result,
                    assertion_fn=assert_no_nulls,
                    assertion_args=(
                        ctx.X[[col]].copy(),
                    ),
                    assertion_kwargs={
                        "columns": [col],
                    },
                    suggested_test=(
                        self._gen_null_test(col)
                    ),
                    scanner_name=self.name,
                ))
        return findings

    # ----------------------------------------------------------
    # PII checks
    # ----------------------------------------------------------

    def _check_pii(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Check string columns for PII."""
        findings: list[ScanFinding] = []
        str_cols = [
            col for col in ctx.X.columns
            if (
                pd.api.types.is_string_dtype(
                    ctx.X[col],
                )
                or pd.api.types.is_object_dtype(
                    ctx.X[col],
                )
            )
        ]
        if not str_cols:
            return findings

        for col in str_cols:
            try:
                assert_no_pii(
                    ctx.X,
                    columns=[col],
                )
            except MltkAssertionError as exc:
                result = TestResult(
                    name=f"scan.data.pii.{col}",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=(
                        f"PII detected in column "
                        f"'{col}'"
                    ),
                    details={
                        "column": col,
                        **exc.result.details,
                    },
                )
                findings.append(ScanFinding(
                    result=result,
                    assertion_fn=assert_no_pii,
                    assertion_args=(
                        ctx.X[[col]].copy(),
                    ),
                    assertion_kwargs={
                        "columns": [col],
                    },
                    suggested_test=(
                        self._gen_pii_test(col)
                    ),
                    scanner_name=self.name,
                ))
        return findings

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    @staticmethod
    def _gen_null_test(col: str) -> str:
        """Generate a pytest snippet for null check."""
        return (
            f"def test_no_nulls_{col}():\n"
            f"    \"\"\"Column '{col}' must have"
            f" no nulls.\"\"\"\n"
            f"    assert_no_nulls(\n"
            f"        X_test,\n"
            f"        columns=['{col}'],\n"
            f"    )\n"
        )

    @staticmethod
    def _gen_pii_test(col: str) -> str:
        """Generate a pytest snippet for PII check."""
        return (
            f"def test_no_pii_{col}():\n"
            f"    \"\"\"Column '{col}' must have"
            f" no PII.\"\"\"\n"
            f"    assert_no_pii(\n"
            f"        X_test,\n"
            f"        columns=['{col}'],\n"
            f"    )\n"
        )
