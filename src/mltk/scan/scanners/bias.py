"""Bias scanner -- detect fairness violations across demographic groups.

An approval model might accept 80% of one gender but only 51%
of another. This violates the US four-fifths rule, the EU AI
Act (effective Aug 2026), and basic fairness expectations.
BiasScanner checks every sensitive column against multiple
fairness metrics so you catch discrimination before deployment.

How it works:
  1. Identify sensitive columns (user-specified or auto-detected
     from column names like 'gender', 'race', 'age').
  2. For each sensitive column, run ``assert_no_bias`` with
     demographic parity (selection rate equality).
  3. For classifiers, also run equalized odds (TPR/FPR parity).
  4. User-specified columns produce CRITICAL/WARNING findings;
     auto-detected columns produce INFO findings (advisory).
  5. Each finding carries the assertion call for reproduction.

Note: Chouldechova-Kleinberg impossibility theorem means you
cannot satisfy all fairness metrics simultaneously when group
base rates differ. Choose the metric for your use case.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.model.bias import assert_no_bias
from mltk.scan.config import ScanContext
from mltk.scan.detect import _SENSITIVE_KEYWORDS
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["BiasScanner"]

# Column names that suggest sensitive attributes.


def _detect_sensitive_columns(
    columns: list[str],
) -> list[str]:
    """Return column names that look like sensitive attrs.

    Matches against ``_SENSITIVE_KEYWORDS`` using
    case-insensitive substring matching.

    Args:
        columns: All column names in the dataset.

    Returns:
        List of column names that match sensitive keywords.
    """
    detected: list[str] = []
    for col in columns:
        col_lower = col.lower().replace("-", "_")
        for kw in _SENSITIVE_KEYWORDS:
            if kw in col_lower:
                detected.append(col)
                break
    return detected


class BiasScanner(Scanner):
    """Detect fairness violations across demographic groups.

    Tests each sensitive column with demographic parity and
    (for classifiers) equalized odds. User-specified columns
    get CRITICAL severity; auto-detected columns get INFO
    so teams can triage without alert fatigue.

    Requires:
        model_fn, X, y
    """

    name = "bias"
    category = "fairness"
    requires: set[str] = {"model_fn", "X", "y"}

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run bias analysis and return findings.

        Args:
            ctx: Scan context with model, data, config, and
                 optionally ``sensitive_columns``.

        Returns:
            List of ScanFinding for every fairness violation.
        """
        findings: list[ScanFinding] = []
        y_pred = np.asarray(ctx.model_fn(ctx.X))
        y_true = np.asarray(ctx.y)

        user_cols = set(ctx.sensitive_columns or [])
        auto_cols = _detect_sensitive_columns(
            list(ctx.X.columns),
        )

        # User-specified columns first (higher severity).
        for col in user_cols:
            if col not in ctx.X.columns:
                continue
            findings.extend(
                self._check_column(
                    ctx, y_true, y_pred, col,
                    is_user_specified=True,
                ),
            )

        # Auto-detected columns that user did not specify.
        for col in auto_cols:
            if col in user_cols:
                continue
            if col not in ctx.X.columns:
                continue
            findings.extend(
                self._check_column(
                    ctx, y_true, y_pred, col,
                    is_user_specified=False,
                ),
            )

        return findings

    # ----------------------------------------------------------
    # per-column checks
    # ----------------------------------------------------------

    def _check_column(
        self,
        ctx: ScanContext,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        col: str,
        *,
        is_user_specified: bool,
    ) -> list[ScanFinding]:
        """Test one sensitive column with all methods."""
        findings: list[ScanFinding] = []
        groups = np.asarray(ctx.X[col])

        # Skip columns with fewer than 2 unique groups.
        if len(np.unique(groups)) < 2:
            return findings

        # demographic_parity always runs.
        finding = self._run_method(
            y_true,
            y_pred,
            groups,
            col,
            method="demographic_parity",
            is_user_specified=is_user_specified,
        )
        if finding is not None:
            findings.append(finding)

        # equalized_odds only for classifiers.
        if ctx.model_type == "classifier":
            finding = self._run_method(
                y_true,
                y_pred,
                groups,
                col,
                method="equalized_odds",
                is_user_specified=is_user_specified,
            )
            if finding is not None:
                findings.append(finding)

        return findings

    def _run_method(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        col: str,
        method: str,
        *,
        is_user_specified: bool,
    ) -> ScanFinding | None:
        """Run one bias method. Return finding or None."""
        severity = (
            Severity.CRITICAL
            if is_user_specified
            else Severity.INFO
        )
        try:
            assert_no_bias(
                y_true,
                y_pred,
                sensitive_feature=groups,
                method=method,
                severity=severity,
            )
        except MltkAssertionError as exc:
            # Determine severity from the drop magnitude.
            actual_sev = self._classify_severity(
                exc.result, is_user_specified,
            )
            result = TestResult(
                name=f"scan.bias.{col}.{method}",
                passed=False,
                severity=actual_sev,
                message=(
                    f"{method.replace('_', ' ').title()}"
                    f" violation on '{col}'"
                    f" ({self._stat_summary(exc.result)})"
                ),
                details={
                    "column": col,
                    "method": method,
                    "user_specified": is_user_specified,
                    **exc.result.details,
                },
            )
            return ScanFinding(
                result=result,
                assertion_fn=assert_no_bias,
                assertion_args=(
                    y_true.copy(),
                    y_pred.copy(),
                    groups.copy(),
                ),
                assertion_kwargs={
                    "method": method,
                    "severity": severity,
                },
                suggested_test=self._gen_test(
                    col, method,
                ),
                scanner_name=self.name,
            )
        return None

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    @staticmethod
    def _classify_severity(
        result: TestResult,
        is_user_specified: bool,
    ) -> Severity:
        """Pick severity based on source and magnitude."""
        if not is_user_specified:
            return Severity.INFO
        stat = result.details.get("statistic", 0.0)
        # Large violations (> 0.20 diff) are critical.
        if isinstance(stat, (int, float)) and stat > 0.20:
            return Severity.CRITICAL
        return Severity.WARNING

    @staticmethod
    def _stat_summary(result: TestResult) -> str:
        """One-line summary from result details."""
        stat = result.details.get("statistic")
        if stat is not None:
            return f"statistic={stat:.4f}"
        return result.message[:60]

    @staticmethod
    def _gen_test(col: str, method: str) -> str:
        """Generate a pytest snippet for this finding."""
        return (
            f"def test_bias_{col}_{method}():\n"
            f"    \"\"\"No {method.replace('_', ' ')}"
            f" bias on '{col}'.\"\"\"\n"
            f"    assert_no_bias(\n"
            f"        y_test, y_pred,\n"
            f"        sensitive_feature="
            f"X_test['{col}'],\n"
            f"        method='{method}',\n"
            f"    )\n"
        )
