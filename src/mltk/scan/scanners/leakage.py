"""Leakage scanner -- detect data leakage from features to target.

Data leakage is the #1 cause of "works in dev, fails in prod."
A feature like ``future_revenue`` that has 0.98 correlation with
the target ``will_churn`` produces a model that looks 99%
accurate during evaluation but collapses in production because
that feature is unavailable at prediction time.

How it works:
  1. Build a temporary DataFrame from X and y.
  2. For each numeric feature, compute Pearson correlation
     with the target column.
  3. Flag features with |correlation| above the threshold
     (default 0.80).
  4. Call ``assert_no_target_leakage`` from the training
     module to get a formal TestResult.
  5. Each finding carries the assertion call for reproduction.

No model is needed -- this is a data-only scanner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner
from mltk.training.leakage import assert_no_target_leakage

__all__ = ["LeakageScanner"]

_TARGET_COL_NAME = "__mltk_target__"


class LeakageScanner(Scanner):
    """Detect features with suspiciously high target correlation.

    Scans every numeric feature for Pearson correlation with
    the target. Features above the threshold are likely data
    leakage -- information that is available during training
    but not at inference time.

    This scanner does NOT require a trained model. It operates
    on raw data only, so it can run before model training to
    prevent wasted compute.

    Requires:
        X, y
    """

    name = "leakage"
    category = "data_quality"
    requires: set[str] = {"X", "y"}

    # Default correlation threshold for per-feature flagging.
    _DEFAULT_CORR_THRESHOLD: float = 0.80

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run leakage analysis and return findings.

        Args:
            ctx: Scan context with features X, target y,
                 and config.

        Returns:
            List of ScanFinding for leaky features.
        """
        findings: list[ScanFinding] = []
        corr_threshold = self._get_threshold(ctx)

        # Individual feature correlation checks.
        findings.extend(
            self._scan_correlations(ctx, corr_threshold),
        )

        # Formal assertion via assert_no_target_leakage.
        finding = self._run_formal_assertion(
            ctx, corr_threshold,
        )
        if finding is not None:
            findings.append(finding)

        return findings

    # ----------------------------------------------------------
    # per-feature correlation scan
    # ----------------------------------------------------------

    def _scan_correlations(
        self,
        ctx: ScanContext,
        corr_threshold: float,
    ) -> list[ScanFinding]:
        """Flag individual features above threshold."""
        findings: list[ScanFinding] = []
        y_arr = np.asarray(ctx.y, dtype=float)

        for col in ctx.numeric_columns:
            if col not in ctx.X.columns:
                continue
            col_vals = ctx.X[col].values.astype(float)

            # Skip constant columns (zero variance).
            if np.std(col_vals) == 0.0:
                continue

            # Pearson correlation.
            corr = self._pearson(col_vals, y_arr)
            abs_corr = abs(corr)

            if abs_corr < corr_threshold:
                continue

            severity = (
                Severity.CRITICAL
                if abs_corr >= 0.95
                else Severity.WARNING
            )
            result = TestResult(
                name=f"scan.leakage.{col}",
                passed=False,
                severity=severity,
                message=(
                    f"Feature '{col}' has "
                    f"{abs_corr:.2f} correlation with "
                    f"target (threshold: "
                    f"{corr_threshold})"
                ),
                details={
                    "column": col,
                    "correlation": round(corr, 4),
                    "abs_correlation": round(abs_corr, 4),
                    "threshold": corr_threshold,
                },
            )

            # Build a temp DataFrame for reproduction.
            df_for_assert = self._build_df(ctx, col)
            findings.append(ScanFinding(
                result=result,
                assertion_fn=assert_no_target_leakage,
                assertion_args=(
                    df_for_assert,
                    _TARGET_COL_NAME,
                ),
                assertion_kwargs={
                    "feature_cols": [col],
                    "corr_threshold": corr_threshold,
                },
                suggested_test=self._gen_test(
                    col, corr_threshold,
                ),
                scanner_name=self.name,
            ))

        return findings

    # ----------------------------------------------------------
    # formal assertion (all features at once)
    # ----------------------------------------------------------

    def _run_formal_assertion(
        self,
        ctx: ScanContext,
        corr_threshold: float,
    ) -> ScanFinding | None:
        """Run assert_no_target_leakage on all features."""
        df = self._build_full_df(ctx)
        feature_cols = [
            c for c in ctx.numeric_columns
            if c in ctx.X.columns
        ]
        if not feature_cols:
            return None

        try:
            assert_no_target_leakage(
                df,
                target_col=_TARGET_COL_NAME,
                feature_cols=feature_cols,
                corr_threshold=corr_threshold,
            )
        except MltkAssertionError as exc:
            result = TestResult(
                name="scan.leakage.all_features",
                passed=False,
                severity=Severity.CRITICAL,
                message=exc.result.message,
                details=exc.result.details,
            )
            return ScanFinding(
                result=result,
                assertion_fn=assert_no_target_leakage,
                assertion_args=(
                    df.copy(),
                    _TARGET_COL_NAME,
                ),
                assertion_kwargs={
                    "feature_cols": feature_cols,
                    "corr_threshold": corr_threshold,
                },
                suggested_test=self._gen_test_all(
                    corr_threshold,
                ),
                scanner_name=self.name,
            )
        return None

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _get_threshold(
        self, ctx: ScanContext,
    ) -> float:
        """Read threshold from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return float(
            scanner_cfg.get(
                "corr_threshold",
                self._DEFAULT_CORR_THRESHOLD,
            ),
        )

    @staticmethod
    def _pearson(
        a: np.ndarray, b: np.ndarray,
    ) -> float:
        """Compute Pearson correlation coefficient.

        Pure numpy, no scipy/sklearn needed.

        Args:
            a: First array.
            b: Second array.

        Returns:
            Correlation coefficient in [-1, 1].
        """
        if len(a) < 2:
            return 0.0
        a_f = a.astype(float)
        b_f = b.astype(float)
        a_mean = np.mean(a_f)
        b_mean = np.mean(b_f)
        a_diff = a_f - a_mean
        b_diff = b_f - b_mean
        numerator = np.sum(a_diff * b_diff)
        denom = np.sqrt(
            np.sum(a_diff ** 2) * np.sum(b_diff ** 2),
        )
        if denom == 0.0:
            return 0.0
        return float(numerator / denom)

    @staticmethod
    def _build_df(
        ctx: ScanContext, col: str,
    ) -> pd.DataFrame:
        """Build a 2-column DataFrame for one feature."""
        df = pd.DataFrame({
            col: ctx.X[col].values.copy(),
            _TARGET_COL_NAME: np.asarray(ctx.y).copy(),
        })
        return df

    @staticmethod
    def _build_full_df(ctx: ScanContext) -> pd.DataFrame:
        """Build DataFrame with all numeric features + target."""
        df = ctx.X.copy()
        df[_TARGET_COL_NAME] = np.asarray(ctx.y).copy()
        return df

    @staticmethod
    def _gen_test(
        col: str, threshold: float,
    ) -> str:
        """Generate a pytest snippet for one feature."""
        return (
            f"def test_no_leakage_{col}():\n"
            f"    \"\"\"Feature '{col}' must not "
            f"leak target info.\"\"\"\n"
            f"    df = X_test.copy()\n"
            f"    df['target'] = y_test\n"
            f"    assert_no_target_leakage(\n"
            f"        df,\n"
            f"        target_col='target',\n"
            f"        feature_cols=['{col}'],\n"
            f"        corr_threshold={threshold},\n"
            f"    )\n"
        )

    @staticmethod
    def _gen_test_all(threshold: float) -> str:
        """Generate a pytest snippet for all features."""
        return (
            "def test_no_leakage_all_features():\n"
            "    \"\"\"No feature should leak "
            "target information.\"\"\"\n"
            "    df = X_test.copy()\n"
            "    df['target'] = y_test\n"
            "    assert_no_target_leakage(\n"
            "        df,\n"
            "        target_col='target',\n"
            f"        corr_threshold={threshold},\n"
            "    )\n"
        )
