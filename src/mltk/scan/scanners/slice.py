"""Slice scanner -- find subgroups where the model underperforms.

A model with 91% overall accuracy might drop to 58% for users
over age 55. This is invisible in aggregate metrics but
devastating in production: an entire demographic gets bad
predictions. SliceScanner systematically tests every meaningful
subgroup so you catch these failures before users do.

How it works:
  1. Compute overall accuracy on the full test set.
  2. For each categorical column, test every unique value as
     a slice (e.g., gender='F', region='West').
  3. For each numeric column, bin into quartiles and test each
     bin as a slice.
  4. For each slice with enough samples, call ``assert_metric``
     and catch failures.
  5. Classify severity by how far accuracy dropped from overall.

Every finding carries the exact assertion call needed to
reproduce it, so ``to_suite()`` and ``to_test_file()`` work
out of the box.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.model.metrics import assert_metric
from mltk.scan.config import ScanContext
from mltk.scan.finding import FixSuggestion, ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["SliceScanner"]


class SliceScanner(Scanner):
    """Find data slices where model accuracy drops.

    Iterates categorical columns (each unique value) and
    numeric columns (quartile bins) to discover subgroups
    where predictions degrade significantly compared to the
    overall population.

    Requires:
        model_fn, X, y
    """

    name = "slice"
    category = "performance"
    requires: set[str] = {"model_fn", "X", "y"}

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run slice analysis and return findings.

        Args:
            ctx: Scan context with model, data, and config.

        Returns:
            List of ScanFinding for every slice that fails
            the accuracy threshold.
        """
        findings: list[ScanFinding] = []
        y_pred = np.asarray(ctx.model_fn(ctx.X))
        y_true = np.asarray(ctx.y)
        overall = self._accuracy(y_true, y_pred)

        findings.extend(
            self._scan_categorical(ctx, y_true, y_pred, overall),
        )
        findings.extend(
            self._scan_numeric(ctx, y_true, y_pred, overall),
        )
        return findings

    # ----------------------------------------------------------
    # categorical slicing
    # ----------------------------------------------------------

    def _scan_categorical(
        self,
        ctx: ScanContext,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        overall: float,
    ) -> list[ScanFinding]:
        findings: list[ScanFinding] = []
        cfg = ctx.config
        for col in ctx.categorical_columns:
            values = ctx.X[col].unique()
            limit = cfg.max_slices_per_column
            for val in values[:limit]:
                mask = (ctx.X[col] == val).values
                if int(mask.sum()) < cfg.min_slice_samples:
                    continue
                finding = self._check_slice(
                    y_true,
                    y_pred,
                    mask,
                    overall,
                    ctx,
                    col,
                    f"{col}={val}",
                )
                if finding is not None:
                    findings.append(finding)
        return findings

    # ----------------------------------------------------------
    # numeric slicing (quartile bins)
    # ----------------------------------------------------------

    def _scan_numeric(
        self,
        ctx: ScanContext,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        overall: float,
    ) -> list[ScanFinding]:
        findings: list[ScanFinding] = []
        cfg = ctx.config
        percentiles = [25, 50, 75]
        for col in ctx.numeric_columns:
            col_vals = ctx.X[col].dropna().values
            if len(col_vals) == 0:
                continue
            edges = np.percentile(col_vals, percentiles)
            bins = self._build_bins(edges)
            for label, lo, hi in bins:
                mask = self._range_mask(
                    ctx.X[col].values, lo, hi,
                )
                if int(mask.sum()) < cfg.min_slice_samples:
                    continue
                desc = f"{col} {label}"
                finding = self._check_slice(
                    y_true,
                    y_pred,
                    mask,
                    overall,
                    ctx,
                    col,
                    desc,
                )
                if finding is not None:
                    findings.append(finding)
        return findings

    # ----------------------------------------------------------
    # single-slice check
    # ----------------------------------------------------------

    def _check_slice(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
        overall: float,
        ctx: ScanContext,
        col: str,
        description: str,
    ) -> ScanFinding | None:
        """Test one slice. Return ScanFinding or None."""
        cfg = ctx.config
        y_slice = y_true[mask].copy()
        p_slice = y_pred[mask].copy()
        threshold = overall - cfg.critical_drop

        try:
            assert_metric(
                y_slice,
                p_slice,
                metric="accuracy",
                threshold=threshold,
            )
        except MltkAssertionError:
            drop = overall - self._accuracy(y_slice, p_slice)
            severity = self._classify(drop, cfg)
            result = TestResult(
                name=f"scan.slice.{col}",
                passed=False,
                severity=severity,
                message=(
                    f"Accuracy drops to "
                    f"{self._accuracy(y_slice, p_slice):.2f}"
                    f" for {description}"
                    f" (overall: {overall:.2f})"
                ),
                details={
                    "column": col,
                    "slice": description,
                    "slice_accuracy": round(
                        self._accuracy(y_slice, p_slice), 4,
                    ),
                    "overall_accuracy": round(overall, 4),
                    "drop": round(drop, 4),
                    "slice_size": int(mask.sum()),
                },
            )
            slice_acc = self._accuracy(
                y_slice, p_slice,
            )
            return ScanFinding(
                result=result,
                assertion_fn=assert_metric,
                assertion_args=(y_slice, p_slice),
                assertion_kwargs={
                    "metric": "accuracy",
                    "threshold": threshold,
                },
                suggested_test=self._gen_test(
                    col, description, threshold,
                ),
                scanner_name=self.name,
                suggested_fixes=self._gen_fix(
                    col,
                    description,
                    slice_acc,
                    threshold,
                ),
            )
        return None

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    @staticmethod
    def _accuracy(
        y_true: np.ndarray, y_pred: np.ndarray,
    ) -> float:
        """Compute accuracy without sklearn."""
        if len(y_true) == 0:
            return 0.0
        return float(
            (np.asarray(y_true) == np.asarray(y_pred)).mean(),
        )

    @staticmethod
    def _classify(
        drop: float,
        cfg: Any,
    ) -> Severity:
        if drop >= cfg.critical_drop:
            return Severity.CRITICAL
        if drop >= cfg.warning_drop:
            return Severity.WARNING
        return Severity.INFO

    @staticmethod
    def _build_bins(
        edges: np.ndarray,
    ) -> list[tuple[str, float, float]]:
        """Build labeled (label, lo, hi) bins from edges.

        Returns quartile ranges:
          <= p25, p25-p50, p50-p75, > p75
        """
        bins: list[tuple[str, float, float]] = []
        bins.append(
            (f"<= {edges[0]:.2f}", -np.inf, edges[0]),
        )
        for i in range(len(edges) - 1):
            bins.append((
                f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                edges[i],
                edges[i + 1],
            ))
        bins.append(
            (f"> {edges[-1]:.2f}", edges[-1], np.inf),
        )
        return bins

    @staticmethod
    def _range_mask(
        values: np.ndarray,
        lo: float,
        hi: float,
    ) -> np.ndarray:
        """Boolean mask for values in (lo, hi]."""
        if lo == -np.inf:
            return values <= hi
        if hi == np.inf:
            return values > lo
        return (values > lo) & (values <= hi)

    @staticmethod
    def _gen_fix(
        slice_col: str,
        slice_desc: str,
        metric_val: float,
        threshold: float,
    ) -> list[FixSuggestion]:
        """Generate fix suggestions for a slice failure.

        Args:
            slice_col: Column name used for slicing.
            slice_desc: Human-readable slice description
                (e.g. "gender=F").
            metric_val: Actual metric value for the slice.
            threshold: Required metric threshold.

        Returns:
            3 FixSuggestions ranked by confidence.
        """
        return [
            FixSuggestion(
                category="data",
                title=(
                    f"Collect more data for "
                    f"'{slice_desc}'"
                ),
                description=(
                    f"Slice '{slice_desc}' "
                    f"underperforms "
                    f"(metric={metric_val:.3f} < "
                    f"{threshold}). Collect more "
                    f"training examples for this "
                    f"subgroup."
                ),
                confidence="high",
            ),
            FixSuggestion(
                category="code",
                title=(
                    "Add slice-specific model or "
                    "feature engineering"
                ),
                description=(
                    f"Consider training a "
                    f"specialized model or adding "
                    f"features that capture the "
                    f"characteristics of the "
                    f"'{slice_desc}' subgroup."
                ),
                confidence="medium",
            ),
            FixSuggestion(
                category="process",
                title="Add slice monitoring",
                description=(
                    "Add per-slice performance "
                    "monitoring to catch subgroup "
                    "degradation early."
                ),
                confidence="medium",
                code_snippet=(
                    f"assert_metric(y_true[mask], "
                    f"y_pred[mask], "
                    f"metric='accuracy', "
                    f"threshold={threshold})"
                ),
            ),
        ]

    @staticmethod
    def _gen_test(
        col: str,
        description: str,
        threshold: float,
    ) -> str:
        """Generate a pytest snippet for this finding."""
        return (
            f"def test_slice_{col}_performance():\n"
            f"    \"\"\"Accuracy for {description}"
            f" must meet threshold.\"\"\"\n"
            f"    mask = X_test['{col}'] "
            f"  # filter for slice\n"
            f"    y_slice = y_test[mask]\n"
            f"    p_slice = y_pred[mask]\n"
            f"    assert_metric(\n"
            f"        y_slice, p_slice,\n"
            f"        metric='accuracy',\n"
            f"        threshold={threshold:.4f},\n"
            f"    )\n"
        )
