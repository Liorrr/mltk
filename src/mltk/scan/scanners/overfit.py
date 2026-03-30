"""Overfit scanner -- detect train-test performance gap.

A model with 99% training accuracy and 60% test accuracy has
memorised the training data instead of learning generalisable
patterns. OverfitScanner compares accuracy on training vs test
data and flags large gaps.

How it works:
  1. Compute accuracy on training data (X_train, y_train).
  2. Compute accuracy on test data (X, y).
  3. Call ``assert_no_overfitting`` with the gap.
  4. Each failure becomes a ScanFinding with the exact
     assertion call for reproduction.

Only runs when training data (X_train, y_train) is available
in the ScanContext.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.model.overfitting import assert_no_overfitting
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["OverfitScanner"]


class OverfitScanner(Scanner):
    """Detect overfitting by comparing train vs test accuracy.

    Computes accuracy on both training and test data, then
    checks whether the gap exceeds the allowed threshold.
    Only runs when training data is available in the scan
    context.

    Requires:
        model_fn, X, y, X_train, y_train
    """

    name = "overfit"
    category = "model_quality"
    requires: set[str] = {
        "model_fn", "X", "y", "X_train", "y_train",
    }

    _DEFAULT_MAX_GAP: float = 0.10

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run overfitting check and return findings.

        Args:
            ctx: Scan context with model_fn, train and test
                data.

        Returns:
            List of ScanFinding if overfitting detected.
        """
        findings: list[ScanFinding] = []
        max_gap = self._get_max_gap(ctx)

        train_score = self._accuracy(
            np.asarray(ctx.y_train),
            np.asarray(ctx.model_fn(ctx.X_train)),
        )
        test_score = self._accuracy(
            np.asarray(ctx.y),
            np.asarray(ctx.model_fn(ctx.X)),
        )

        try:
            assert_no_overfitting(
                train_score=train_score,
                test_score=test_score,
                max_gap=max_gap,
            )
        except MltkAssertionError as exc:
            gap = train_score - test_score
            result = TestResult(
                name="scan.overfit",
                passed=False,
                severity=Severity.CRITICAL,
                message=(
                    f"Overfitting detected: "
                    f"train={train_score:.4f}, "
                    f"test={test_score:.4f}, "
                    f"gap={gap:.4f}"
                ),
                details={
                    "train_score": round(
                        train_score, 4,
                    ),
                    "test_score": round(
                        test_score, 4,
                    ),
                    "gap": round(gap, 4),
                    "max_gap": max_gap,
                    **exc.result.details,
                },
            )
            findings.append(ScanFinding(
                result=result,
                assertion_fn=assert_no_overfitting,
                assertion_args=(),
                assertion_kwargs={
                    "train_score": train_score,
                    "test_score": test_score,
                    "max_gap": max_gap,
                },
                suggested_test=self._gen_test(
                    max_gap,
                ),
                scanner_name=self.name,
            ))
        return findings

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    @staticmethod
    def _accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute accuracy without sklearn."""
        if len(y_true) == 0:
            return 0.0
        return float(
            (
                np.asarray(y_true)
                == np.asarray(y_pred)
            ).mean(),
        )

    def _get_max_gap(
        self, ctx: ScanContext,
    ) -> float:
        """Read max gap from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return float(
            scanner_cfg.get(
                "max_gap", self._DEFAULT_MAX_GAP,
            ),
        )

    @staticmethod
    def _gen_test(max_gap: float) -> str:
        """Generate a pytest snippet for overfitting."""
        return (
            "def test_no_overfitting():\n"
            "    \"\"\"Train-test gap must be"
            " bounded.\"\"\"\n"
            "    train_acc = accuracy_score(\n"
            "        y_train, model.predict("
            "X_train),\n"
            "    )\n"
            "    test_acc = accuracy_score(\n"
            "        y_test, model.predict("
            "X_test),\n"
            "    )\n"
            "    assert_no_overfitting(\n"
            "        train_score=train_acc,\n"
            "        test_score=test_acc,\n"
            f"        max_gap={max_gap},\n"
            "    )\n"
        )
