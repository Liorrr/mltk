"""Calibration scanner -- detect probability miscalibration.

A classifier that says "90% confident" should be correct about
90% of the time. When predicted probabilities do not match
observed frequencies, downstream decision-making (thresholds,
risk scoring, triage) breaks silently.

How it works:
  1. Extract predicted probabilities via ``predict_proba_fn``.
  2. For binary classifiers, use column 1 (positive class).
  3. Call ``assert_calibration`` to compute ECE.
  4. Each failure becomes a ScanFinding with the exact
     assertion call for reproduction.

Only runs when ``predict_proba_fn`` is available.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.model.slicing import assert_calibration
from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["CalibrationScanner"]


class CalibrationScanner(Scanner):
    """Detect prediction probability miscalibration.

    Computes Expected Calibration Error (ECE) by comparing
    predicted probabilities against observed frequencies in
    equal-width bins. Only runs for classifiers that provide
    a ``predict_proba_fn``.

    Requires:
        predict_proba_fn, X, y
    """

    name = "calibration"
    category = "model_quality"
    requires: set[str] = {
        "predict_proba_fn", "X", "y",
    }

    _DEFAULT_MAX_ECE: float = 0.05
    _DEFAULT_N_BINS: int = 10

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run calibration check and return findings.

        Args:
            ctx: Scan context with predict_proba_fn, X, y.

        Returns:
            List of ScanFinding if calibration fails.
        """
        findings: list[ScanFinding] = []
        max_ece = self._get_max_ece(ctx)
        n_bins = self._get_n_bins(ctx)

        y_true = np.asarray(ctx.y)
        raw_proba = ctx.predict_proba_fn(ctx.X)
        y_prob = self._extract_positive_proba(
            raw_proba,
        )

        try:
            assert_calibration(
                y_true,
                y_prob,
                max_error=max_ece,
                n_bins=n_bins,
            )
        except MltkAssertionError as exc:
            result = TestResult(
                name="scan.calibration",
                passed=False,
                severity=Severity.WARNING,
                message=(
                    f"Model is poorly calibrated: "
                    f"{exc.result.message}"
                ),
                details={
                    "max_ece": max_ece,
                    "n_bins": n_bins,
                    **exc.result.details,
                },
            )
            findings.append(ScanFinding(
                result=result,
                assertion_fn=assert_calibration,
                assertion_args=(
                    y_true.copy(),
                    y_prob.copy(),
                ),
                assertion_kwargs={
                    "max_error": max_ece,
                    "n_bins": n_bins,
                },
                suggested_test=(
                    self._gen_test(max_ece, n_bins)
                ),
                scanner_name=self.name,
            ))
        return findings

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    @staticmethod
    def _extract_positive_proba(
        raw_proba: np.ndarray,
    ) -> np.ndarray:
        """Extract positive-class probabilities.

        For binary classifiers returning (n, 2) arrays,
        take column 1. For 1D arrays, use as-is.
        """
        proba = np.asarray(raw_proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        return proba

    def _get_max_ece(
        self, ctx: ScanContext,
    ) -> float:
        """Read max ECE from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return float(
            scanner_cfg.get(
                "max_ece", self._DEFAULT_MAX_ECE,
            ),
        )

    def _get_n_bins(
        self, ctx: ScanContext,
    ) -> int:
        """Read bin count from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return int(
            scanner_cfg.get(
                "n_bins", self._DEFAULT_N_BINS,
            ),
        )

    @staticmethod
    def _gen_test(
        max_ece: float, n_bins: int,
    ) -> str:
        """Generate a pytest snippet for calibration."""
        return (
            "def test_calibration():\n"
            "    \"\"\"Predicted probabilities must"
            " be well-calibrated.\"\"\"\n"
            "    y_prob = model.predict_proba("
            "X_test)[:, 1]\n"
            "    assert_calibration(\n"
            "        y_test, y_prob,\n"
            f"        max_error={max_ece},\n"
            f"        n_bins={n_bins},\n"
            "    )\n"
        )
