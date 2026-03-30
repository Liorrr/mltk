from __future__ import annotations

"""Tests for mltk.scan.scanners.calibration -- CalibrationScanner.

CalibrationScanner checks whether predicted probabilities match
observed frequencies. Tests cover:
- Detection of miscalibrated probabilities
- Well-calibrated models producing no findings
- Correct extraction of positive-class probabilities
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.scanners.calibration import (
        CalibrationScanner,
    )
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_CALIBRATION = True
except ImportError:
    _HAS_CALIBRATION = False

pytestmark = pytest.mark.skipif(
    not _HAS_CALIBRATION,
    reason=(
        "mltk.scan.scanners.calibration not available"
    ),
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(
    predict_proba_fn, X, y, **config_kw,
):
    """Build a ScanContext for the calibration scanner."""
    cfg = ScanConfig(**config_kw)
    numeric = [
        c for c in X.columns
        if X[c].dtype.kind in ("f", "i")
    ]
    categorical = [
        c for c in X.columns
        if c not in numeric
    ]
    return ScanContext(
        model_fn=None,
        predict_proba_fn=predict_proba_fn,
        X=X,
        y=y,
        y_train=None,
        X_train=None,
        sensitive_columns=[],
        numeric_columns=numeric,
        categorical_columns=categorical,
        model_type="classifier",
        config=cfg,
        seed=42,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestCalibrationScannerDetection:
    """CalibrationScanner detects miscalibration."""

    def test_finds_miscalibration(self) -> None:
        """Detects when probabilities do not match reality."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })

        def bad_proba(x):
            """Always predict high confidence regardless
            of true label -- systematically overconfident.
            """
            # Always say 0.95 positive, even when y=0
            probs = np.full(len(y), 0.95)
            return np.column_stack(
                [1 - probs, probs],
            )

        ctx = _build_context(bad_proba, X, y)
        scanner = CalibrationScanner()
        findings = scanner.scan(ctx)
        assert len(findings) >= 1

    def test_well_calibrated_no_findings(self) -> None:
        """Well-calibrated model yields no findings."""
        rng = np.random.default_rng(42)
        n = 500
        # Generate well-calibrated probabilities
        true_prob = rng.uniform(0, 1, n)
        y = (
            rng.random(n) < true_prob
        ).astype(int)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })

        def good_proba(x):
            """Return calibrated probabilities."""
            return np.column_stack(
                [1 - true_prob, true_prob],
            )

        ctx = _build_context(good_proba, X, y)
        scanner = CalibrationScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_handles_1d_proba(self) -> None:
        """Handles predict_proba returning 1D array."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })

        def proba_1d(x):
            """Return 1D probabilities (extreme)."""
            return np.where(
                y == 1,
                rng.uniform(0.95, 0.99, len(y)),
                rng.uniform(0.01, 0.05, len(y)),
            )

        ctx = _build_context(proba_1d, X, y)
        scanner = CalibrationScanner()
        # Should not crash on 1D input
        findings = scanner.scan(ctx)
        assert isinstance(findings, list)


class TestCalibrationScannerMetadata:
    """CalibrationScanner findings have correct metadata."""

    def test_scanner_name(self) -> None:
        """Findings carry correct scanner name."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })

        def bad_proba(x):
            probs = np.where(
                y == 1, 0.99, 0.01,
            )
            return np.column_stack(
                [1 - probs, probs],
            )

        ctx = _build_context(bad_proba, X, y)
        scanner = CalibrationScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.scanner_name == "calibration"

    def test_finding_has_suggested_test(self) -> None:
        """Findings carry a suggested test string."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })

        def bad_proba(x):
            probs = np.where(
                y == 1, 0.99, 0.01,
            )
            return np.column_stack(
                [1 - probs, probs],
            )

        ctx = _build_context(bad_proba, X, y)
        scanner = CalibrationScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert len(f.suggested_test) > 0
