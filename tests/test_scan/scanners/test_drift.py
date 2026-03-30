from __future__ import annotations

"""Tests for mltk.scan.scanners.drift -- DriftScanner.

DriftScanner detects distribution shifts by splitting each
numeric column into reference and current halves. Tests cover:
- Detection of shifted distributions
- Clean distributions producing no findings
- Respect for scanner config overrides
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.scanners.drift import DriftScanner
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_DRIFT = True
except ImportError:
    _HAS_DRIFT = False

pytestmark = pytest.mark.skipif(
    not _HAS_DRIFT,
    reason="mltk.scan.scanners.drift not available",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(X, **config_kw):
    """Build a ScanContext for the drift scanner."""
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
        predict_proba_fn=None,
        X=X,
        y=None,
        y_train=None,
        X_train=None,
        sensitive_columns=[],
        numeric_columns=numeric,
        categorical_columns=categorical,
        model_type="unknown",
        config=cfg,
        seed=42,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestDriftScannerDetection:
    """DriftScanner detects distribution shifts."""

    def test_finds_drift_in_shifted_column(self) -> None:
        """Detects drift when second half is shifted."""
        rng = np.random.default_rng(42)
        n = 400
        # First half: N(0,1), second half: N(5,1)
        first_half = rng.normal(0, 1, n // 2)
        second_half = rng.normal(5, 1, n // 2)
        values = np.concatenate(
            [first_half, second_half],
        )
        X = pd.DataFrame({"feature": values})
        ctx = _build_context(X)
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        assert len(findings) >= 1
        assert any(
            "feature" in f.result.details.get(
                "column", "",
            )
            for f in findings
        )

    def test_no_drift_stable_column(self) -> None:
        """Stable distribution yields no findings."""
        rng = np.random.default_rng(42)
        n = 400
        values = rng.normal(0, 1, n)
        X = pd.DataFrame({"feature": values})
        ctx = _build_context(X)
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_skips_short_columns(self) -> None:
        """Columns with fewer than 2 values are skipped."""
        X = pd.DataFrame({"feature": [1.0]})
        ctx = _build_context(X)
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0


class TestDriftScannerConfig:
    """DriftScanner respects configuration."""

    def test_custom_method(self) -> None:
        """Scanner reads method from config."""
        rng = np.random.default_rng(42)
        n = 400
        first_half = rng.normal(0, 1, n // 2)
        second_half = rng.normal(5, 1, n // 2)
        values = np.concatenate(
            [first_half, second_half],
        )
        X = pd.DataFrame({"feature": values})
        ctx = _build_context(
            X,
            scanner_config={
                "drift": {"method": "psi"},
            },
        )
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        # Should still detect the drift with PSI
        assert len(findings) >= 1


class TestDriftScannerMetadata:
    """DriftScanner findings have correct metadata."""

    def test_scanner_name(self) -> None:
        """Findings carry correct scanner name."""
        rng = np.random.default_rng(42)
        n = 400
        first_half = rng.normal(0, 1, n // 2)
        second_half = rng.normal(5, 1, n // 2)
        values = np.concatenate(
            [first_half, second_half],
        )
        X = pd.DataFrame({"feature": values})
        ctx = _build_context(X)
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.scanner_name == "drift"

    def test_finding_has_assertion_fn(self) -> None:
        """Findings carry assertion function."""
        rng = np.random.default_rng(42)
        n = 400
        first_half = rng.normal(0, 1, n // 2)
        second_half = rng.normal(5, 1, n // 2)
        values = np.concatenate(
            [first_half, second_half],
        )
        X = pd.DataFrame({"feature": values})
        ctx = _build_context(X)
        scanner = DriftScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.assertion_fn is not None
