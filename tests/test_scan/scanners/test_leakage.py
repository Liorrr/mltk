from __future__ import annotations

"""Tests for mltk.scan.scanners.leakage -- LeakageScanner.

LeakageScanner detects features that have suspiciously high
correlation with the target, indicating data leakage.  Tests
cover:
- Detection of a leaky (near-perfect correlation) feature
- Clean data with no leaky features
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.scanners.leakage import LeakageScanner
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_LEAKAGE = True
except ImportError:
    _HAS_LEAKAGE = False

pytestmark = pytest.mark.skipif(
    not _HAS_LEAKAGE,
    reason=(
        "mltk.scan.scanners.leakage not yet implemented"
    ),
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(X, y, **config_kw):
    """Build a ScanContext for the leakage scanner."""
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


class TestLeakageScannerDetection:
    """LeakageScanner finds high-correlation features."""

    def test_finds_leaky_feature(self) -> None:
        """Feature derived from target triggers leakage
        finding."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "legit": rng.normal(0, 1, n),
            "leaky": y + rng.normal(0, 0.01, n),
        })
        ctx = _build_context(X, y)
        scanner = LeakageScanner()
        findings = scanner.scan(ctx)
        assert len(findings) > 0
        # At least one finding should reference the
        # leaky column
        msgs = " ".join(
            str(f.result.message) for f in findings
        )
        details = " ".join(
            str(getattr(f.result, "details", {}))
            for f in findings
        )
        combined = (msgs + details).lower()
        assert "leaky" in combined

    def test_no_findings_for_clean_data(self) -> None:
        """Independent features produce no leakage
        findings."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.uniform(0, 1, n),
        })
        ctx = _build_context(X, y)
        scanner = LeakageScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_moderate_correlation_no_finding(self) -> None:
        """Moderately correlated feature (r~0.5) is not
        flagged as leakage."""
        rng = np.random.default_rng(42)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        noise = rng.normal(0, 1, n)
        X = pd.DataFrame({
            "moderate": y * 0.5 + noise * 0.5,
            "clean": rng.normal(0, 1, n),
        })
        ctx = _build_context(X, y.astype(int))
        scanner = LeakageScanner()
        findings = scanner.scan(ctx)
        # Moderate correlation should NOT trigger
        leaky_findings = [
            f for f in findings
            if "moderate" in str(
                getattr(f.result, "details", {}),
            ).lower()
            or "moderate" in str(
                f.result.message,
            ).lower()
        ]
        assert len(leaky_findings) == 0
