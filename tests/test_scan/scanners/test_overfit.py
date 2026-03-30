from __future__ import annotations

"""Tests for mltk.scan.scanners.overfit -- OverfitScanner.

OverfitScanner compares train vs test accuracy to detect
memorisation. Tests cover:
- Detection of large train-test gap
- Well-generalising models producing no findings
- Correct usage of conftest overfit_model fixture
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for overfit scanner tests",
)
from sklearn.tree import DecisionTreeClassifier

try:
    from mltk.scan.scanners.overfit import (
        OverfitScanner,
    )
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_OVERFIT = True
except ImportError:
    _HAS_OVERFIT = False

pytestmark = pytest.mark.skipif(
    not _HAS_OVERFIT,
    reason=(
        "mltk.scan.scanners.overfit not available"
    ),
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(
    model_fn, X, y,
    X_train=None, y_train=None,
    **config_kw,
):
    """Build a ScanContext for the overfit scanner."""
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
        model_fn=model_fn,
        predict_proba_fn=None,
        X=X,
        y=y,
        y_train=y_train,
        X_train=X_train,
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


class TestOverfitScannerDetection:
    """OverfitScanner detects train-test gap."""

    def test_finds_overfitting(
        self, overfit_model,
    ) -> None:
        """Detects large train-test accuracy gap."""
        model_fn, X_train, y_train, X_test, y_test = (
            overfit_model
        )
        ctx = _build_context(
            model_fn, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
        )
        scanner = OverfitScanner()
        findings = scanner.scan(ctx)
        assert len(findings) >= 1
        assert any(
            "overfit" in f.result.name.lower()
            or "overfit" in f.result.message.lower()
            for f in findings
        )

    def test_no_overfitting_good_model(self) -> None:
        """Well-generalising model yields no findings."""
        rng = np.random.default_rng(42)
        n = 200
        # Simple linearly separable data
        X_train = pd.DataFrame({
            "a": np.concatenate([
                rng.normal(-2, 0.5, n // 2),
                rng.normal(2, 0.5, n // 2),
            ]),
        })
        y_train = np.array(
            [0] * (n // 2) + [1] * (n // 2),
        )
        X_test = pd.DataFrame({
            "a": np.concatenate([
                rng.normal(-2, 0.5, n // 2),
                rng.normal(2, 0.5, n // 2),
            ]),
        })
        y_test = np.array(
            [0] * (n // 2) + [1] * (n // 2),
        )

        def simple_fn(X):
            arr = np.asarray(X)
            if arr.ndim == 2:
                return (arr[:, 0] > 0).astype(int)
            return (arr > 0).astype(int)

        ctx = _build_context(
            simple_fn, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
        )
        scanner = OverfitScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_custom_max_gap(self) -> None:
        """Respects custom max_gap from scanner config."""
        rng = np.random.default_rng(42)
        n_train, n_test = 200, 200
        X_train = pd.DataFrame({
            "a": rng.normal(0, 1, n_train),
            "b": rng.normal(0, 1, n_train),
        })
        y_train = rng.integers(0, 2, n_train)
        X_test = pd.DataFrame({
            "a": rng.normal(0, 1, n_test),
            "b": rng.normal(0, 1, n_test),
        })
        y_test = rng.integers(0, 2, n_test)
        clf = DecisionTreeClassifier(
            max_depth=None, random_state=42,
        )
        clf.fit(X_train, y_train)

        # Very strict max_gap should trigger finding
        ctx = _build_context(
            clf.predict, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
            scanner_config={
                "overfit": {"max_gap": 0.01},
            },
        )
        scanner = OverfitScanner()
        findings = scanner.scan(ctx)
        assert len(findings) >= 1


class TestOverfitScannerMetadata:
    """OverfitScanner findings have correct metadata."""

    def test_scanner_name(
        self, overfit_model,
    ) -> None:
        """Findings carry correct scanner name."""
        model_fn, X_train, y_train, X_test, y_test = (
            overfit_model
        )
        ctx = _build_context(
            model_fn, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
        )
        scanner = OverfitScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.scanner_name == "overfit"

    def test_finding_details_include_gap(
        self, overfit_model,
    ) -> None:
        """Finding details include gap information."""
        model_fn, X_train, y_train, X_test, y_test = (
            overfit_model
        )
        ctx = _build_context(
            model_fn, X_test, y_test,
            X_train=X_train,
            y_train=y_train,
        )
        scanner = OverfitScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert "gap" in f.result.details
            assert "train_score" in f.result.details
            assert "test_score" in f.result.details
