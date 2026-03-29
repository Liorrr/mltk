from __future__ import annotations

"""Tests for mltk.scan.scanners.bias -- BiasScanner.

BiasScanner checks for demographic parity violations and
other fairness issues across sensitive groups.  Tests cover:
- Detection of biased predictions
- Clean model producing no bias findings
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for bias scanner tests",
)

try:
    from mltk.scan.scanners.bias import BiasScanner
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_BIAS = True
except ImportError:
    _HAS_BIAS = False

pytestmark = pytest.mark.skipif(
    not _HAS_BIAS,
    reason="mltk.scan.scanners.bias not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(
    model_fn, X, y,
    sensitive_columns=None,
    **config_kw,
):
    """Build a ScanContext for the bias scanner."""
    cfg = ScanConfig(**config_kw)
    sens = sensitive_columns or []
    numeric = [
        c for c in X.columns
        if X[c].dtype.kind in ("f", "i")
        and X[c].nunique() > cfg.categorical_threshold
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
        y_train=None,
        X_train=None,
        sensitive_columns=sens,
        numeric_columns=numeric,
        categorical_columns=categorical,
        model_type="classifier",
        config=cfg,
        seed=42,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestBiasScannerDetection:
    """BiasScanner finds demographic parity violations."""

    def test_finds_bias_in_biased_model(
        self, biased_model,
    ) -> None:
        """Biased model triggers bias findings."""
        model_fn, X, y, sensitive = biased_model
        ctx = _build_context(
            model_fn, X, y,
            sensitive_columns=sensitive,
        )
        scanner = BiasScanner()
        findings = scanner.scan(ctx)
        assert len(findings) > 0

    def test_no_findings_for_fair_model(self) -> None:
        """A fair model produces no bias findings."""
        rng = np.random.default_rng(42)
        n = 400
        gender = rng.choice([0, 1], size=n)
        # Label is independent of gender
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "gender": gender,
            "feat": rng.normal(0, 1, n),
        })
        # Predictions are also independent of gender
        preds = y.copy()

        def model_fn(x):
            return preds[x.index]

        ctx = _build_context(
            model_fn, X, y,
            sensitive_columns=["gender"],
        )
        scanner = BiasScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_no_sensitive_columns_skips(self) -> None:
        """Scanner produces no findings when no sensitive
        columns are provided."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "a": rng.normal(0, 1, n),
        })
        y = rng.integers(0, 2, n)

        def model_fn(x):
            return y[x.index]

        ctx = _build_context(
            model_fn, X, y,
            sensitive_columns=[],
        )
        scanner = BiasScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0
