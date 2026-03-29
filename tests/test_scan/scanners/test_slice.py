from __future__ import annotations

"""Tests for mltk.scan.scanners.slice -- SliceScanner.

SliceScanner detects subgroups where model performance drops
significantly below the overall metric.  Tests cover:
- Detection of underperforming slices
- Respect for min_slice_samples threshold
- Clean models producing no findings
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for slice scanner tests",
)
from sklearn.tree import DecisionTreeClassifier

try:
    from mltk.scan.scanners.slice import SliceScanner
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_SLICE = True
except ImportError:
    _HAS_SLICE = False

pytestmark = pytest.mark.skipif(
    not _HAS_SLICE,
    reason="mltk.scan.scanners.slice not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(
    model_fn, X, y, **config_kw,
):
    """Build a ScanContext for the slice scanner."""
    cfg = ScanConfig(**config_kw)
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


class TestSliceScannerDetection:
    """SliceScanner finds underperforming subgroups."""

    def test_finds_weak_slice(self) -> None:
        """Detects a categorical slice with low accuracy."""
        rng = np.random.default_rng(42)
        n = 300
        group = np.array(
            ["good"] * 200 + ["bad"] * 100,
        )
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "group": group,
            "feat": rng.normal(0, 1, n),
        })
        # Build a model that is bad on "bad" group
        preds = y.copy()
        # Flip all "bad" group predictions
        bad_mask = group == "bad"
        preds[bad_mask] = 1 - preds[bad_mask]

        def model_fn(x):
            idx = x.index
            return preds[idx]

        ctx = _build_context(model_fn, X, y)
        scanner = SliceScanner()
        findings = scanner.scan(ctx)
        assert len(findings) > 0

    def test_respects_min_slice_samples(self) -> None:
        """Slices smaller than min_slice_samples are
        skipped."""
        rng = np.random.default_rng(42)
        n = 100
        # Only 5 samples in the "tiny" group
        group = np.array(
            ["big"] * 95 + ["tiny"] * 5,
        )
        y = rng.integers(0, 2, n)
        X = pd.DataFrame({
            "group": group,
            "feat": rng.normal(0, 1, n),
        })
        preds = y.copy()
        # Flip the tiny group (but it should be skipped)
        preds[95:] = 1 - preds[95:]

        def model_fn(x):
            idx = x.index
            return preds[idx]

        ctx = _build_context(
            model_fn, X, y, min_slice_samples=30,
        )
        scanner = SliceScanner()
        findings = scanner.scan(ctx)
        # Should NOT find tiny group (only 5 samples)
        tiny_findings = [
            f for f in findings
            if "tiny" in str(f.result.message).lower()
            or "tiny" in str(
                getattr(f.result, "details", {}),
            ).lower()
        ]
        assert len(tiny_findings) == 0

    def test_uniform_model_no_findings(
        self, clean_model,
    ) -> None:
        """A uniformly good model yields no slice findings."""
        model_fn, X, y = clean_model
        ctx = _build_context(model_fn, X, y)
        scanner = SliceScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0
