from __future__ import annotations

"""Tests for mltk.scan.scanners.robustness -- RobustnessScanner.

RobustnessScanner adds small gaussian noise to numeric features
and checks prediction stability. Tests cover:
- Detection of fragile models
- Robust models producing no findings
- Skipping when no numeric columns exist
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.scanners.robustness import (
        RobustnessScanner,
    )
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_ROBUSTNESS = True
except ImportError:
    _HAS_ROBUSTNESS = False

pytestmark = pytest.mark.skipif(
    not _HAS_ROBUSTNESS,
    reason=(
        "mltk.scan.scanners.robustness not available"
    ),
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(
    model_fn, X, y,
    numeric_columns=None,
    **config_kw,
):
    """Build a ScanContext for the robustness scanner."""
    cfg = ScanConfig(**config_kw)
    if numeric_columns is None:
        numeric_columns = [
            c for c in X.columns
            if X[c].dtype.kind in ("f", "i")
        ]
    categorical = [
        c for c in X.columns
        if c not in numeric_columns
    ]
    return ScanContext(
        model_fn=model_fn,
        predict_proba_fn=None,
        X=X,
        y=y,
        y_train=None,
        X_train=None,
        sensitive_columns=[],
        numeric_columns=numeric_columns,
        categorical_columns=categorical,
        model_type="classifier",
        config=cfg,
        seed=42,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestRobustnessScannerDetection:
    """RobustnessScanner detects fragile models."""

    def test_finds_fragile_model(self) -> None:
        """Detects model that flips on tiny noise."""
        rng = np.random.default_rng(42)
        n = 400
        X = pd.DataFrame({
            "x1": rng.uniform(-0.01, 0.01, n),
            "x2": rng.uniform(-0.01, 0.01, n),
        })
        y = (
            X["x1"] + X["x2"] > 0
        ).astype(int).values

        def fragile_fn(inputs):
            """Predict based on sum > 0."""
            arr = np.asarray(inputs)
            return (
                arr.sum(axis=1) > 0
            ).astype(int)

        ctx = _build_context(fragile_fn, X, y)
        scanner = RobustnessScanner()
        findings = scanner.scan(ctx)
        assert len(findings) >= 1

    def test_robust_model_no_findings(self) -> None:
        """Robust model yields no findings."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        })
        y = (
            X["x1"] + X["x2"] > 0
        ).astype(int).values

        def robust_fn(inputs):
            """Predict based on large margin."""
            arr = np.asarray(inputs)
            return (
                arr.sum(axis=1) > 0
            ).astype(int)

        ctx = _build_context(robust_fn, X, y)
        scanner = RobustnessScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0

    def test_skips_no_numeric_columns(self) -> None:
        """Returns empty when no numeric columns."""
        X = pd.DataFrame({
            "cat": ["a", "b", "c"],
        })
        y = np.array([0, 1, 0])

        def dummy_fn(inputs):
            return np.zeros(len(inputs))

        ctx = _build_context(
            dummy_fn, X, y,
            numeric_columns=[],
        )
        scanner = RobustnessScanner()
        findings = scanner.scan(ctx)
        assert len(findings) == 0


class TestRobustnessScannerMetadata:
    """RobustnessScanner findings have correct metadata."""

    def test_scanner_name(self) -> None:
        """Findings carry correct scanner name."""
        rng = np.random.default_rng(42)
        n = 400
        X = pd.DataFrame({
            "x1": rng.uniform(-0.01, 0.01, n),
            "x2": rng.uniform(-0.01, 0.01, n),
        })
        y = (
            X["x1"] + X["x2"] > 0
        ).astype(int).values

        def fragile_fn(inputs):
            arr = np.asarray(inputs)
            return (
                arr.sum(axis=1) > 0
            ).astype(int)

        ctx = _build_context(fragile_fn, X, y)
        scanner = RobustnessScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.scanner_name == "robustness"

    def test_finding_details_include_epsilon(
        self,
    ) -> None:
        """Finding details include epsilon value."""
        rng = np.random.default_rng(42)
        n = 400
        X = pd.DataFrame({
            "x1": rng.uniform(-0.01, 0.01, n),
            "x2": rng.uniform(-0.01, 0.01, n),
        })
        y = (
            X["x1"] + X["x2"] > 0
        ).astype(int).values

        def fragile_fn(inputs):
            arr = np.asarray(inputs)
            return (
                arr.sum(axis=1) > 0
            ).astype(int)

        ctx = _build_context(fragile_fn, X, y)
        scanner = RobustnessScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert "epsilon" in f.result.details
