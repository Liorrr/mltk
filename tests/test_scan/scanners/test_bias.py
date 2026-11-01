"""Tests for mltk.scan.scanners.bias -- BiasScanner.

BiasScanner checks for demographic parity violations and
other fairness issues across sensitive groups.  Tests cover:
- Detection of biased predictions
- Clean model producing no bias findings
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._snippet_helpers import TinyModel, execute_finding_snippet

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for bias scanner tests",
)

try:
    from mltk.scan.config import ScanConfig, ScanContext
    from mltk.scan.scanners.bias import BiasScanner
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

    def test_auto_detected_bias_is_advisory_finding(self) -> None:
        """Detected sensitive columns retain violations at INFO severity."""
        gender = np.array(["M"] * 100 + ["F"] * 100)
        y = np.ones(200, dtype=int)
        preds = np.array([1] * 100 + [0] * 100)
        X = pd.DataFrame({"gender": gender, "feature": np.arange(200)})

        def model_fn(x):
            return preds[x.index]

        findings = BiasScanner().scan(_build_context(model_fn, X, y))
        gender_findings = [
            f for f in findings
            if f.result.details["column"] == "gender"
        ]
        assert gender_findings
        assert all(f.result.severity.value == "info" for f in gender_findings)

    def test_auto_detected_fair_column_has_no_finding(self) -> None:
        """Auto-detection does not report a fair sensitive column."""
        gender = np.array(["M"] * 100 + ["F"] * 100)
        y = np.array([1, 0] * 100)
        X = pd.DataFrame({"gender": gender, "feature": np.arange(200)})

        def model_fn(x):
            return y[x.index]

        findings = BiasScanner().scan(_build_context(model_fn, X, y))
        assert not findings

    def test_user_specified_bias_remains_critical(self) -> None:
        """Explicit sensitive columns keep the critical policy."""
        gender = np.array(["M"] * 100 + ["F"] * 100)
        y = np.ones(200, dtype=int)
        preds = np.array([1] * 100 + [0] * 100)
        X = pd.DataFrame({"gender": gender, "feature": np.arange(200)})

        def model_fn(x):
            return preds[x.index]

        findings = BiasScanner().scan(
            _build_context(model_fn, X, y, sensitive_columns=["gender"]),
        )
        assert findings
        assert any(f.result.severity.value == "critical" for f in findings)

    def test_suggested_body_contract_and_execution(self) -> None:
        """A critical bias finding emits an executable test body."""
        gender = np.array(["M"] * 100 + ["F"] * 100)
        y = np.ones(200, dtype=int)
        preds = np.array([1] * 100 + [0] * 100)
        X = pd.DataFrame({"gender": gender, "feature": np.arange(200)})

        def model_fn(x):
            return preds[x.index]

        findings = BiasScanner().scan(
            _build_context(model_fn, X, y, sensitive_columns=["gender"]),
        )
        execute_finding_snippet(
            findings[0],
            X,
            y,
            model=TinyModel(preds),
        )
