from __future__ import annotations

"""Tests for mltk.scan.scanners.data -- DataScanner.

DataScanner checks data quality before any model evaluation.
Tests cover:
- Detection of null values in columns
- Detection of PII in string columns
- Clean data producing no findings
"""

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.scanners.data import DataScanner
    from mltk.scan.config import ScanConfig, ScanContext
    _HAS_DATA = True
except ImportError:
    _HAS_DATA = False

pytestmark = pytest.mark.skipif(
    not _HAS_DATA,
    reason="mltk.scan.scanners.data not available",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _build_context(X, **config_kw):
    """Build a ScanContext for the data scanner."""
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


class TestDataScannerNulls:
    """DataScanner detects null values."""

    def test_finds_nulls(self) -> None:
        """Detects columns with null values."""
        X = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        null_findings = [
            f for f in findings
            if "null" in f.result.name
        ]
        assert len(null_findings) >= 1
        assert any(
            "a" in f.result.details.get("column", "")
            for f in null_findings
        )

    def test_no_nulls_clean(self) -> None:
        """Clean data yields no null findings."""
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        null_findings = [
            f for f in findings
            if "null" in f.result.name
        ]
        assert len(null_findings) == 0

    def test_multiple_null_columns(self) -> None:
        """Detects nulls in multiple columns."""
        X = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "y": [np.nan, 2.0, 3.0],
            "z": [1.0, 2.0, 3.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        null_findings = [
            f for f in findings
            if "null" in f.result.name
        ]
        assert len(null_findings) >= 2


class TestDataScannerPII:
    """DataScanner detects PII in string columns."""

    def test_finds_email_pii(self) -> None:
        """Detects email addresses as PII."""
        X = pd.DataFrame({
            "notes": [
                "contact user@example.com",
                "no pii here",
                "plain text",
            ],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        pii_findings = [
            f for f in findings
            if "pii" in f.result.name
        ]
        assert len(pii_findings) >= 1

    def test_no_pii_clean(self) -> None:
        """Clean string data yields no PII findings."""
        X = pd.DataFrame({
            "notes": [
                "good data",
                "clean text",
                "no issues",
            ],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        pii_findings = [
            f for f in findings
            if "pii" in f.result.name
        ]
        assert len(pii_findings) == 0

    def test_skips_numeric_columns(self) -> None:
        """Numeric-only data skips PII checks."""
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        pii_findings = [
            f for f in findings
            if "pii" in f.result.name
        ]
        assert len(pii_findings) == 0


class TestDataScannerMetadata:
    """DataScanner findings have correct metadata."""

    def test_scanner_name(self) -> None:
        """Findings carry correct scanner name."""
        X = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.scanner_name == "data"

    def test_assertion_fn_set(self) -> None:
        """Findings carry an assertion function."""
        X = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
        })
        ctx = _build_context(X)
        scanner = DataScanner()
        findings = scanner.scan(ctx)
        for f in findings:
            assert f.assertion_fn is not None
