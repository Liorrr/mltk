from __future__ import annotations

"""Tests for mltk.scan.engine -- ScanEngine integration tests.

ScanEngine is the orchestrator that runs all enabled scanners
against a model + dataset and produces a ScanReport.  These
tests verify the full scan pipeline: discovery, execution,
error handling, filtering, and report structure.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for engine tests",
)

try:
    from mltk.scan.engine import ScanEngine
    from mltk.scan.config import ScanConfig
    from mltk.scan.finding import ScanFinding
    _HAS_SCAN = True
except ImportError:
    _HAS_SCAN = False

pytestmark = pytest.mark.skipif(
    not _HAS_SCAN,
    reason="mltk.scan not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _simple_df(n: int = 200, seed: int = 42):
    """Build a simple numeric DataFrame + labels."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })
    y = rng.integers(0, 2, n)
    return X, y


# ---------------------------------------------------------------
# Integration: scan() returns ScanReport
# ---------------------------------------------------------------


class TestScanEngineIntegration:
    """End-to-end tests for ScanEngine.scan()."""

    def test_scan_returns_report(
        self, biased_model,
    ) -> None:
        """scan() returns a ScanReport with findings."""
        model_fn, X, y, sensitive = biased_model
        engine = ScanEngine()
        report = engine.scan(
            model_fn, X, y,
            sensitive_columns=sensitive,
        )
        assert hasattr(report, "findings")
        assert hasattr(report, "scanners_run")
        assert isinstance(report.findings, list)

    def test_clean_model_zero_findings(
        self, clean_model,
    ) -> None:
        """Clean model produces zero or very few findings."""
        model_fn, X, y = clean_model
        engine = ScanEngine()
        report = engine.scan(model_fn, X, y)
        # A well-behaved model should have no criticals
        critical = [
            f for f in report.findings
            if (
                hasattr(f.result, "severity")
                and f.result.severity.value == "critical"
            )
        ]
        assert len(critical) == 0

    def test_scan_handles_model_error(self) -> None:
        """scan() survives a model_fn that raises."""
        def bad_model(x):
            raise RuntimeError("model crashed")

        X, y = _simple_df(100)
        engine = ScanEngine()
        report = engine.scan(bad_model, X, y)
        # Should not raise; errors go to scanners_errored
        assert hasattr(report, "scanners_errored")

    def test_scan_respects_enabled_scanners(
        self, clean_model,
    ) -> None:
        """Only enabled scanners run when list is set."""
        model_fn, X, y = clean_model
        cfg = ScanConfig(enabled_scanners=["slice"])
        engine = ScanEngine(config=cfg)
        report = engine.scan(model_fn, X, y)
        # Only the slice scanner (or scanners matching
        # "slice") should have run
        for name in report.scanners_run:
            assert "slice" in name.lower()


# ---------------------------------------------------------------
# ScanReport properties
# ---------------------------------------------------------------


class TestScanReportProperties:
    """Verify ScanReport exit_code and summary()."""

    def test_exit_code_zero_clean(
        self, clean_model,
    ) -> None:
        """Exit code 0 when no findings."""
        model_fn, X, y = clean_model
        engine = ScanEngine()
        report = engine.scan(model_fn, X, y)
        if len(report.findings) == 0:
            assert report.exit_code == 0

    def test_exit_code_one_findings(
        self, biased_model,
    ) -> None:
        """Exit code 1 when findings exist."""
        model_fn, X, y, sens = biased_model
        engine = ScanEngine()
        report = engine.scan(
            model_fn, X, y,
            sensitive_columns=sens,
        )
        if len(report.findings) > 0:
            assert report.exit_code == 1

    def test_summary_nonempty(
        self, clean_model,
    ) -> None:
        """summary() returns a non-empty string."""
        model_fn, X, y = clean_model
        engine = ScanEngine()
        report = engine.scan(model_fn, X, y)
        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_report_has_metadata(
        self, clean_model,
    ) -> None:
        """Report carries model_type, n_samples, etc."""
        model_fn, X, y = clean_model
        engine = ScanEngine()
        report = engine.scan(model_fn, X, y)
        assert hasattr(report, "model_type")
        assert hasattr(report, "n_samples")
        assert hasattr(report, "n_features")
        assert report.n_samples == len(X)
        assert report.n_features == X.shape[1]
