from __future__ import annotations

"""Tests for mltk.scan.config -- ScanConfig and ScanContext.

ScanConfig holds scanner thresholds, timeouts, and sampling
parameters.  ScanContext bundles all inputs a scanner might
need.  These tests verify defaults match the plan and that
ScanContext.available_fields accurately reports populated
fields.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    from mltk.scan.config import ScanConfig, ScanContext
except ImportError:
    ScanConfig = None  # type: ignore[assignment,misc]
    ScanContext = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    ScanConfig is None,
    reason="mltk.scan.config not yet implemented",
)


# ---------------------------------------------------------------
# ScanConfig defaults
# ---------------------------------------------------------------


class TestScanConfigDefaults:
    """ScanConfig default values match the approved plan."""

    def test_max_scan_rows(self) -> None:
        cfg = ScanConfig()
        assert cfg.max_scan_rows == 10_000

    def test_seed(self) -> None:
        cfg = ScanConfig()
        assert cfg.seed == 42

    def test_sample_strategy(self) -> None:
        cfg = ScanConfig()
        assert cfg.sample_strategy == "stratified"

    def test_time_budget(self) -> None:
        cfg = ScanConfig()
        assert cfg.time_budget_seconds == 60.0

    def test_per_scanner_timeout(self) -> None:
        cfg = ScanConfig()
        assert cfg.per_scanner_timeout == 30.0

    def test_categorical_threshold(self) -> None:
        cfg = ScanConfig()
        assert cfg.categorical_threshold == 20

    def test_min_slice_samples(self) -> None:
        cfg = ScanConfig()
        assert cfg.min_slice_samples == 30

    def test_critical_drop(self) -> None:
        cfg = ScanConfig()
        assert cfg.critical_drop == pytest.approx(0.20)

    def test_warning_drop(self) -> None:
        cfg = ScanConfig()
        assert cfg.warning_drop == pytest.approx(0.10)

    def test_enabled_scanners_default_none(self) -> None:
        cfg = ScanConfig()
        assert cfg.enabled_scanners is None

    def test_disabled_scanners_default_none(self) -> None:
        cfg = ScanConfig()
        assert cfg.disabled_scanners is None

    def test_scanner_config_default_empty(self) -> None:
        cfg = ScanConfig()
        assert cfg.scanner_config == {}


# ---------------------------------------------------------------
# ScanContext.available_fields
# ---------------------------------------------------------------


class TestScanContextAvailableFields:
    """ScanContext.available_fields reports non-None fields."""

    def _make_context(self, **overrides):
        """Build a minimal ScanContext."""
        rng = np.random.default_rng(42)
        defaults = {
            "model_fn": lambda x: x,
            "predict_proba_fn": None,
            "X": pd.DataFrame({"a": rng.normal(0, 1, 10)}),
            "y": rng.integers(0, 2, 10),
            "y_train": None,
            "X_train": None,
            "sensitive_columns": [],
            "numeric_columns": ["a"],
            "categorical_columns": [],
            "model_type": "classifier",
            "config": ScanConfig(),
            "seed": 42,
        }
        defaults.update(overrides)
        return ScanContext(**defaults)

    def test_model_fn_present(self) -> None:
        ctx = self._make_context()
        assert "model_fn" in ctx.available_fields

    def test_proba_absent(self) -> None:
        ctx = self._make_context(predict_proba_fn=None)
        assert "predict_proba_fn" not in ctx.available_fields

    def test_proba_present(self) -> None:
        ctx = self._make_context(
            predict_proba_fn=lambda x: x,
        )
        assert "predict_proba_fn" in ctx.available_fields

    def test_train_data_absent(self) -> None:
        ctx = self._make_context(
            X_train=None, y_train=None,
        )
        fields = ctx.available_fields
        assert "X_train" not in fields
        assert "y_train" not in fields

    def test_train_data_present(self) -> None:
        rng = np.random.default_rng(42)
        ctx = self._make_context(
            X_train=pd.DataFrame({"a": [1, 2]}),
            y_train=np.array([0, 1]),
        )
        fields = ctx.available_fields
        assert "X_train" in fields
        assert "y_train" in fields
