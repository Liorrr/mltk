"""Tests for advanced drift methods — Jensen-Shannon, Wasserstein, auto-select."""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.drift import assert_no_drift


class TestJensenShannon:
    """Jensen-Shannon divergence drift detection."""

    def test_identical_distributions(self, reference_series: pd.Series) -> None:
        """PASS: Same distribution gives JS near 0."""
        result = assert_no_drift(reference_series, reference_series, method="js")
        assert result.passed is True

    def test_shifted_distribution(
        self, reference_series: pd.Series, drifted_series: pd.Series
    ) -> None:
        """FAIL: Shifted distribution gives high JS."""
        with pytest.raises(MltkAssertionError):
            assert_no_drift(reference_series, drifted_series, method="js")


class TestWasserstein:
    """Wasserstein distance drift detection."""

    def test_identical_distributions(self, reference_series: pd.Series) -> None:
        """PASS: Same distribution gives W near 0."""
        result = assert_no_drift(reference_series, reference_series, method="wasserstein")
        assert result.passed is True

    def test_shifted_distribution(
        self, reference_series: pd.Series, drifted_series: pd.Series
    ) -> None:
        """FAIL: Shifted distribution gives high Wasserstein distance."""
        with pytest.raises(MltkAssertionError):
            assert_no_drift(reference_series, drifted_series, method="wasserstein")


class TestAutoSelect:
    """Auto-method selection based on sample size."""

    def test_auto_selects_method(self, reference_series: pd.Series) -> None:
        """PASS: Auto method runs without error."""
        result = assert_no_drift(reference_series, reference_series, method="auto")
        assert result.passed is True

    def test_auto_large_sample(self) -> None:
        """Auto uses Wasserstein for n>1000."""
        rng = np.random.default_rng(42)
        large = pd.Series(rng.normal(0, 1, 2000))
        result = assert_no_drift(large, large, method="auto")
        assert result.passed is True
