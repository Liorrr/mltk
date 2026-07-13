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


class TestJensenShannonScale:
    """JS drift reports the normalized [0,1] scale shared with mltk._rust."""

    def test_statistic_matches_rust_js_divergence(self) -> None:
        """Regression: _drift_js computed its own unnormalized (nats)
        JS, ~1.44x smaller than mltk._rust.js_divergence for the same
        inputs — two different values for the same concept in one
        codebase. The assertion now delegates, so the reported
        statistic must equal the library function exactly.
        """
        from mltk._rust import js_divergence

        rng = np.random.default_rng(11)
        ref = pd.Series(rng.normal(0, 1, 400))
        cur = pd.Series(rng.normal(0.4, 1.3, 400))
        try:
            result = assert_no_drift(ref, cur, method="js", threshold=1.1)
        except MltkAssertionError as exc:
            result = exc.result
        expected = js_divergence(
            ref.to_numpy().tolist(), cur.to_numpy().tolist(), 10
        )
        assert result.details["statistic"] == pytest.approx(expected, rel=1e-12)

    def test_statistic_is_normalized(self) -> None:
        """PASS: disjoint distributions approach 1.0, never exceed it."""
        ref = pd.Series([0.0] * 100)
        cur = pd.Series([100.0] * 100)
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_drift(ref, cur, method="js")
        stat = exc.value.result.details["statistic"]
        assert 0.9 < stat <= 1.0
