"""Tests for mltk.data.drift -- distribution drift detection.

Drift tests validate that production data still looks like training data.
Each test simulates a real-world scenario: identical distributions (no drift),
shifted distributions (drift detected), and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.drift import assert_no_drift


class TestKSDrift:
    """KS test drift detection -- best for continuous numeric features."""

    def test_identical_distributions_ks(
        self, reference_series: pd.Series
    ) -> None:
        """PASS: Same distribution compared to itself -- no drift.

        Scenario: Monthly data refresh, distribution hasn't changed.
        """
        result = assert_no_drift(reference_series, reference_series, method="ks")
        assert result.passed is True
        assert result.details["p_value"] > 0.05

    def test_shifted_distribution_ks(
        self, reference_series: pd.Series, drifted_series: pd.Series
    ) -> None:
        """FAIL: Mean-shifted distribution detected as drift.

        Scenario: Customer income distribution shifted significantly --
        model trained on old distribution will make wrong predictions.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_drift(reference_series, drifted_series, method="ks")
        assert "Drift detected" in str(exc.value)


class TestPSIDrift:
    """PSI drift detection -- industry standard for financial models."""

    def test_identical_distributions_psi(
        self, reference_series: pd.Series
    ) -> None:
        """PASS: Same distribution gives PSI near 0.

        Scenario: Stable feature distribution between training and serving.
        """
        result = assert_no_drift(reference_series, reference_series, method="psi")
        assert result.passed is True
        assert result.details["statistic"] < 0.01

    def test_shifted_distribution_psi(
        self, reference_series: pd.Series, drifted_series: pd.Series
    ) -> None:
        """FAIL: Shifted distribution gives PSI > threshold.

        Scenario: Credit score distribution shifted after economic event.
        PSI > 0.2 = significant change, model should be retrained.
        """
        with pytest.raises(MltkAssertionError):
            assert_no_drift(reference_series, drifted_series, method="psi")


class TestKLDrift:
    """KL divergence drift detection -- information-theoretic measure."""

    def test_identical_distributions_kl(
        self, reference_series: pd.Series
    ) -> None:
        """PASS: KL divergence near 0 for identical distributions."""
        result = assert_no_drift(reference_series, reference_series, method="kl")
        assert result.passed is True

    def test_shifted_distribution_kl(
        self, reference_series: pd.Series, drifted_series: pd.Series
    ) -> None:
        """FAIL: KL divergence high for shifted distributions."""
        with pytest.raises(MltkAssertionError):
            assert_no_drift(reference_series, drifted_series, method="kl")


class TestChi2Drift:
    """Chi-squared drift detection -- for categorical features."""

    def test_same_categorical_distribution(self) -> None:
        """PASS: Same category proportions -- no drift.

        Scenario: Product category distribution is stable month-over-month.
        """
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]))
        cur = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]))
        result = assert_no_drift(ref, cur, method="chi2")
        assert result.passed is True

    def test_shifted_categorical_distribution(self) -> None:
        """FAIL: Category proportions changed significantly.

        Scenario: Category 'C' suddenly dominates -- maybe a new product
        launch skewed the distribution. Model needs retraining.
        """
        ref = pd.Series(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        cur = pd.Series(["A"] * 50 + ["B"] * 50 + ["C"] * 200)
        with pytest.raises(MltkAssertionError):
            assert_no_drift(ref, cur, method="chi2")


class TestDriftEdgeCases:
    """Edge cases and configuration options."""

    def test_custom_threshold(self, reference_series: pd.Series) -> None:
        """Custom threshold overrides the default.

        Scenario: Your domain requires stricter drift detection (p > 0.1
        instead of default 0.05).
        """
        result = assert_no_drift(
            reference_series, reference_series, method="ks", threshold=0.01
        )
        assert result.passed is True

    def test_unknown_method(self, reference_series: pd.Series) -> None:
        """FAIL: Invalid method name raises clear error.

        Scenario: Typo in method name -- should fail fast, not silently.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_drift(reference_series, reference_series, method="invalid")
        assert "Unknown method" in str(exc.value)
