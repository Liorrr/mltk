"""Tests for mltk.model.adversarial -- robustness under input perturbations.

Each test simulates a model that is either robust (stable under noise)
or fragile (predictions flip with tiny perturbations).
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.adversarial import assert_robust


def _stable_model(x: np.ndarray) -> np.ndarray:
    """A robust model: predictions based on sign of sum of features."""
    return (x.sum(axis=1) > 0).astype(int)


def _fragile_model(x: np.ndarray) -> np.ndarray:
    """A fragile model: prediction flips on tiny changes near boundary."""
    # Uses a very tight boundary — any noise flips the prediction
    return (x[:, 0] > 0.5).astype(int)


class TestAssertRobust:
    """Tests for adversarial robustness."""

    def test_stable_model(self) -> None:
        """PASS: Robust model maintains predictions under small noise.

        Scenario: Model uses sum of all features — small per-feature
        noise doesn't change the overall direction.
        """
        rng = np.random.default_rng(42)
        inputs = rng.normal(0, 5, (100, 10))  # Large margin from boundary
        result = assert_robust(_stable_model, inputs, epsilon=0.01, stability=0.95)
        assert result.passed is True
        assert result.details["stability_score"] >= 0.95

    def test_fragile_model(self) -> None:
        """FAIL: Fragile model changes predictions with tiny noise.

        Scenario: Model decision boundary is very tight. Points near
        0.5 flip prediction with epsilon=0.1 noise.
        """
        # All inputs near the decision boundary (0.5)
        inputs = np.full((100, 1), 0.5)
        with pytest.raises(MltkAssertionError) as exc:
            assert_robust(_fragile_model, inputs, epsilon=0.1, stability=0.95)
        assert "Fragile" in str(exc.value)

    def test_gaussian_perturbation(self) -> None:
        """Gaussian noise with custom epsilon applied correctly."""
        rng = np.random.default_rng(42)
        inputs = rng.normal(0, 10, (50, 5))
        result = assert_robust(_stable_model, inputs, perturbation="gaussian", epsilon=0.001)
        assert result.details["perturbation"] == "gaussian"

    def test_uniform_perturbation(self) -> None:
        """Uniform noise in [-epsilon, epsilon]."""
        rng = np.random.default_rng(42)
        inputs = rng.normal(0, 10, (50, 5))
        result = assert_robust(_stable_model, inputs, perturbation="uniform", epsilon=0.001)
        assert result.details["perturbation"] == "uniform"

    def test_unknown_perturbation(self) -> None:
        """FAIL: Invalid perturbation type raises error."""
        with pytest.raises(MltkAssertionError):
            assert_robust(_stable_model, np.array([[1.0]]), perturbation="invalid")

    def test_empty_inputs(self) -> None:
        """FAIL: Empty input handled gracefully."""
        with pytest.raises(MltkAssertionError):
            assert_robust(_stable_model, np.array([]))
