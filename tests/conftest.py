"""Shared test fixtures for mltk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --- Data fixtures (Sprint 1-2) ---


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(100),
            "feature_a": rng.normal(0, 1, 100),
            "feature_b": rng.integers(0, 10, 100),
            "label": rng.integers(0, 2, 100),
        }
    )


@pytest.fixture
def reference_series() -> pd.Series:
    """Reference distribution for drift testing."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 1000), name="feature")


@pytest.fixture
def drifted_series() -> pd.Series:
    """Drifted distribution for drift testing."""
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(2, 1.5, 1000), name="feature")


# --- Model fixtures (Sprint 3) ---


@pytest.fixture
def binary_classification() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary classification results: (y_true, y_pred, y_prob).

    A decent classifier with ~85% accuracy on balanced binary data.
    """
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, n)
    # Make predictions mostly correct with some noise
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    # Probabilities roughly aligned with predictions
    y_prob = np.where(y_pred == 1, rng.uniform(0.6, 0.95, n), rng.uniform(0.05, 0.4, n))
    return y_true, y_pred, y_prob


@pytest.fixture
def multiclass_classification() -> tuple[np.ndarray, np.ndarray]:
    """Multiclass classification results: (y_true, y_pred).

    3-class classifier with ~80% accuracy.
    """
    rng = np.random.default_rng(42)
    n = 300
    y_true = rng.integers(0, 3, n)
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=int(n * 0.2), replace=False)
    y_pred[flip_idx] = rng.integers(0, 3, len(flip_idx))
    return y_true, y_pred


@pytest.fixture
def regression_results() -> tuple[np.ndarray, np.ndarray]:
    """Regression results: (y_true, y_pred).

    Reasonable regression with noise.
    """
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.normal(100, 20, n)
    y_pred = y_true + rng.normal(0, 5, n)  # predictions with small noise
    return y_true, y_pred
