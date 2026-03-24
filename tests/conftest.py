"""Shared test fixtures for mltk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
