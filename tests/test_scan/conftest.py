from __future__ import annotations

"""Adversarial benchmark fixtures for mltk scan tests.

Six fixtures covering the key failure modes that scanners
must detect: bias, leakage, overfitting, miscalibration,
fragility, and a clean baseline.  All use sklearn models
with fixed seed 42 for reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip(
    "sklearn",
    reason="sklearn required for scan fixtures",
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------
# Fixture 1: Biased model (gender correlates with outcome)
# ---------------------------------------------------------------


@pytest.fixture
def biased_model():
    """Model that correlates predictions with gender.

    Training data makes gender strongly predictive of the
    label, so the fitted model inherits demographic bias.
    Returns (model_fn, X_test, y_test, sensitive_cols).
    """
    rng = np.random.default_rng(42)
    n = 400
    gender = rng.choice([0, 1], size=n)
    feat = rng.normal(0, 1, n)
    # Label is ~90% determined by gender
    y = (gender + rng.normal(0, 0.3, n) > 0.5).astype(int)
    X = pd.DataFrame({
        "gender": gender,
        "feat": feat,
    })
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    return clf.predict, X, y, ["gender"]


# ---------------------------------------------------------------
# Fixture 2: Leaky data (feature derived from target)
# ---------------------------------------------------------------


@pytest.fixture
def leaky_data():
    """Dataset where one feature is a function of the target.

    ``future_revenue`` is target + tiny noise, creating a
    near-perfect correlation that indicates data leakage.
    Returns (model_fn, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 300
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "legit_feat": rng.normal(0, 1, n),
        "future_revenue": y + rng.normal(0, 0.01, n),
    })
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    return clf.predict, X, y


# ---------------------------------------------------------------
# Fixture 3: Overfit model (99% train, ~60% test)
# ---------------------------------------------------------------


@pytest.fixture
def overfit_model():
    """Model with 99% train accuracy but ~60% test accuracy.

    A deep decision tree memorises noise in the training set
    but cannot generalise to held-out data.
    Returns (model_fn, X_train, y_train, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 200
    X_train = pd.DataFrame({
        "a": rng.normal(0, 1, n_train),
        "b": rng.normal(0, 1, n_train),
        "c": rng.normal(0, 1, n_train),
    })
    y_train = rng.integers(0, 2, n_train)
    X_test = pd.DataFrame({
        "a": rng.normal(0, 1, n_test),
        "b": rng.normal(0, 1, n_test),
        "c": rng.normal(0, 1, n_test),
    })
    y_test = rng.integers(0, 2, n_test)
    clf = DecisionTreeClassifier(
        max_depth=None, random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf.predict, X_train, y_train, X_test, y_test


# ---------------------------------------------------------------
# Fixture 4: Miscalibrated model (predict_proba unreliable)
# ---------------------------------------------------------------


@pytest.fixture
def miscalibrated_model():
    """Model whose predict_proba values are unreliable.

    The model always outputs extreme probabilities (>0.95 or
    <0.05) even when its predictions are wrong ~20% of the
    time.
    Returns (predict_fn, predict_proba_fn, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 300
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
    })
    clf = DecisionTreeClassifier(
        max_depth=1, random_state=42,
    )
    clf.fit(X, y)

    def fake_proba(x):
        """Return extreme probabilities regardless."""
        preds = clf.predict(x)
        proba = np.where(
            preds == 1,
            rng.uniform(0.95, 0.99, len(x)),
            rng.uniform(0.01, 0.05, len(x)),
        )
        return np.column_stack([1 - proba, proba])

    return clf.predict, fake_proba, X, y


# ---------------------------------------------------------------
# Fixture 5: Fragile model (flips on tiny noise)
# ---------------------------------------------------------------


@pytest.fixture
def fragile_model():
    """Model that flips predictions on tiny input noise.

    A shallow tree on near-boundary data means even 1%
    noise changes many predictions.
    Returns (model_fn, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 400
    # Features clustered around the decision boundary
    X = pd.DataFrame({
        "x1": rng.uniform(-0.1, 0.1, n),
        "x2": rng.uniform(-0.1, 0.1, n),
    })
    y = (X["x1"] + X["x2"] > 0).astype(int).values
    clf = DecisionTreeClassifier(
        max_depth=2, random_state=42,
    )
    clf.fit(X, y)
    return clf.predict, X, y


# ---------------------------------------------------------------
# Fixture 6: Clean model (no issues expected)
# ---------------------------------------------------------------


@pytest.fixture
def clean_model():
    """Well-behaved model -- scan should find zero findings.

    Balanced data, no sensitive features, strong separation,
    and a well-regularised classifier.
    Returns (model_fn, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 400
    # Features with moderate signal — enough to learn but not
    # directly correlated with target (avoids leakage detection)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    # Target based on a nonlinear combination + noise
    logits = 0.5 * f1 + 0.3 * f2 + rng.normal(0, 0.5, n)
    y = (logits > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": f2})
    clf = LogisticRegression(random_state=42, C=1.0)
    clf.fit(X, y)
    return clf.predict, X, y
