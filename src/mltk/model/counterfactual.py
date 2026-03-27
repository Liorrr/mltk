"""Counterfactual fairness testing -- would the prediction change if a person's
protected attribute were different?

Standard fairness metrics (demographic parity, equalized odds) measure AGGREGATE
disparities: "does group A get approved more often than group B?" But aggregate
metrics can hide individual-level discrimination. A model might achieve perfect
demographic parity while still using gender to decide SPECIFIC cases -- as long
as the errors balance out across the population.

Counterfactual fairness asks a sharper question for EACH individual:
    "If THIS person had a different gender/race/age -- but everything else
     stayed the same -- would the model's prediction change?"

Example: A loan model approves John (male, 30, $80K salary). We create a
counterfactual twin: Jane (female, 30, $80K salary) -- identical except for
the protected attribute. If the model now denies Jane, it is causally using
gender in its decision, even if aggregate metrics look clean.

The flip rate is the fraction of individuals whose prediction changes under
the counterfactual intervention. A fair model should have a flip rate near zero.

Reference: Kusner et al., "Counterfactual Fairness" (NeurIPS 2017).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _default_perturbation(
    X: np.ndarray, sensitive_col: int
) -> np.ndarray:
    """Flip the sensitive column: binary 0<->1, categorical cycles through unique values.

    For binary features (exactly two unique values), each sample's sensitive
    attribute is swapped to the other value. For categorical features with k>2
    unique values, each sample is shifted to the next value in sorted order
    (wrapping around), guaranteeing every sample receives a different value.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        sensitive_col: Column index of the protected attribute.

    Returns:
        A copy of X with the sensitive column perturbed.
    """
    X_perturbed = X.copy()
    col = X_perturbed[:, sensitive_col]
    unique_vals = np.unique(col)

    if len(unique_vals) <= 1:
        # Cannot perturb a constant column -- return unchanged copy.
        return X_perturbed

    if len(unique_vals) == 2:
        # Binary flip: 0->1, 1->0 (works for any two distinct values).
        val_a, val_b = unique_vals
        new_col = np.where(col == val_a, val_b, val_a)
    else:
        # Categorical cycle: shift each value to the next in sorted order.
        val_to_next = {
            unique_vals[i]: unique_vals[(i + 1) % len(unique_vals)]
            for i in range(len(unique_vals))
        }
        new_col = np.array([val_to_next[v] for v in col], dtype=col.dtype)

    X_perturbed[:, sensitive_col] = new_col
    return X_perturbed


@timed_assertion
def assert_counterfactual_fairness(
    model_fn: Callable[..., Any],
    X: np.ndarray,
    sensitive_col: int,
    perturbation_fn: Callable[..., np.ndarray] | None = None,
    max_flip_rate: float = 0.05,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a model's predictions do not change when the protected
    attribute is perturbed -- i.e., the model is counterfactually fair.

    How it works:
        1. Get original predictions: y_orig = model_fn(X)
        2. Perturb ONLY the sensitive column (flip gender, change race, etc.)
        3. Get counterfactual predictions: y_cf = model_fn(X_perturbed)
        4. Count how many predictions flipped: flip_rate = mean(y_orig != y_cf)
        5. Pass if flip_rate <= max_flip_rate

    A high flip rate means the model is causally using the protected attribute.
    Even a model with perfect aggregate fairness metrics can fail this test.

    Args:
        model_fn: Callable that takes X (ndarray) and returns predictions (ndarray).
            Must accept a 2-D array and return a 1-D array of predictions.
        X: Feature matrix of shape (n_samples, n_features).
        sensitive_col: Column index of the protected attribute in X.
        perturbation_fn: Optional custom perturbation function.
            Signature: (X: ndarray) -> X_perturbed: ndarray.
            If None, uses the default binary flip / categorical cycle.
        max_flip_rate: Maximum allowed fraction of predictions that change
            after perturbation. Default 0.05 (5%). Lower = stricter.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with details: flip_rate, max_flip_rate, n_flipped, n_total,
        sensitive_col.

    Example:
        >>> import numpy as np
        >>> # Fair model: ignores sensitive column entirely
        >>> model = lambda X: (X[:, 1] > 0.5).astype(int)
        >>> X = np.column_stack([np.array([0, 1, 0, 1]), np.array([0.8, 0.3, 0.9, 0.2])])
        >>> assert_counterfactual_fairness(model, X, sensitive_col=0)
    """
    X = np.asarray(X)

    if X.ndim != 2:
        return assert_true(
            False,
            name="model.counterfactual_fairness",
            message="X must be a 2-D array (n_samples, n_features)",
            severity=severity,
        )

    n_total = X.shape[0]
    if n_total == 0:
        return assert_true(
            False,
            name="model.counterfactual_fairness",
            message="Cannot evaluate counterfactual fairness on empty data",
            severity=severity,
        )

    # Step 1: Original predictions.
    y_orig = np.asarray(model_fn(X))

    # Step 2: Perturb sensitive attribute.
    if perturbation_fn is not None:
        X_perturbed = perturbation_fn(X)
    else:
        X_perturbed = _default_perturbation(X, sensitive_col)

    # Step 3: Counterfactual predictions.
    y_cf = np.asarray(model_fn(X_perturbed))

    # Step 4: Compute flip rate.
    n_flipped = int(np.sum(y_orig != y_cf))
    flip_rate = n_flipped / n_total

    # Step 5: Check threshold.
    passed = flip_rate <= max_flip_rate
    message = (
        f"Counterfactual flip_rate={flip_rate:.4f} <= {max_flip_rate} "
        f"({n_flipped}/{n_total} predictions changed)"
        if passed
        else f"Counterfactual fairness violated: flip_rate={flip_rate:.4f} > "
        f"{max_flip_rate} ({n_flipped}/{n_total} predictions changed)"
    )

    return assert_true(
        passed,
        name="model.counterfactual_fairness",
        message=message,
        severity=severity,
        flip_rate=flip_rate,
        max_flip_rate=max_flip_rate,
        n_flipped=n_flipped,
        n_total=n_total,
        sensitive_col=sensitive_col,
    )
