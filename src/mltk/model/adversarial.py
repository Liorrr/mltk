"""Adversarial robustness testing -- check model stability under input perturbations.

A robust model should not change predictions when tiny noise is added to inputs.
Fragile models fail unpredictably on slightly unusual production data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_robust(
    model_fn: Callable[..., Any],
    inputs: Any,
    perturbation: str = "gaussian",
    epsilon: float = 0.01,
    stability: float = 0.95,
) -> TestResult:
    """Assert model predictions are stable under input perturbations.

    Args:
        model_fn: Function that takes inputs and returns predictions.
        inputs: Test inputs (2D array: samples x features).
        perturbation: Noise type -- "gaussian" or "uniform".
        epsilon: Noise magnitude (std for gaussian, range for uniform).
        stability: Min fraction of inputs that must keep same prediction.

    Returns:
        TestResult with stability score and details.

    Example:
        >>> import numpy as np
        >>> def clf(x): return (x.sum(axis=1) > 0).astype(int)
        >>> inputs = np.random.randn(100, 5)
        >>> assert_robust(clf, inputs, epsilon=0.01, stability=0.9)
    """
    x = np.asarray(inputs, dtype=np.float64)

    if x.size == 0:
        return assert_true(
            False,
            name="model.robust",
            message="Cannot test robustness on empty inputs",
            severity=Severity.CRITICAL,
        )

    if perturbation not in ("gaussian", "uniform"):
        return assert_true(
            False,
            name="model.robust",
            message=f"Unknown perturbation: '{perturbation}'. Supported: gaussian, uniform",
            severity=Severity.CRITICAL,
        )

    # Get original predictions
    original_preds = np.asarray(model_fn(x))

    # Generate noise
    rng = np.random.default_rng(42)
    if perturbation == "gaussian":
        noise = rng.normal(0, epsilon, x.shape)
    else:  # uniform
        noise = rng.uniform(-epsilon, epsilon, x.shape)

    # Get perturbed predictions
    x_noisy = x + noise
    perturbed_preds = np.asarray(model_fn(x_noisy))

    # Compute stability
    n_total = len(original_preds)
    n_stable = int((original_preds == perturbed_preds).sum())
    stability_score = n_stable / n_total if n_total > 0 else 0.0

    passed = stability_score >= stability

    message = (
        f"Stability={stability_score:.4f} >= {stability} ({n_stable}/{n_total} stable)"
        if passed
        else f"Fragile: stability={stability_score:.4f} < {stability} "
        f"({n_total - n_stable}/{n_total} predictions changed)"
    )

    return assert_true(
        passed,
        name="model.robust",
        message=message,
        severity=Severity.CRITICAL,
        perturbation=perturbation,
        epsilon=epsilon,
        stability_score=stability_score,
        stability_threshold=stability,
        n_stable=n_stable,
        n_total=n_total,
        n_changed=n_total - n_stable,
    )
