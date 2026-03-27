"""A/B test significance -- compare two model versions with statistical rigor.

Provides bootstrap-based significance testing to determine whether model B
is statistically better than model A. Avoids parametric assumptions by
resampling score differences and computing a confidence interval.

Functions:
    assert_ab_significance -- model B significantly outperforms model A
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_ab_significance(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    method: str = "bootstrap",
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert model B is significantly better than model A.

    Uses bootstrap resampling to build a confidence interval for the mean
    score difference (B - A). If the entire CI is above zero, the
    improvement is statistically significant at the given alpha level.

    Args:
        scores_a: Per-sample scores for model A.
        scores_b: Per-sample scores for model B.
        method: Statistical method -- ``"bootstrap"`` (default).
        alpha: Significance level (default 0.05 = 95% CI).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with CI bounds, mean difference, and p-value estimate.

    Example:
        >>> scores_a = [0.80, 0.82, 0.79, 0.81, 0.83]
        >>> scores_b = [0.88, 0.90, 0.87, 0.89, 0.91]
        >>> assert_ab_significance(scores_a, scores_b)
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if len(scores_a) == 0 or len(scores_b) == 0:
        return assert_true(
            False,
            name="model.ab_significance",
            message="Cannot run A/B test with empty score arrays",
            severity=severity,
        )

    if len(scores_a) != len(scores_b):
        return assert_true(
            False,
            name="model.ab_significance",
            message=(
                f"Score arrays must have equal length: "
                f"A has {len(scores_a)}, B has {len(scores_b)}"
            ),
            severity=severity,
        )

    if method != "bootstrap":
        return assert_true(
            False,
            name="model.ab_significance",
            message=f"Unknown method: '{method}'. Supported: 'bootstrap'",
            severity=severity,
        )

    # Compute observed difference
    diffs = scores_b - scores_a
    observed_diff = float(np.mean(diffs))

    # Bootstrap: resample differences with replacement, compute mean each time
    rng = np.random.default_rng(seed=42)
    n = len(diffs)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    # Confidence interval (percentile method)
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Significant if CI excludes 0 (entirely above 0 means B > A)
    significant = ci_lower > 0

    # Estimate p-value as fraction of bootstrap means <= 0
    p_value = float(np.mean(boot_means <= 0))

    message = (
        f"A/B significant: mean_diff={observed_diff:.4f}, "
        f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}"
        if significant
        else f"A/B not significant: mean_diff={observed_diff:.4f}, "
        f"CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}"
    )

    return assert_true(
        significant,
        name="model.ab_significance",
        message=message,
        severity=severity,
        mean_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        method=method,
    )
