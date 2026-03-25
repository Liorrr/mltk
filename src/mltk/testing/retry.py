"""Non-deterministic test retry with confidence intervals."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RetryResult:
    """Result of a confidence-interval retry run."""

    pass_count: int
    fail_count: int
    pass_rate: float
    confidence_lower: float
    confidence_upper: float
    is_passing: bool  # True if CI lower bound > failure_threshold


def _wilson_interval(
    successes: int,
    n: int,
    confidence: float,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval.

    Args:
        successes: Number of successful trials.
        n: Total number of trials.
        confidence: Confidence level in (0, 1), e.g. 0.95.

    Returns:
        ``(lower, upper)`` bounds of the interval.
    """
    if n == 0:
        return 0.0, 1.0

    # z-score for two-sided interval
    alpha = 1.0 - confidence
    # Approximation via quantile of standard normal.
    # scipy is not a dependency so we use a direct closed-form for common values.
    # For full generality we use the inverse-error-function approach.
    z = _normal_ppf(1.0 - alpha / 2.0)

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


def _normal_ppf(p: float) -> float:
    """Percent-point function of the standard normal (no scipy required)."""
    # Rational approximation by Peter Acklam — absolute error < 1.15e-9.
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )


def retry_until_confident(
    func: Callable[[], None],
    min_runs: int = 3,
    max_runs: int = 10,
    confidence: float = 0.95,
    failure_threshold: float = 0.5,
) -> RetryResult:
    """Run a test multiple times and evaluate it with a Wilson confidence interval.

    Continues running until ``max_runs`` is reached.  After each run (starting
    from ``min_runs``) evaluates the Wilson score interval.

    The test is considered **passing** if the *lower* bound of the CI is
    strictly above ``failure_threshold``.

    Args:
        func: Zero-argument callable; raises on failure.
        min_runs: Minimum executions before early termination is considered.
        max_runs: Hard cap on executions.
        confidence: Confidence level for the Wilson interval (e.g. 0.95).
        failure_threshold: Lower-bound must exceed this to declare passing.

    Returns:
        :class:`RetryResult` with counts, pass rate, CI bounds, and verdict.

    Example:
        >>> def flaky_model_check():
        ...     import random
        ...     assert random.random() > 0.1  # passes 90% of the time
        >>> result = retry_until_confident(flaky_model_check, max_runs=20)
        >>> result.is_passing
        True
    """
    pass_count = 0
    fail_count = 0

    for _i in range(max_runs):
        try:
            func()
            pass_count += 1
        except Exception:  # noqa: BLE001
            fail_count += 1

        total = pass_count + fail_count
        if total >= min_runs:
            lower, upper = _wilson_interval(pass_count, total, confidence)
            # Early exit if verdict is clear
            if lower > failure_threshold or upper <= failure_threshold:
                break

    total = pass_count + fail_count
    pass_rate = pass_count / total if total > 0 else 0.0
    lower, upper = _wilson_interval(pass_count, total, confidence)
    is_passing = lower > failure_threshold

    return RetryResult(
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
        confidence_lower=lower,
        confidence_upper=upper,
        is_passing=is_passing,
    )
