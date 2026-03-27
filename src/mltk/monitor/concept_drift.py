"""Concept drift detection -- detect when P(Y|X) changes between windows.

Input drift (P(X)) catches feature distribution shifts. Output drift (P(Y-hat))
catches prediction distribution shifts. But neither detects the case where the
*relationship* between inputs and outputs changes -- a model can receive
identically distributed inputs and produce identically distributed outputs
while getting the answers wrong in a completely different pattern.

Concept drift (P(Y|X)) catches this by comparing error rates between a
reference window and a current window. If the model's error rate has changed
significantly, the underlying concept has likely shifted.

Supports 3 methods:
- chi2: Chi-squared test on a 2x2 contingency table (correct/error x ref/cur)
- fisher: Fisher's exact test for small samples (falls back to chi2 without scipy)
- proportion: Z-test for proportions (pure numpy, no scipy needed)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

_SUPPORTED_METHODS = ("chi2", "fisher", "proportion")


def _chi2_manual(table: np.ndarray) -> tuple[float, float]:
    """Compute chi-squared statistic and p-value for a 2x2 contingency table.

    Uses the chi-squared approximation with Yates' continuity correction.
    The p-value is computed from the survival function of the chi-squared
    distribution with 1 degree of freedom (Wilson-Hilferty approximation
    for the CDF, exact enough for practical monitoring).

    Args:
        table: 2x2 numpy array [[correct_ref, error_ref],
                                 [correct_cur, error_cur]].

    Returns:
        Tuple of (chi2_statistic, p_value).
    """
    table = table.astype(np.float64)
    n = table.sum()
    if n == 0:
        return 0.0, 1.0

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)

    # Expected frequencies under independence
    expected = np.outer(row_sums, col_sums) / n

    # Avoid division by zero in degenerate tables
    if np.any(expected == 0):
        return 0.0, 1.0

    # Chi-squared statistic (no continuity correction for general use)
    chi2_stat = float(np.sum((table - expected) ** 2 / expected))

    # Compute p-value from chi2 distribution with df=1
    # Use the regularized incomplete gamma function approximation
    p_value = _chi2_sf(chi2_stat, df=1)
    return chi2_stat, p_value


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) for the chi-squared distribution.

    For df=1, chi2 is the square of a standard normal, so:
        P(X > x) = 2 * (1 - Phi(sqrt(x)))

    Uses the complementary error function (math.erfc) for accuracy.

    Args:
        x: Chi-squared test statistic.
        df: Degrees of freedom (only df=1 is used here).

    Returns:
        p-value (probability of observing a value >= x).
    """
    if x <= 0:
        return 1.0
    if df == 1:
        # For df=1: chi2 survival = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2.0))
    # Fallback for other df (not used in this module)
    return math.erfc(math.sqrt(x / 2.0))


def _proportion_ztest(
    errors_ref: np.ndarray,
    errors_cur: np.ndarray,
) -> tuple[float, float]:
    """Two-proportion Z-test comparing error rates.

    Args:
        errors_ref: Boolean array of errors in reference window.
        errors_cur: Boolean array of errors in current window.

    Returns:
        Tuple of (z_statistic, p_value). Two-sided test.
    """
    n1 = len(errors_ref)
    n2 = len(errors_cur)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    p1 = errors_ref.mean()
    p2 = errors_cur.mean()
    p_pool = (errors_ref.sum() + errors_cur.sum()) / (n1 + n2)

    # If pooled proportion is 0 or 1, there's no variance to test
    if p_pool == 0.0 or p_pool == 1.0:
        return 0.0, 1.0

    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0

    z = (p1 - p2) / se

    # Two-sided p-value from standard normal
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), p_value


@timed_assertion
def assert_no_concept_drift(
    y_true_ref: np.ndarray | list,
    y_pred_ref: np.ndarray | list,
    y_true_cur: np.ndarray | list,
    y_pred_cur: np.ndarray | list,
    method: str = "chi2",
    alpha: float = 0.05,
) -> TestResult:
    """Assert no concept drift (P(Y|X)) between reference and current windows.

    Compares classification error rates between two time windows using a
    statistical test. A significant difference indicates that the relationship
    between inputs and outputs has changed -- the model's accuracy is
    degrading or improving in a statistically meaningful way.

    This completes the drift detection story:
    - P(X) drift: ``assert_no_drift`` (input feature distribution shift)
    - P(Y-hat) drift: ``assert_no_output_drift`` (prediction distribution shift)
    - P(Y|X) drift: ``assert_no_concept_drift`` (concept/relationship shift)

    Supported methods:
    - ``"chi2"``: Chi-squared test on 2x2 contingency table. Uses scipy if
      available, otherwise a pure-numpy fallback.
    - ``"fisher"``: Fisher's exact test (scipy required). Falls back to chi2
      if scipy is not installed.
    - ``"proportion"``: Z-test for two proportions. Pure numpy, no scipy needed.

    Args:
        y_true_ref: True labels for the reference window.
        y_pred_ref: Predicted labels for the reference window.
        y_true_cur: True labels for the current window.
        y_pred_cur: Predicted labels for the current window.
        method: Statistical test method -- "chi2", "fisher", or "proportion".
        alpha: Significance level. Pass if p_value >= alpha.

    Returns:
        TestResult with concept drift details including error rates, p-value,
        and test statistic.

    Raises:
        MltkAssertionError: When concept drift is detected (p_value < alpha).

    Example:
        >>> y_true_ref = [0, 1, 1, 0, 1, 0, 0, 1]
        >>> y_pred_ref = [0, 1, 1, 0, 1, 0, 0, 1]  # perfect
        >>> y_true_cur = [0, 1, 1, 0, 1, 0, 0, 1]
        >>> y_pred_cur = [1, 0, 0, 1, 0, 1, 1, 0]  # all wrong
        >>> assert_no_concept_drift(y_true_ref, y_pred_ref, y_true_cur, y_pred_cur)
    """
    name = "monitor.concept_drift"

    if method not in _SUPPORTED_METHODS:
        return assert_true(
            False,
            name=name,
            message=f"Unknown method: '{method}'. Supported: {list(_SUPPORTED_METHODS)}",
            severity=Severity.CRITICAL,
            method=method,
            alpha=alpha,
        )

    # Convert to numpy arrays
    y_true_ref_arr = np.asarray(y_true_ref).ravel()
    y_pred_ref_arr = np.asarray(y_pred_ref).ravel()
    y_true_cur_arr = np.asarray(y_true_cur).ravel()
    y_pred_cur_arr = np.asarray(y_pred_cur).ravel()

    n_ref = len(y_true_ref_arr)
    n_cur = len(y_true_cur_arr)

    if n_ref == 0 or n_cur == 0:
        return assert_true(
            True,
            name=name,
            message="No samples to compare (empty array)",
            severity=Severity.CRITICAL,
            method=method,
            alpha=alpha,
            n_ref=n_ref,
            n_cur=n_cur,
            drift_detected=False,
        )

    # Compute error indicators
    errors_ref = (y_true_ref_arr != y_pred_ref_arr).astype(int)
    errors_cur = (y_true_cur_arr != y_pred_cur_arr).astype(int)

    error_rate_ref = float(errors_ref.mean())
    error_rate_cur = float(errors_cur.mean())
    error_rate_diff = error_rate_cur - error_rate_ref

    # Build common details dict
    base_details: dict[str, Any] = {
        "error_rate_ref": error_rate_ref,
        "error_rate_cur": error_rate_cur,
        "error_rate_diff": error_rate_diff,
        "alpha": alpha,
        "n_ref": n_ref,
        "n_cur": n_cur,
    }

    if method == "chi2":
        statistic, p_value = _run_chi2(errors_ref, errors_cur)
        actual_method = "chi2"
    elif method == "fisher":
        statistic, p_value, actual_method = _run_fisher(errors_ref, errors_cur)
    else:  # proportion
        statistic, p_value = _proportion_ztest(errors_ref, errors_cur)
        actual_method = "proportion"

    passed = bool(p_value >= alpha)
    base_details["p_value"] = p_value
    base_details["method"] = actual_method
    base_details["statistic"] = statistic
    base_details["drift_detected"] = not passed

    message = (
        f"No concept drift: error_rate_ref={error_rate_ref:.4f}, "
        f"error_rate_cur={error_rate_cur:.4f}, p={p_value:.4f} "
        f"(alpha={alpha}, method={actual_method})"
        if passed
        else f"Concept drift detected: error_rate_ref={error_rate_ref:.4f}, "
        f"error_rate_cur={error_rate_cur:.4f}, p={p_value:.4f} < {alpha} "
        f"(method={actual_method})"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        **base_details,
    )


def _run_chi2(
    errors_ref: np.ndarray,
    errors_cur: np.ndarray,
) -> tuple[float, float]:
    """Run chi-squared test on error contingency table.

    Tries scipy first, falls back to manual implementation.

    Args:
        errors_ref: Binary error array for reference window.
        errors_cur: Binary error array for current window.

    Returns:
        Tuple of (chi2_statistic, p_value).
    """
    correct_ref = int((errors_ref == 0).sum())
    error_ref = int(errors_ref.sum())
    correct_cur = int((errors_cur == 0).sum())
    error_cur = int(errors_cur.sum())

    table = np.array([[correct_ref, error_ref], [correct_cur, error_cur]])

    # Degenerate case: a column is all zeros (e.g., both windows perfect or
    # both windows all-wrong). No variance to test -- error rates are identical.
    if np.any(table.sum(axis=0) == 0):
        return 0.0, 1.0

    try:
        from scipy.stats import chi2_contingency

        stat, p_value, _, _ = chi2_contingency(table, correction=False)
        return float(stat), float(p_value)
    except ImportError:
        return _chi2_manual(table)


def _run_fisher(
    errors_ref: np.ndarray,
    errors_cur: np.ndarray,
) -> tuple[float, float, str]:
    """Run Fisher's exact test, falling back to chi2 if scipy is unavailable.

    Args:
        errors_ref: Binary error array for reference window.
        errors_cur: Binary error array for current window.

    Returns:
        Tuple of (statistic, p_value, method_used).
    """
    correct_ref = int((errors_ref == 0).sum())
    error_ref = int(errors_ref.sum())
    correct_cur = int((errors_cur == 0).sum())
    error_cur = int(errors_cur.sum())

    table = np.array([[correct_ref, error_ref], [correct_cur, error_cur]])

    # Degenerate case: a column is all zeros -- error rates are identical.
    if np.any(table.sum(axis=0) == 0):
        return 0.0, 1.0, "fisher"

    try:
        from scipy.stats import fisher_exact

        odds_ratio, p_value = fisher_exact(table)
        return float(odds_ratio), float(p_value), "fisher"
    except ImportError:
        stat, p_value = _chi2_manual(table)
        return stat, p_value, "chi2"
