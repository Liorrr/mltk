"""Causal inference assertions -- validate treatment effects and confounding.

In ML, we constantly make causal claims: "Model B is better than Model A,"
"this feature improves conversion," "the new ranking algorithm increases
engagement." But correlation is not causation. Without proper causal analysis,
teams deploy "improvements" that are just noise or confounded by other factors.

This module provides two essential causal checks:

1. **Average Treatment Effect (ATE)** -- Does the treatment actually cause a
   statistically significant difference in outcomes? Uses a two-sample t-test
   to distinguish real effects from random variation.

   Example: You A/B test two recommendation models. Model B shows +2% click
   rate. Is that real, or would you see a similar difference by chance? ATE
   significance testing answers this. Without it, teams ship "improvements"
   that are just lucky samples.

2. **No Confounding** -- Is treatment assignment independent of features?
   If treatment correlates with covariates, the ATE is biased.

   Example: Your A/B test routes power users to Model B and casual users to
   Model A. Model B "wins" -- but is it the model or the user segment? If
   user_activity correlates with treatment assignment, you have confounding,
   and the ATE is unreliable.

   This check computes Pearson correlation between each feature and the
   treatment indicator. High correlations signal confounded experiments.

References:
    - Rubin, "Estimating Causal Effects of Treatments in Randomized and
      Nonrandomized Studies" (1974)
    - Imbens & Rubin, "Causal Inference for Statistics, Social, and
      Biomedical Sciences" (2015)
"""

from __future__ import annotations

import math

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _welch_t_test(
    group_a: np.ndarray, group_b: np.ndarray
) -> tuple[float, float]:
    """Welch's two-sample t-test (unequal variances).

    Computes the t-statistic and two-sided p-value without requiring scipy.
    Uses the Welch-Satterthwaite approximation for degrees of freedom.

    Why Welch's instead of Student's t-test? Student's assumes equal variance
    in both groups, which rarely holds in ML experiments (treatment and control
    groups often have different variance). Welch's is more robust.

    Args:
        group_a: Outcome values for the control group.
        group_b: Outcome values for the treatment group.

    Returns:
        Tuple of (t_statistic, p_value). p_value is two-sided.
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))

    se_a = var_a / n_a
    se_b = var_b / n_b
    se_diff = math.sqrt(se_a + se_b)

    if se_diff == 0.0:
        # Both groups have zero variance.  If means differ, effect is infinite;
        # if means are equal, there is no effect.
        if mean_a == mean_b:
            return 0.0, 1.0
        # Infinite t-stat => p ~ 0.
        return float("inf"), 0.0

    t_stat = (mean_b - mean_a) / se_diff

    # Welch-Satterthwaite degrees of freedom.
    numerator = (se_a + se_b) ** 2
    denom = (se_a**2 / (n_a - 1)) + (se_b**2 / (n_b - 1))
    if denom == 0.0:
        df = max(n_a + n_b - 2, 1)
    else:
        df = numerator / denom

    # Approximate p-value using the regularized incomplete beta function.
    # This avoids a scipy dependency while being accurate for df > 1.
    p_value = _t_distribution_cdf_two_sided(t_stat, df)
    return t_stat, p_value


def _t_distribution_cdf_two_sided(t: float, df: float) -> float:
    """Two-sided p-value from the Student-t distribution.

    Uses the relationship between the t-distribution CDF and the regularized
    incomplete beta function, computed via a continued-fraction expansion.
    Accurate to ~6 decimal places for df > 1.

    Args:
        t: The t-statistic.
        df: Degrees of freedom (can be non-integer via Welch-Satterthwaite).

    Returns:
        Two-sided p-value.
    """
    if df <= 0 or math.isnan(t) or math.isinf(df):
        return 1.0
    if math.isinf(t):
        return 0.0

    x = df / (df + t * t)
    # CDF of t uses the regularized incomplete beta function I_x(a, b)
    # with a = df/2, b = 0.5.
    p = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return p  # Already two-sided via this formulation.


def _regularized_incomplete_beta(
    x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz continued fraction.

    Args:
        x: Evaluation point in [0, 1].
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).
        max_iter: Maximum continued-fraction iterations.
        tol: Convergence tolerance.

    Returns:
        I_x(a, b), the regularized incomplete beta function value.
    """
    if x <= 0.0:
        return 1.0  # For the two-sided p-value formulation.
    if x >= 1.0:
        return 0.0

    # Use the symmetry relation for numerical stability.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter, tol)

    # Log-beta for normalization.
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - ln_beta
    ) / a

    # Lentz continued fraction for I_x(a, b).
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step.
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c

        # Odd step.
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    # Return p-value (two-sided formulation: I_x(df/2, 0.5) is already two-sided).
    return max(0.0, min(1.0, front * f))


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays.

    Pure numpy implementation -- no scipy needed. Handles constant arrays
    gracefully (returns 0.0 instead of NaN).

    Args:
        x: First array.
        y: Second array.

    Returns:
        Pearson r in [-1, 1]. Returns 0.0 if either array is constant.
    """
    n = len(x)
    if n < 2:
        return 0.0

    x_f = x.astype(float)
    y_f = y.astype(float)

    x_mean = np.mean(x_f)
    y_mean = np.mean(y_f)

    x_centered = x_f - x_mean
    y_centered = y_f - y_mean

    numerator = float(np.sum(x_centered * y_centered))
    denom_x = float(np.sum(x_centered**2))
    denom_y = float(np.sum(y_centered**2))

    denom = math.sqrt(denom_x * denom_y)
    if denom == 0.0:
        return 0.0

    return numerator / denom


@timed_assertion
def assert_ate_significant(
    treatment: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that the Average Treatment Effect is statistically significant.

    The ATE is the difference in mean outcomes between the treatment group
    (treatment=1) and the control group (treatment=0). Statistical significance
    is assessed via Welch's two-sample t-test at the given alpha level.

    Why this matters:
        Without significance testing, teams deploy model changes based on
        A/B tests that are just random noise. A +1% lift on 50 samples is
        meaningless; the same +1% on 50,000 samples is worth shipping.

    Interpretation:
        - PASS (p < alpha): The treatment effect is statistically significant.
          The observed difference is unlikely to arise from chance alone.
        - FAIL (p >= alpha): The effect is not distinguishable from noise.
          Do not ship based on this evidence.

    Args:
        treatment: Binary array (0=control, 1=treatment) indicating group
            assignment for each observation.
        outcome: Numeric outcome array (e.g., conversion, revenue, click rate).
        alpha: Significance level. Default 0.05. Lower = stricter
            (fewer false positives, but harder to detect real effects).
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with details: ate, p_value, alpha, n_treatment, n_control,
        mean_treatment, mean_control.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> treatment = np.array([0]*100 + [1]*100)
        >>> outcome = np.concatenate([rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)])
        >>> assert_ate_significant(treatment, outcome, alpha=0.05)
    """
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome, dtype=float)

    if len(treatment) != len(outcome):
        return assert_true(
            False,
            name="model.causal.ate_significant",
            message="treatment and outcome arrays must have the same length",
            severity=severity,
        )

    control_mask = treatment == 0
    treated_mask = treatment == 1

    n_control = int(control_mask.sum())
    n_treatment = int(treated_mask.sum())

    if n_control == 0 or n_treatment == 0:
        return assert_true(
            False,
            name="model.causal.ate_significant",
            message=(
                f"Need both treatment and control groups. "
                f"Got n_control={n_control}, n_treatment={n_treatment}"
            ),
            severity=severity,
            n_treatment=n_treatment,
            n_control=n_control,
        )

    outcome_control = outcome[control_mask]
    outcome_treated = outcome[treated_mask]

    mean_control = float(np.mean(outcome_control))
    mean_treatment = float(np.mean(outcome_treated))
    ate = mean_treatment - mean_control

    _, p_value = _welch_t_test(outcome_control, outcome_treated)

    passed = p_value < alpha
    message = (
        f"ATE={ate:.4f} is significant (p={p_value:.4f} < {alpha})"
        if passed
        else f"ATE={ate:.4f} is NOT significant (p={p_value:.4f} >= {alpha})"
    )

    return assert_true(
        passed,
        name="model.causal.ate_significant",
        message=message,
        severity=severity,
        ate=ate,
        p_value=p_value,
        alpha=alpha,
        n_treatment=n_treatment,
        n_control=n_control,
        mean_treatment=mean_treatment,
        mean_control=mean_control,
    )


@timed_assertion
def assert_no_confounding(
    X: np.ndarray,
    treatment: np.ndarray,
    max_correlation: float = 0.1,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that no features are correlated with treatment assignment.

    In a well-randomized experiment, treatment assignment should be independent
    of all observed covariates. If feature j correlates with the treatment
    indicator, it is a potential confounder -- the estimated ATE may be biased
    because the "treatment effect" partially reflects the effect of feature j.

    Why this matters:
        Imagine an e-commerce A/B test where premium users (high spend, many
        sessions) are more likely to be routed to the new model. The new model
        "wins" -- but is it the model or the user quality? Confounding makes
        your A/B test results unreliable.

    This check computes the Pearson correlation between each feature column
    and the treatment indicator. If ANY correlation exceeds max_correlation,
    the check fails and reports which features are confounded.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        treatment: Binary array (0=control, 1=treatment).
        max_correlation: Maximum allowed absolute correlation between any
            feature and the treatment indicator. Default 0.1.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with details: max_observed_correlation, confounded_features
        (list of column indices), correlations (dict mapping column index to
        correlation value).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> X = rng.standard_normal((200, 3))
        >>> treatment = rng.integers(0, 2, 200)  # Random assignment
        >>> assert_no_confounding(X, treatment, max_correlation=0.1)
    """
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        return assert_true(
            False,
            name="model.causal.no_confounding",
            message="X must be a 1-D or 2-D array",
            severity=severity,
        )

    if len(X) != len(treatment):
        return assert_true(
            False,
            name="model.causal.no_confounding",
            message="X and treatment must have the same number of rows",
            severity=severity,
        )

    n_features = X.shape[1]
    correlations: dict[int, float] = {}
    confounded: list[int] = []
    max_obs = 0.0

    for j in range(n_features):
        r = abs(_pearson_correlation(X[:, j], treatment))
        correlations[j] = r
        if r > max_obs:
            max_obs = r
        if r > max_correlation:
            confounded.append(j)

    passed = len(confounded) == 0
    message = (
        f"No confounding detected: max |r|={max_obs:.4f} <= {max_correlation}"
        if passed
        else f"Confounding detected: {len(confounded)} feature(s) correlated "
        f"with treatment (max |r|={max_obs:.4f} > {max_correlation}). "
        f"Confounded features: {confounded}"
    )

    return assert_true(
        passed,
        name="model.causal.no_confounding",
        message=message,
        severity=severity,
        max_observed_correlation=max_obs,
        confounded_features=confounded,
        correlations=correlations,
    )
