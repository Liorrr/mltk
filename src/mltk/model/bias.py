"""Model bias and fairness testing -- detect discrimination across demographic groups.

EU AI Act (effective Aug 2, 2026) mandates bias detection for high-risk AI.
US four-fifths rule requires selection rates within 80% across groups.
Implements 5 fairness metrics with zero dependencies (pure numpy).

Note: Chouldechova-Kleinberg impossibility theorem means you cannot satisfy
demographic parity, equalized odds, AND predictive parity simultaneously
when group base rates differ. Choose the metric that matches your use case.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "demographic_parity": 0.10,
    "equalized_odds": 0.10,
    "predictive_parity": 0.10,
    "disparate_impact": 0.80,
    "equal_opportunity": 0.10,
}


def _group_rates(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, dict[str, float]]:
    """Compute per-group confusion matrix rates.

    Args:
        y_true: Binary ground truth labels.
        y_pred: Binary predicted labels.
        groups: Group membership array.

    Returns:
        Dict mapping group name to rates dict with keys:
        selection_rate, tpr, fpr, ppv, count.
    """
    unique_groups = np.unique(groups)
    result: dict[str, dict[str, float]] = {}

    for g in unique_groups:
        mask = groups == g
        g_true = y_true[mask]
        g_pred = y_pred[mask]
        n = len(g_true)

        if n == 0:
            continue

        tp = int(((g_true == 1) & (g_pred == 1)).sum())
        fp = int(((g_true == 0) & (g_pred == 1)).sum())
        tn = int(((g_true == 0) & (g_pred == 0)).sum())
        fn = int(((g_true == 1) & (g_pred == 0)).sum())

        selection_rate = (tp + fp) / n if n > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        result[str(g)] = {
            "selection_rate": selection_rate,
            "tpr": tpr,
            "fpr": fpr,
            "ppv": ppv,
            "count": n,
        }

    return result


@timed_assertion
def assert_no_bias(
    y_true: Any,
    y_pred: Any,
    sensitive_feature: Any,
    method: str = "demographic_parity",
    threshold: float | None = None,
) -> TestResult:
    """Assert no bias across demographic groups.

    Args:
        y_true: Ground truth labels (binary 0/1).
        y_pred: Model predictions (binary 0/1).
        sensitive_feature: Protected attribute array (e.g., gender, race).
        method: Fairness metric to evaluate.
        threshold: Custom threshold. None = method-specific default.

    Returns:
        TestResult with per-group metrics and bias statistics.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 0, 1, 0, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0, 0])
        >>> groups = np.array(["M", "M", "M", "F", "F", "F"])
        >>> assert_no_bias(y_true, y_pred, groups, method="demographic_parity")
    """
    if method not in _DEFAULT_THRESHOLDS:
        return assert_true(
            False,
            name="model.bias",
            message=f"Unknown method: '{method}'. Supported: {list(_DEFAULT_THRESHOLDS.keys())}",
            severity=Severity.CRITICAL,
        )

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    groups = np.asarray(sensitive_feature)
    thresh = threshold if threshold is not None else _DEFAULT_THRESHOLDS[method]

    if len(y_t) == 0:
        return assert_true(
            False,
            name="model.bias",
            message="Cannot compute bias on empty arrays",
            severity=Severity.CRITICAL,
        )

    rates = _group_rates(y_t, y_p, groups)

    if len(rates) <= 1:
        return assert_true(
            True,
            name=f"model.bias.{method}",
            message=f"Only {len(rates)} group(s) — bias check not applicable",
            severity=Severity.INFO,
            method=method,
            group_rates=rates,
        )

    if method == "demographic_parity":
        return _check_demographic_parity(rates, thresh, method)
    elif method == "equalized_odds":
        return _check_equalized_odds(rates, thresh, method)
    elif method == "predictive_parity":
        return _check_predictive_parity(rates, thresh, method)
    elif method == "disparate_impact":
        return _check_disparate_impact(rates, thresh, method)
    else:  # equal_opportunity
        return _check_equal_opportunity(rates, thresh, method)


def _check_demographic_parity(
    rates: dict[str, dict[str, float]], threshold: float, method: str
) -> TestResult:
    """Max selection rate difference across groups.

    Args:
        rates: Per-group rates from _group_rates.
        threshold: Maximum allowed selection rate difference.
        method: Method name for TestResult.

    Returns:
        TestResult with demographic parity statistic.
    """
    sel_rates = [r["selection_rate"] for r in rates.values()]
    diff = max(sel_rates) - min(sel_rates)
    passed = diff <= threshold

    return assert_true(
        passed,
        name=f"model.bias.{method}",
        message=(
            f"Demographic parity diff={diff:.4f} <= {threshold}"
            if passed
            else f"Bias detected: demographic parity diff={diff:.4f} > {threshold}"
        ),
        severity=Severity.CRITICAL,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_equalized_odds(
    rates: dict[str, dict[str, float]], threshold: float, method: str
) -> TestResult:
    """Max of (TPR diff, FPR diff) across groups.

    Args:
        rates: Per-group rates from _group_rates.
        threshold: Maximum allowed TPR or FPR difference.
        method: Method name for TestResult.

    Returns:
        TestResult with equalized odds statistic.
    """
    tprs = [r["tpr"] for r in rates.values()]
    fprs = [r["fpr"] for r in rates.values()]
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    diff = max(tpr_diff, fpr_diff)
    passed = diff <= threshold

    return assert_true(
        passed,
        name=f"model.bias.{method}",
        message=(
            f"Equalized odds diff={diff:.4f} <= {threshold}"
            if passed
            else f"Bias detected: equalized odds diff={diff:.4f} > {threshold}"
        ),
        severity=Severity.CRITICAL,
        method=method,
        statistic=diff,
        tpr_diff=tpr_diff,
        fpr_diff=fpr_diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_predictive_parity(
    rates: dict[str, dict[str, float]], threshold: float, method: str
) -> TestResult:
    """Max PPV difference across groups.

    Args:
        rates: Per-group rates from _group_rates.
        threshold: Maximum allowed PPV difference.
        method: Method name for TestResult.

    Returns:
        TestResult with predictive parity statistic.
    """
    ppvs = [r["ppv"] for r in rates.values()]
    diff = max(ppvs) - min(ppvs)
    passed = diff <= threshold

    return assert_true(
        passed,
        name=f"model.bias.{method}",
        message=(
            f"Predictive parity diff={diff:.4f} <= {threshold}"
            if passed
            else f"Bias detected: predictive parity diff={diff:.4f} > {threshold}"
        ),
        severity=Severity.CRITICAL,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_disparate_impact(
    rates: dict[str, dict[str, float]], threshold: float, method: str
) -> TestResult:
    """Min/max selection rate ratio (four-fifths rule).

    Args:
        rates: Per-group rates from _group_rates.
        threshold: Minimum allowed ratio (0.8 = four-fifths rule).
        method: Method name for TestResult.

    Returns:
        TestResult with disparate impact ratio.
    """
    sel_rates = [r["selection_rate"] for r in rates.values()]
    max_rate = max(sel_rates)
    min_rate = min(sel_rates)
    ratio = min_rate / max_rate if max_rate > 0 else 0.0
    passed = ratio >= threshold

    return assert_true(
        passed,
        name=f"model.bias.{method}",
        message=(
            f"Disparate impact ratio={ratio:.4f} >= {threshold}"
            if passed
            else f"Bias detected: disparate impact ratio={ratio:.4f} < {threshold} "
            f"(four-fifths rule)"
        ),
        severity=Severity.CRITICAL,
        method=method,
        statistic=ratio,
        threshold=threshold,
        group_rates=rates,
    )


def _check_equal_opportunity(
    rates: dict[str, dict[str, float]], threshold: float, method: str
) -> TestResult:
    """Max TPR difference across groups (relaxed equalized odds).

    Args:
        rates: Per-group rates from _group_rates.
        threshold: Maximum allowed TPR difference.
        method: Method name for TestResult.

    Returns:
        TestResult with equal opportunity statistic.
    """
    tprs = [r["tpr"] for r in rates.values()]
    diff = max(tprs) - min(tprs)
    passed = diff <= threshold

    return assert_true(
        passed,
        name=f"model.bias.{method}",
        message=(
            f"Equal opportunity diff={diff:.4f} <= {threshold}"
            if passed
            else f"Bias detected: equal opportunity diff={diff:.4f} > {threshold}"
        ),
        severity=Severity.CRITICAL,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )
