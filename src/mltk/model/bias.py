"""Model bias and fairness testing -- detect discrimination across demographic groups.

EU AI Act (effective Aug 2, 2026) mandates bias detection for high-risk AI.
US four-fifths rule requires selection rates within 80% across groups.
Implements 5 fairness metrics with zero dependencies (pure numpy).

Note: Chouldechova-Kleinberg impossibility theorem means you cannot satisfy
demographic parity, equalized odds, AND predictive parity simultaneously
when group base rates differ. Choose the metric that matches your use case.
"""

from __future__ import annotations

import itertools
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
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert no bias across demographic groups.

    Args:
        y_true: Ground truth labels (binary 0/1).
        y_pred: Model predictions (binary 0/1).
        sensitive_feature: Protected attribute array (e.g., gender, race).
        method: Fairness metric to evaluate.
        threshold: Custom threshold. None = method-specific default.
        severity: Severity level for the assertion (default CRITICAL).

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
            severity=severity,
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
            severity=severity,
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
        return _check_demographic_parity(rates, thresh, method, severity)
    elif method == "equalized_odds":
        return _check_equalized_odds(rates, thresh, method, severity)
    elif method == "predictive_parity":
        return _check_predictive_parity(rates, thresh, method, severity)
    elif method == "disparate_impact":
        return _check_disparate_impact(rates, thresh, method, severity)
    else:  # equal_opportunity
        return _check_equal_opportunity(rates, thresh, method, severity)


def _check_demographic_parity(
    rates: dict[str, dict[str, float]],
    threshold: float,
    method: str,
    severity: Severity = Severity.CRITICAL,
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
        severity=severity,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_equalized_odds(
    rates: dict[str, dict[str, float]],
    threshold: float,
    method: str,
    severity: Severity = Severity.CRITICAL,
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
        severity=severity,
        method=method,
        statistic=diff,
        tpr_diff=tpr_diff,
        fpr_diff=fpr_diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_predictive_parity(
    rates: dict[str, dict[str, float]],
    threshold: float,
    method: str,
    severity: Severity = Severity.CRITICAL,
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
        severity=severity,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )


def _check_disparate_impact(
    rates: dict[str, dict[str, float]],
    threshold: float,
    method: str,
    severity: Severity = Severity.CRITICAL,
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
        severity=severity,
        method=method,
        statistic=ratio,
        threshold=threshold,
        group_rates=rates,
    )


def _check_equal_opportunity(
    rates: dict[str, dict[str, float]],
    threshold: float,
    method: str,
    severity: Severity = Severity.CRITICAL,
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
            else (
                f"Bias detected: equal opportunity "
                f"diff={diff:.4f} > {threshold}"
            )
        ),
        severity=severity,
        method=method,
        statistic=diff,
        threshold=threshold,
        group_rates=rates,
    )


_INTERSECTIONAL_DEFAULTS: dict[str, float] = {
    "demographic_parity": 0.10,
    "equalized_odds": 0.10,
    "disparate_impact": 0.80,
}


def _subgroup_label(
    attrs: dict[str, str],
) -> str:
    """Build a readable label like 'gender=F & race=Black'.

    Args:
        attrs: Dict mapping attribute name to value.

    Returns:
        Human-readable subgroup identifier string.
    """
    parts = [f"{k}={v}" for k, v in sorted(attrs.items())]
    return " & ".join(parts)


def _subgroup_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str,
) -> dict[str, float]:
    """Compute metric for a single subgroup.

    Args:
        y_true: Binary ground truth for the subgroup.
        y_pred: Binary predictions for the subgroup.
        method: Fairness method name.

    Returns:
        Dict with the relevant metric values.
    """
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sel_rate = (tp + fp) / n if n > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "selection_rate": sel_rate,
        "tpr": tpr,
        "fpr": fpr,
        "count": n,
    }


@timed_assertion
def assert_intersectional_fairness(
    y_true: Any,
    y_pred: Any,
    sensitive_features: dict[str, Any],
    method: str = "demographic_parity",
    threshold: float | None = None,
    min_subgroup_size: int = 30,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert fairness across ALL intersectional subgroups.

    Implements Crenshaw intersectionality: tests every
    combination of protected attributes. A model fair for
    women AND fair for Black people is NOT guaranteed fair
    for Black women.

    Args:
        y_true: Binary ground truth (0/1).
        y_pred: Binary predictions (0/1).
        sensitive_features: Dict mapping attribute name to
            array of values (e.g. ``{"gender": [...],
            "race": [...]}``)
        method: ``"demographic_parity"``,
            ``"equalized_odds"``, or
            ``"disparate_impact"``.
        threshold: Custom threshold. None uses defaults
            (0.10/0.10/0.80).
        min_subgroup_size: Minimum samples per combo.
            Smaller subgroups are skipped.
        severity: Severity level for the assertion.

    Returns:
        TestResult with worst-case disparity, evaluated
        and skipped subgroups, and per-subgroup metrics.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1,0,1,0,1,0]*10)
        >>> y_pred = np.array([1,0,1,0,0,0]*10)
        >>> features = {
        ...     "gender": np.array(["M","M","F","F"]*15),
        ...     "age": np.array(["young","old"]*30),
        ... }
        >>> assert_intersectional_fairness(
        ...     y_true, y_pred, features,
        ...     min_subgroup_size=5,
        ... )
    """
    if method not in _INTERSECTIONAL_DEFAULTS:
        supported = list(_INTERSECTIONAL_DEFAULTS.keys())
        return assert_true(
            False,
            name="model.intersectional_fairness",
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {supported}"
            ),
            severity=severity,
        )

    thresh = (
        threshold
        if threshold is not None
        else _INTERSECTIONAL_DEFAULTS[method]
    )

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) == 0:
        return assert_true(
            False,
            name="model.intersectional_fairness",
            message="Cannot compute fairness on empty arrays",
            severity=severity,
        )

    # Convert feature arrays to numpy
    attr_names = sorted(sensitive_features.keys())
    attr_arrays = {
        k: np.asarray(sensitive_features[k])
        for k in attr_names
    }
    attr_uniques = {
        k: list(np.unique(v)) for k, v in attr_arrays.items()
    }

    # Enumerate all combinations
    combo_values = [attr_uniques[k] for k in attr_names]
    all_combos = list(itertools.product(*combo_values))

    evaluated: dict[str, dict[str, float]] = {}
    skipped: dict[str, int] = {}

    for combo in all_combos:
        attrs = dict(zip(attr_names, combo, strict=True))
        label = _subgroup_label(attrs)

        mask = np.ones(len(y_t), dtype=bool)
        for k, v in attrs.items():
            mask &= attr_arrays[k] == v

        n = int(mask.sum())
        if n < min_subgroup_size:
            skipped[label] = n
            continue

        metrics = _subgroup_metric(
            y_t[mask], y_p[mask], method,
        )
        evaluated[label] = metrics

    n_total = len(all_combos)
    n_evaluated = len(evaluated)
    n_skipped = len(skipped)

    if n_evaluated < 2:
        return assert_true(
            True,
            name="model.intersectional_fairness",
            message=(
                f"Only {n_evaluated} subgroup(s) with "
                f">={min_subgroup_size} samples "
                f"-- check not applicable"
            ),
            severity=Severity.INFO,
            method=method,
            evaluated_subgroups=evaluated,
            skipped_subgroups=skipped,
            n_total_combos=n_total,
            n_evaluated=n_evaluated,
            n_skipped=n_skipped,
        )

    # Compute worst-case disparity
    worst_stat: float
    worst_label: str

    if method == "demographic_parity":
        rates = [
            (lbl, m["selection_rate"])
            for lbl, m in evaluated.items()
        ]
        vals = [r[1] for r in rates]
        diff = max(vals) - min(vals)
        worst_stat = diff
        max_lbl = rates[vals.index(max(vals))][0]
        min_lbl = rates[vals.index(min(vals))][0]
        worst_label = f"{max_lbl} vs {min_lbl}"

    elif method == "equalized_odds":
        tprs = [
            (lbl, m["tpr"])
            for lbl, m in evaluated.items()
        ]
        fprs = [
            (lbl, m["fpr"])
            for lbl, m in evaluated.items()
        ]
        tpr_vals = [r[1] for r in tprs]
        fpr_vals = [r[1] for r in fprs]
        tpr_diff = max(tpr_vals) - min(tpr_vals)
        fpr_diff = max(fpr_vals) - min(fpr_vals)
        if tpr_diff >= fpr_diff:
            worst_stat = tpr_diff
            mx = tprs[tpr_vals.index(max(tpr_vals))][0]
            mn = tprs[tpr_vals.index(min(tpr_vals))][0]
        else:
            worst_stat = fpr_diff
            mx = fprs[fpr_vals.index(max(fpr_vals))][0]
            mn = fprs[fpr_vals.index(min(fpr_vals))][0]
        worst_label = f"{mx} vs {mn}"

    else:  # disparate_impact
        rates = [
            (lbl, m["selection_rate"])
            for lbl, m in evaluated.items()
        ]
        vals = [r[1] for r in rates]
        max_r = max(vals)
        min_r = min(vals)
        ratio = min_r / max_r if max_r > 0 else 0.0
        worst_stat = ratio
        max_lbl = rates[vals.index(max_r)][0]
        min_lbl = rates[vals.index(min_r)][0]
        worst_label = f"{min_lbl} vs {max_lbl}"

    # Check pass/fail
    if method == "disparate_impact":
        passed = worst_stat >= thresh
    else:
        passed = worst_stat <= thresh

    stat_fmt = f"{worst_stat:.4f}"
    op = ">=" if method == "disparate_impact" else "<="

    if passed:
        message = (
            f"Intersectional {method} "
            f"{stat_fmt} {op} {thresh} "
            f"across {n_evaluated} subgroups"
        )
    else:
        message = (
            f"Intersectional bias: {method} "
            f"{stat_fmt} {'<' if method == 'disparate_impact' else '>'} "
            f"{thresh} ({worst_label})"
        )

    return assert_true(
        passed,
        name="model.intersectional_fairness",
        message=message,
        severity=severity,
        method=method,
        worst_case_subgroup=worst_label,
        worst_case_statistic=worst_stat,
        threshold=thresh,
        evaluated_subgroups=evaluated,
        skipped_subgroups=skipped,
        n_total_combos=n_total,
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
    )
