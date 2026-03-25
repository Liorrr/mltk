"""Tabular feature testing — per-column drift and importance stability."""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    method: str = "psi",
    threshold: float = 0.1,
    columns: list[str] | None = None,
) -> TestResult:
    """Assert no drift across DataFrame columns.

    Args:
        ref_df: Reference DataFrame (baseline).
        cur_df: Current DataFrame to compare.
        method: Drift method per column ("psi" or "ks").
        threshold: Drift threshold per column.
        columns: Columns to check. None = all numeric columns.

    Returns:
        TestResult with per-column drift scores.

    Example:
        >>> import pandas as pd
        >>> ref = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
        >>> cur = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
        >>> assert_feature_drift(ref, cur, method="psi")
    """
    from mltk.data.drift import assert_no_drift

    check_cols = columns or [
        c for c in ref_df.columns
        if c in cur_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])
    ]

    drifted: dict[str, float] = {}
    stable: dict[str, float] = {}

    for col in check_cols:
        try:
            result = assert_no_drift(
                ref_df[col], cur_df[col], method=method, threshold=threshold
            )
            stable[col] = result.details.get("statistic", 0.0)
        except Exception:
            drifted[col] = threshold + 0.01  # Mark as drifted

    passed = len(drifted) == 0
    message = (
        f"No drift in {len(check_cols)} features"
        if passed
        else f"Drift in {len(drifted)} feature(s): {list(drifted.keys())}"
    )

    return assert_true(
        passed, name="tabular.feature_drift", message=message,
        severity=Severity.CRITICAL,
        drifted_features=drifted, stable_features=stable,
        total_features=len(check_cols), method=method, threshold=threshold,
    )


@timed_assertion
def assert_feature_importance_stable(
    shap_ref: dict[str, float],
    shap_cur: dict[str, float],
    max_rank_change: int = 3,
) -> TestResult:
    """Assert SHAP feature importance rankings are stable.

    Args:
        shap_ref: Reference feature importances {feature: importance}.
        shap_cur: Current feature importances.
        max_rank_change: Maximum allowed rank change for any feature.

    Returns:
        TestResult with rank changes.

    Example:
        >>> ref = {"age": 0.5, "income": 0.3, "score": 0.2}
        >>> cur = {"age": 0.4, "income": 0.35, "score": 0.25}
        >>> assert_feature_importance_stable(ref, cur, max_rank_change=2)
    """
    ref_ranked = sorted(shap_ref, key=lambda k: shap_ref[k], reverse=True)
    cur_ranked = sorted(shap_cur, key=lambda k: shap_cur[k], reverse=True)

    ref_ranks = {f: i for i, f in enumerate(ref_ranked)}
    cur_ranks = {f: i for i, f in enumerate(cur_ranked)}

    rank_changes: dict[str, int] = {}
    max_change = 0

    for feature in ref_ranks:
        if feature in cur_ranks:
            change = abs(ref_ranks[feature] - cur_ranks[feature])
            rank_changes[feature] = change
            max_change = max(max_change, change)

    passed = max_change <= max_rank_change
    message = (
        f"Feature rankings stable (max change: {max_change})"
        if passed
        else f"Feature importance shifted: max rank change={max_change} > {max_rank_change}"
    )

    return assert_true(
        passed, name="tabular.feature_importance", message=message,
        severity=Severity.WARNING,
        max_change=max_change, max_rank_change=max_rank_change,
        rank_changes=rank_changes,
    )
