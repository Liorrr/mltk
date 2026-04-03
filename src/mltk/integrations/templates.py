"""ML ticket templates — pre-formatted descriptions for common failure types."""

from __future__ import annotations

from typing import Any

_TEMPLATES: dict[str, str] = {
    "data_quality": (
        "[DATA] {test_name}\n\n"
        "Data quality check failed.\n"
        "- Assertion: {assertion_type}\n"
        "- Message: {message}\n"
        "- Detection time: {timestamp}\n\n"
        "Recommendation: Review data pipeline for breaking changes."
    ),
    "model_regression": (
        "[MODEL] {test_name}\n\n"
        "Model performance regression detected.\n"
        "- Metric: {metric_name}\n"
        "- Expected: {expected}\n"
        "- Actual: {actual}\n"
        "- Regression: {regression_pct}%\n\n"
        "Recommendation: Compare model versions, check training data."
    ),
    "drift_detection": (
        "[DRIFT] {test_name}\n\n"
        "Distribution drift detected.\n"
        "- Method: {method}\n"
        "- Statistic: {statistic}\n"
        "- Threshold: {threshold}\n\n"
        "Recommendation: Retrain model on recent data."
    ),
    "bias_violation": (
        "[BIAS] {test_name}\n\n"
        "Fairness constraint violated.\n"
        "- Method: {method}\n"
        "- Disparity: {disparity}\n"
        "- Threshold: {threshold}\n"
        "- Affected groups: {groups}\n\n"
        "Recommendation: Review training data for demographic imbalance."
    ),
    "default": (
        "[MLTK] {test_name}\n\n"
        "ML test assertion failed.\n"
        "- Type: {assertion_type}\n"
        "- Message: {message}\n"
        "- Severity: {severity}\n"
    ),
    "finding_issue": (
        "[MLTK:{assertion_type}] {test_name}\n\n"
        "Scan finding from mltk.\n"
        "- Scanner: {assertion_type}\n"
        "- Severity: {severity}\n"
        "- Message: {message}\n"
        "- Detection time: {timestamp}\n\n"
        "Recommendation: Review the finding and apply the suggested fix."
    ),
}


def render_ticket(
    template_name: str,
    **kwargs: Any,
) -> dict[str, str]:
    """Render a ticket title and description from a template.

    Args:
        template_name: Template key (data_quality, model_regression,
            drift_detection, bias_violation, default).
        **kwargs: Values to format into the template.

    Returns:
        Dict with 'title' and 'description' strings.

    Example:
        >>> ticket = render_ticket("drift_detection", test_name="test_income_drift",
        ...     method="PSI", statistic=0.35, threshold=0.2)
        >>> print(ticket["title"])
    """
    template = _TEMPLATES.get(template_name, _TEMPLATES["default"])

    # Fill template, using empty string for missing keys
    description = template.format_map(_DefaultDict(kwargs))
    title = description.split("\n")[0]

    return {"title": title, "description": description}


class _DefaultDict(dict):  # type: ignore[type-arg]
    """Dict that returns '{key}' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
