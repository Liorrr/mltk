"""ML Test Score -- Google's 28-test rubric for ML production readiness.

Score = minimum across 4 categories. Weakest category determines overall readiness.
"""

from __future__ import annotations

from typing import Any

# Category mappings: test name prefix → category
_CATEGORY_MAP: dict[str, str] = {
    "data.schema": "data",
    "data.no_nulls": "data",
    "data.dtypes": "data",
    "data.range": "data",
    "data.unique": "data",
    "data.no_outliers": "data",
    "data.freshness": "data",
    "data.row_count": "data",
    "data.drift": "data",
    "data.pii": "data",
    "data.label": "data",
    "model.metric": "model",
    "model.no_regression": "model",
    "model.slice": "model",
    "model.calibration": "model",
    "model.bias": "model",
    "model.robust": "model",
    "inference.latency": "infrastructure",
    "inference.cold_start": "infrastructure",
    "inference.throughput": "infrastructure",
    "inference.contract": "infrastructure",
    "pipeline": "infrastructure",
}

_CATEGORIES = ["data", "model", "infrastructure", "monitoring"]
_MAX_PER_CATEGORY = 7


def compute_ml_test_score(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute ML Test Score from test results.

    Args:
        results: List of test result dicts with 'outcome' and optional 'ml_result'.

    Returns:
        Dict with total score, max, percentage, and per-category breakdown.
    """
    category_counts: dict[str, dict[str, int]] = {
        cat: {"passed": 0, "total": 0} for cat in _CATEGORIES
    }

    for r in results:
        ml_result = r.get("ml_result")
        if ml_result and hasattr(ml_result, "name"):
            name = ml_result.name
        else:
            # Try to infer category from nodeid
            nodeid = str(r.get("nodeid", ""))
            if "test_data" in nodeid:
                name = "data."
            elif "test_model" in nodeid:
                name = "model."
            elif "test_inference" in nodeid:
                name = "inference."
            elif "test_pipeline" in nodeid:
                name = "pipeline"
            else:
                continue

        # Find category
        category = None
        for prefix, cat in _CATEGORY_MAP.items():
            if name.startswith(prefix):
                category = cat
                break
        if category is None:
            # Default based on name
            for cat in _CATEGORIES:
                if cat in name.lower():
                    category = cat
                    break

        if category and category in category_counts:
            category_counts[category]["total"] += 1
            if r.get("outcome") == "passed":
                category_counts[category]["passed"] += 1

    # Compute scores per category (max 7 each)
    categories = {}
    for cat in _CATEGORIES:
        counts = category_counts[cat]
        score = min(counts["passed"], _MAX_PER_CATEGORY)
        categories[cat] = {
            "score": score,
            "max": _MAX_PER_CATEGORY,
            "tests_passed": counts["passed"],
            "tests_total": counts["total"],
        }

    total = sum(c["score"] for c in categories.values())
    max_total = _MAX_PER_CATEGORY * len(_CATEGORIES)
    percentage = (total / max_total * 100) if max_total > 0 else 0

    return {
        "total": total,
        "max": max_total,
        "percentage": round(percentage, 1),
        "categories": categories,
    }
