"""Custom compliance framework builder -- define your own regulation in YAML.

Pure data + helper functions -- YAML parsing uses only the stdlib-compatible
``yaml`` library (PyYAML).

Why custom frameworks
---------------------
Every organisation has internal policies, industry-specific regulations,
or client contracts that define ML testing requirements beyond what any
single standard (EU AI Act, HIPAA, NIST) covers.  Examples:

- A hospital network's internal "ML Model Governance Policy v3.2"
- A fintech's SOC 2 Type II controls mapped to model testing
- A defence contractor's CMMC Level 3 requirements for AI systems
- A client SLA that requires specific bias and latency thresholds

Instead of waiting for mltk to add built-in support for YOUR regulation,
define it yourself in YAML and get the same gap analysis, coverage
assertions, and reporting that built-in frameworks provide.

YAML format
-----------
.. code-block:: yaml

    name: "My Company ML Policy"
    version: "1.0"
    categories:
      data_quality:
        title: "Data Quality Requirements"
        description: "All training data must pass quality gates"
        assertions:
          - "data.schema"
          - "data.no_nulls"
          - "data.drift"
      model_validation:
        title: "Model Validation"
        assertions:
          - "model.metric"
          - "model.no_regression"
          - "model.no_bias"

Each category works exactly like a built-in rule: the ``assertions``
list contains assertion-name prefixes matched via ``startswith``.

How it works under the hood
---------------------------
1. ``load_custom_framework(yaml_path)`` parses and validates the YAML,
   returning a normalized dict with ``name``, ``version``, and
   ``categories``.
2. ``map_results_to_custom(results, framework)`` groups test results
   by category, just like ``map_results_to_rules`` does for HIPAA.
3. ``find_custom_gaps(results, framework)`` identifies categories with
   no matching test results.
4. ``assert_custom_coverage(results, yaml_path, min_coverage)`` is the
   CI gate: load the framework, compute coverage, pass or fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------


def load_custom_framework(yaml_path: str) -> dict:
    """Load a custom compliance framework from a YAML file.

    The YAML file must contain at minimum a ``name`` key and a
    ``categories`` mapping.  Each category must have a ``title`` and
    an ``assertions`` list of assertion-name prefixes.

    Why YAML and not JSON?
    ~~~~~~~~~~~~~~~~~~~~~~
    YAML supports comments, which are essential for compliance documents
    where auditors annotate *why* a particular assertion maps to a
    particular policy clause.  JSON does not support comments.

    Validation
    ~~~~~~~~~~
    The loader validates the YAML structure and raises ``ValueError``
    with a clear message if:

    - The file cannot be parsed as YAML.
    - The top-level structure is not a dict.
    - The ``name`` key is missing.
    - ``categories`` is missing or is not a dict.
    - Any category is missing a ``title``.
    - Any category's ``assertions`` is not a list.

    Args:
        yaml_path: Path to the YAML file (absolute or relative).

    Returns:
        Normalized framework dict with keys: ``name`` (str),
        ``version`` (str), ``categories`` (dict of category dicts).
        Each category dict has ``title`` (str), ``description`` (str),
        and ``assertions`` (list of str).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is malformed or fails validation.

    Example:
        >>> fw = load_custom_framework("policy.yaml")
        >>> fw["name"]
        'My Company ML Policy'
        >>> list(fw["categories"].keys())
        ['data_quality', 'model_validation']
    """
    # Import yaml lazily so the module can be imported even if PyYAML
    # is not installed -- the ImportError surfaces only when called.
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for custom compliance frameworks. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom framework YAML not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc

    # --- Structural validation ---
    if not isinstance(data, dict):
        raise ValueError(
            "Custom framework YAML must be a mapping at the top level, "
            f"got {type(data).__name__}"
        )

    if "name" not in data:
        raise ValueError("Custom framework YAML must contain a 'name' key")

    categories = data.get("categories")
    if categories is None:
        raise ValueError(
            "Custom framework YAML must contain a 'categories' mapping"
        )
    if not isinstance(categories, dict):
        raise ValueError(
            f"'categories' must be a mapping, got {type(categories).__name__}"
        )

    # --- Normalize each category ---
    normalized_categories: dict[str, dict[str, Any]] = {}
    for cat_id, cat_data in categories.items():
        if not isinstance(cat_data, dict):
            raise ValueError(
                f"Category {cat_id!r} must be a mapping, "
                f"got {type(cat_data).__name__}"
            )
        if "title" not in cat_data:
            raise ValueError(f"Category {cat_id!r} is missing a 'title' key")

        assertions = cat_data.get("assertions", [])
        if not isinstance(assertions, list):
            raise ValueError(
                f"Category {cat_id!r} 'assertions' must be a list, "
                f"got {type(assertions).__name__}"
            )

        normalized_categories[cat_id] = {
            "title": str(cat_data["title"]),
            "description": str(cat_data.get("description", "")),
            "assertions": [str(a) for a in assertions],
        }

    return {
        "name": str(data["name"]),
        "version": str(data.get("version", "1.0")),
        "categories": normalized_categories,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assertion_to_category_ids(
    assertion_name: str,
    categories: dict[str, dict],
) -> list[str]:
    """Return all category IDs whose assertion prefixes match *assertion_name*.

    A match occurs when *assertion_name* starts with any prefix listed in
    ``categories[cat_id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name,
            e.g. ``"data.schema.columns"``.
        categories: The ``categories`` dict from a loaded framework.

    Returns:
        List of matching category IDs (may be empty; may contain more
        than one).
    """
    matched: list[str] = []
    for cat_id, meta in categories.items():
        for prefix in meta.get("assertions", []):
            if assertion_name.startswith(prefix):
                matched.append(cat_id)
                break  # one match per category is enough
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_results_to_custom(
    results: list[dict],
    framework: dict,
) -> dict[str, list]:
    """Group test results by custom framework categories.

    Each result dict is expected to have at least:

    - ``name`` (str): assertion name, e.g. ``"data.schema.columns"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any category prefix are placed in
    the special ``"uncategorised"`` bucket.

    Args:
        results: List of result dicts (from JSON or TestResult serialisation).
        framework: A loaded framework dict (from :func:`load_custom_framework`).

    Returns:
        Dict mapping category IDs to lists of result dicts.  Each
        result dict is enriched with a ``"category"`` key.

    Example:
        >>> fw = {"categories": {"dq": {"assertions": ["data.schema"]}}}
        >>> r = [{"name": "data.schema.cols", "passed": True, "message": "ok"}]
        >>> grouped = map_results_to_custom(r, fw)
        >>> "dq" in grouped
        True
    """
    categories = framework.get("categories", {})
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        cat_ids = _assertion_to_category_ids(name, categories)
        if not cat_ids:
            enriched = dict(r)
            enriched["category"] = "uncategorised"
            grouped.setdefault("uncategorised", []).append(enriched)
        else:
            for cat_id in cat_ids:
                enriched = dict(r)
                enriched["category"] = cat_id
                grouped.setdefault(cat_id, []).append(enriched)
    return grouped


def find_custom_gaps(
    results: list[dict],
    framework: dict,
) -> list[str]:
    """Find categories in a custom framework not covered by any test results.

    A category is considered *covered* if at least one result exists whose
    assertion name matches any of the category's assertion prefixes.
    Categories with empty assertion lists are always reported as gaps
    since they cannot be covered by automated tests.

    Args:
        results: List of result dicts (same format as
            :func:`map_results_to_custom`).
        framework: A loaded framework dict (from :func:`load_custom_framework`).

    Returns:
        Sorted list of category IDs that have no matching test results.

    Example:
        >>> fw = {"categories": {"dq": {"assertions": ["data.schema"]}}}
        >>> find_custom_gaps([], fw)
        ['dq']
    """
    categories = framework.get("categories", {})
    covered: set[str] = set()
    for r in results:
        name = str(r.get("name", ""))
        for cat_id in _assertion_to_category_ids(name, categories):
            covered.add(cat_id)
    all_categories = set(categories.keys())
    return sorted(all_categories - covered)


@timed_assertion
def assert_custom_coverage(
    results: list[dict],
    framework_yaml: str,
    min_coverage: float = 0.8,
) -> TestResult:
    """Assert that test results cover a custom compliance framework.

    This is the CI gate for custom frameworks.  It loads the YAML,
    computes coverage, and returns a pass/fail TestResult.

    Coverage is defined as::

        categories_with_at_least_one_test / total_categories

    Args:
        results: List of result dicts (same format as
            :func:`map_results_to_custom`).
        framework_yaml: Path to the YAML file defining the framework.
        min_coverage: Minimum fraction of categories that must be
            covered (0.0--1.0, default 0.8).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating pass/fail with
        ``framework_name``, ``covered_count``, ``total``, ``coverage``,
        ``min_coverage``, and ``gaps`` in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: If coverage is
            below *min_coverage*.
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is malformed.

    Example:
        >>> result = assert_custom_coverage([], "policy.yaml", min_coverage=0.0)
        >>> result.passed
        True
    """
    framework = load_custom_framework(framework_yaml)
    categories = framework.get("categories", {})
    total = len(categories)

    gaps = find_custom_gaps(results, framework)
    covered_count = total - len(gaps)
    coverage = covered_count / total if total > 0 else 1.0
    passed = coverage >= min_coverage

    fw_name = framework.get("name", "Custom Framework")
    message = (
        f"{fw_name} coverage {coverage:.0%} "
        f"({covered_count}/{total} categories) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
    )

    return assert_true(
        passed,
        name="compliance.custom.coverage",
        message=message,
        severity=Severity.CRITICAL,
        framework_name=fw_name,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        gaps=gaps,
    )
