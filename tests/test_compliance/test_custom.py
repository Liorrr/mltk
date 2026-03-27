"""Tests for custom compliance framework builder (mltk.compliance.custom).

Covers: YAML loading (valid + invalid), result mapping, gap detection,
coverage assertion pass/fail, edge cases (empty categories, missing file),
and timing decorator.

The custom framework builder lets organisations define their own compliance
requirements in YAML and get the same gap analysis and coverage assertions
that built-in frameworks (HIPAA, NIST, EU AI Act) provide.  These tests
ensure the loader validates YAML correctly, the mapper groups results by
category, and the CI gate assertion passes/fails at the right thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.compliance.custom import (
    assert_custom_coverage,
    find_custom_gaps,
    load_custom_framework,
    map_results_to_custom,
)
from mltk.core.assertion import MltkAssertionError

# ---------------------------------------------------------------------------
# YAML content constants
# ---------------------------------------------------------------------------

VALID_YAML = """\
name: "Test ML Policy"
version: "2.0"
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
    description: "Models must pass accuracy and fairness checks"
    assertions:
      - "model.metric"
      - "model.regression"
      - "model.bias"
  monitoring:
    title: "Production Monitoring"
    assertions:
      - "monitor.degradation"
      - "monitor.sla"
"""

YAML_NO_CATEGORIES = """\
name: "Empty Framework"
version: "1.0"
"""

YAML_MISSING_NAME = """\
version: "1.0"
categories:
  x:
    title: "X"
    assertions: []
"""

YAML_BAD_ASSERTIONS_TYPE = """\
name: "Bad Types"
categories:
  x:
    title: "X"
    assertions: "not-a-list"
"""

YAML_MISSING_TITLE = """\
name: "Missing Title"
categories:
  x:
    assertions:
      - "data.schema"
"""

YAML_EMPTY_CATEGORIES = """\
name: "No Categories Content"
version: "1.0"
categories: {}
"""

# ---------------------------------------------------------------------------
# Sample result fixtures
# ---------------------------------------------------------------------------

# Covers all 3 categories in VALID_YAML
FULL_RESULTS: list[dict] = [
    {"name": "data.schema.columns",      "passed": True,  "message": "schema ok"},
    {"name": "data.no_nulls.train",      "passed": True,  "message": "0 nulls"},
    {"name": "data.drift.psi",           "passed": True,  "message": "PSI=0.03"},
    {"name": "model.metric.accuracy",    "passed": True,  "message": "accuracy=0.94"},
    {"name": "model.regression.v2_vs_v1","passed": True,  "message": "no regression"},
    {"name": "model.bias.dp",            "passed": False, "message": "disparity=0.12"},
    {"name": "monitor.degradation.week", "passed": True,  "message": "stable"},
    {"name": "monitor.sla.uptime",       "passed": True,  "message": "99.95%"},
]

# Covers only data_quality (data.schema) -- 1 of 3 categories
PARTIAL_RESULTS: list[dict] = [
    {"name": "data.schema.columns", "passed": True, "message": "ok"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str, filename: str = "framework.yaml") -> Path:
    """Write YAML content to a temp file and return its path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1 -- load_custom_framework: valid YAML parsed correctly
# ---------------------------------------------------------------------------


def test_load_valid_yaml(tmp_path: Path):
    """A well-formed YAML file must parse into a normalized framework dict.

    WHY: This is the happy path.  The loader must extract name, version,
    and all categories with their titles, descriptions, and assertion
    lists.  Downstream functions depend on this exact structure.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    fw = load_custom_framework(str(yaml_path))

    assert fw["name"] == "Test ML Policy"
    assert fw["version"] == "2.0"
    assert len(fw["categories"]) == 3

    # data_quality
    dq = fw["categories"]["data_quality"]
    assert dq["title"] == "Data Quality Requirements"
    assert dq["description"] == "All training data must pass quality gates"
    assert dq["assertions"] == ["data.schema", "data.no_nulls", "data.drift"]

    # model_validation
    mv = fw["categories"]["model_validation"]
    assert mv["title"] == "Model Validation"
    assert "model.metric" in mv["assertions"]

    # monitoring -- no description provided, should default to ""
    mon = fw["categories"]["monitoring"]
    assert mon["title"] == "Production Monitoring"
    assert mon["description"] == ""
    assert "monitor.degradation" in mon["assertions"]


# ---------------------------------------------------------------------------
# Test 2 -- load_custom_framework: invalid YAML errors gracefully
# ---------------------------------------------------------------------------


def test_load_missing_file():
    """Loading a non-existent file must raise FileNotFoundError.

    WHY: Clear error messages prevent debugging confusion.  The user
    must know immediately that the file path is wrong.
    """
    with pytest.raises(FileNotFoundError, match="not found"):
        load_custom_framework("/nonexistent/path/framework.yaml")


def test_load_missing_name(tmp_path: Path):
    """YAML without a 'name' key must raise ValueError.

    WHY: The framework name is required for report headers and
    assertion naming.  Omitting it is a structural error.
    """
    yaml_path = _write_yaml(tmp_path, YAML_MISSING_NAME)
    with pytest.raises(ValueError, match="name"):
        load_custom_framework(str(yaml_path))


def test_load_missing_categories(tmp_path: Path):
    """YAML without a 'categories' key must raise ValueError.

    WHY: A framework with no categories is useless for compliance
    mapping.  The error must be clear about what is missing.
    """
    yaml_path = _write_yaml(tmp_path, YAML_NO_CATEGORIES)
    with pytest.raises(ValueError, match="categories"):
        load_custom_framework(str(yaml_path))


def test_load_bad_assertions_type(tmp_path: Path):
    """Category with assertions as a string (not list) must raise ValueError.

    WHY: Assertions must be a list of prefix strings for startswith
    matching.  A string value is a common YAML mistake that must be
    caught early with a clear message.
    """
    yaml_path = _write_yaml(tmp_path, YAML_BAD_ASSERTIONS_TYPE)
    with pytest.raises(ValueError, match="list"):
        load_custom_framework(str(yaml_path))


def test_load_missing_title(tmp_path: Path):
    """Category without a 'title' must raise ValueError.

    WHY: The title is required for report section headers.
    """
    yaml_path = _write_yaml(tmp_path, YAML_MISSING_TITLE)
    with pytest.raises(ValueError, match="title"):
        load_custom_framework(str(yaml_path))


# ---------------------------------------------------------------------------
# Test 3 -- map_results_to_custom: groups results correctly
# ---------------------------------------------------------------------------


def test_map_results_groups_correctly(tmp_path: Path):
    """Results must land in the correct custom category buckets.

    WHY: The mapper drives per-category report sections.  Each assertion
    prefix must route to the category that declares it.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    fw = load_custom_framework(str(yaml_path))

    grouped = map_results_to_custom(FULL_RESULTS, fw)

    # data_quality
    assert "data_quality" in grouped
    dq_names = [r["name"] for r in grouped["data_quality"]]
    assert "data.schema.columns" in dq_names
    assert "data.no_nulls.train" in dq_names
    assert "data.drift.psi" in dq_names

    # model_validation
    assert "model_validation" in grouped
    mv_names = [r["name"] for r in grouped["model_validation"]]
    assert "model.metric.accuracy" in mv_names
    assert "model.regression.v2_vs_v1" in mv_names
    assert "model.bias.dp" in mv_names

    # monitoring
    assert "monitoring" in grouped
    mon_names = [r["name"] for r in grouped["monitoring"]]
    assert "monitor.degradation.week" in mon_names
    assert "monitor.sla.uptime" in mon_names

    # Verify enriched "category" key
    for cat_id, items in grouped.items():
        if cat_id == "uncategorised":
            continue
        for item in items:
            assert item["category"] == cat_id


# ---------------------------------------------------------------------------
# Test 4 -- find_custom_gaps: identifies uncovered categories
# ---------------------------------------------------------------------------


def test_find_gaps_partial(tmp_path: Path):
    """Partial coverage must identify exactly which categories are missing.

    PARTIAL_RESULTS covers only data_quality (data.schema).
    model_validation and monitoring should be reported as gaps.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    fw = load_custom_framework(str(yaml_path))

    gaps = find_custom_gaps(PARTIAL_RESULTS, fw)

    assert "data_quality" not in gaps, "data_quality covered by data.schema"
    assert "model_validation" in gaps, "model_validation has no results"
    assert "monitoring" in gaps, "monitoring has no results"


def test_find_gaps_full_coverage(tmp_path: Path):
    """Full coverage must produce zero gaps."""
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    fw = load_custom_framework(str(yaml_path))

    gaps = find_custom_gaps(FULL_RESULTS, fw)
    assert gaps == [], f"Expected no gaps but got: {gaps}"


def test_find_gaps_empty_results(tmp_path: Path):
    """No results must report all categories as gaps."""
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    fw = load_custom_framework(str(yaml_path))

    gaps = find_custom_gaps([], fw)
    assert len(gaps) == 3
    assert gaps == sorted(["data_quality", "model_validation", "monitoring"])


# ---------------------------------------------------------------------------
# Test 5 -- assert_custom_coverage: pass
# ---------------------------------------------------------------------------


def test_coverage_assertion_pass(tmp_path: Path):
    """Full coverage at 80% threshold must pass.

    WHY: This is the CI gate happy path.  3/3 categories = 100% >= 80%.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)

    result = assert_custom_coverage(
        FULL_RESULTS, str(yaml_path), min_coverage=0.8,
    )

    assert result.passed is True
    assert result.details["covered_count"] == 3
    assert result.details["total"] == 3
    assert result.details["coverage"] == 1.0
    assert result.details["framework_name"] == "Test ML Policy"
    assert result.details["gaps"] == []
    assert "meets" in result.message
    assert result.name == "compliance.custom.coverage"


# ---------------------------------------------------------------------------
# Test 6 -- assert_custom_coverage: fail
# ---------------------------------------------------------------------------


def test_coverage_assertion_fail(tmp_path: Path):
    """Partial coverage below threshold must raise MltkAssertionError.

    PARTIAL_RESULTS covers 1/3 categories (33%) < 80%.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_custom_coverage(
            PARTIAL_RESULTS, str(yaml_path), min_coverage=0.8,
        )

    result = exc_info.value.result
    assert result.passed is False
    assert result.details["coverage"] < 0.8
    assert "below" in result.message
    assert len(result.details["gaps"]) == 2


# ---------------------------------------------------------------------------
# Test 7 -- YAML with empty categories dict handled
# ---------------------------------------------------------------------------


def test_empty_categories_framework(tmp_path: Path):
    """A framework with zero categories should report 100% coverage.

    WHY: If an organisation defines a framework with no categories (yet),
    the coverage assertion should pass vacuously -- there is nothing to
    cover.  This prevents false failures during framework development.
    """
    yaml_path = _write_yaml(tmp_path, YAML_EMPTY_CATEGORIES)

    result = assert_custom_coverage([], str(yaml_path), min_coverage=0.8)

    assert result.passed is True
    assert result.details["total"] == 0
    assert result.details["covered_count"] == 0
    assert result.details["coverage"] == 1.0
    assert result.details["gaps"] == []


# ---------------------------------------------------------------------------
# Test 8 -- Timing decorator populates duration_ms
# ---------------------------------------------------------------------------


def test_coverage_assertion_timing(tmp_path: Path):
    """The @timed_assertion decorator must populate duration_ms.

    WHY: All mltk assertions must report execution time for performance
    tracking and SLA monitoring.  This verifies the decorator is applied.
    """
    yaml_path = _write_yaml(tmp_path, VALID_YAML)

    result = assert_custom_coverage(
        FULL_RESULTS, str(yaml_path), min_coverage=0.1,
    )

    assert isinstance(result.duration_ms, float)
    assert result.duration_ms >= 0.0
