"""Tests for mltk.testing.combinatorial — combinatorial coverage measurement."""
from __future__ import annotations

import itertools

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.testing.combinatorial import assert_combinatorial_coverage, combinatorial_coverage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 3 binary parameters — pairwise total = C(3,2) * 2 * 2 = 12
_BINARY_3 = {"a": [0, 1], "b": [0, 1], "c": [0, 1]}


def _full_cartesian(parameters: dict) -> list[dict]:
    """Return the full Cartesian product of *parameters* as a list of dicts."""
    names = list(parameters.keys())
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*parameters.values())
    ]


# ---------------------------------------------------------------------------
# Tests: combinatorial_coverage (pure helper)
# ---------------------------------------------------------------------------


def test_pairwise_total_binary_3():
    # SCENARIO: 3 binary params, strength=2 (pairwise)
    # WHY: C(3,2) * 2 * 2 = 12 is the canonical pairwise total for 3 binary params
    # EXPECTED: total == 12
    result = combinatorial_coverage([], _BINARY_3, strength=2)

    assert result["total"] == 12


def test_full_cartesian_full_coverage():
    # SCENARIO: test cases are the full Cartesian product of the parameter space
    # WHY: every pairwise combination is represented → coverage must be 1.0
    # EXPECTED: coverage == 1.0, covered == total, uncovered == []
    params = {"x": [0, 1, 2], "y": ["a", "b"]}
    test_cases = _full_cartesian(params)
    result = combinatorial_coverage(test_cases, params, strength=2)

    assert result["coverage"] == pytest.approx(1.0)
    assert result["covered"] == result["total"]
    assert result["uncovered"] == []


def test_empty_test_cases_zero_coverage():
    # SCENARIO: no test cases, non-empty parameter space
    # WHY: 0 cases cannot cover any combination → coverage must be 0.0
    # EXPECTED: covered == 0, coverage == 0.0, uncovered is non-empty
    result = combinatorial_coverage([], _BINARY_3, strength=2)

    assert result["covered"] == 0
    assert result["coverage"] == pytest.approx(0.0)
    assert len(result["uncovered"]) > 0


def test_strength_1_value_coverage():
    # SCENARIO: strength=1 — each (param, value) singleton must appear
    # WHY: t=1 is pure value coverage; total == sum of |values| per param
    # EXPECTED: total == 3+2==5, coverage==1.0 with full Cartesian test set
    params = {"p": [1, 2, 3], "q": ["x", "y"]}
    test_cases = _full_cartesian(params)
    result = combinatorial_coverage(test_cases, params, strength=1)

    assert result["total"] == 3 + 2  # |p| + |q|
    assert result["coverage"] == pytest.approx(1.0)


def test_partial_coverage():
    # SCENARIO: a single test case covers some but not all pairwise combinations
    # WHY: real suites have partial coverage; the metric must track it accurately
    # EXPECTED: 0 < coverage < 1.0, covered < total, uncovered is non-empty
    params = {"a": [0, 1], "b": [0, 1], "c": [0, 1]}
    test_cases = [{"a": 0, "b": 0, "c": 0}]
    result = combinatorial_coverage(test_cases, params, strength=2)

    assert 0.0 < result["coverage"] < 1.0
    assert result["covered"] < result["total"]
    assert len(result["uncovered"]) > 0


def test_uncovered_sample_capped_at_10():
    # SCENARIO: large uncovered set (many missing combinations)
    # WHY: uncovered is a sample; must never exceed 10 entries
    # EXPECTED: len(uncovered) <= 10
    params = {f"p{i}": list(range(5)) for i in range(4)}
    result = combinatorial_coverage([], params, strength=2)

    assert len(result["uncovered"]) <= 10


def test_result_structure_keys():
    # SCENARIO: verify helper always returns exactly the documented keys
    # WHY: callers destructure this dict — unexpected extra or missing keys break them
    # EXPECTED: exactly {"covered", "total", "coverage", "uncovered"}
    result = combinatorial_coverage([{"a": 1}], {"a": [1, 2]}, strength=1)

    assert set(result.keys()) == {"covered", "total", "coverage", "uncovered"}


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------


def test_strength_too_large_raises():
    # SCENARIO: strength > len(parameters) == 3
    # WHY: C(3, 4) is undefined; must raise ValueError with "strength" in message
    # EXPECTED: ValueError
    with pytest.raises(ValueError, match="strength"):
        combinatorial_coverage([], _BINARY_3, strength=4)


def test_empty_parameters_raises():
    # SCENARIO: parameters dict is empty
    # WHY: no parameter space makes the assertion meaningless; must reject early
    # EXPECTED: ValueError mentioning "parameters"
    with pytest.raises(ValueError, match="parameters"):
        combinatorial_coverage([], {}, strength=1)


def test_strength_zero_raises():
    # SCENARIO: strength=0 is below the minimum of 1
    # WHY: t=0 coverage is vacuously true and computationally undefined; must reject
    # EXPECTED: ValueError mentioning "strength"
    with pytest.raises(ValueError, match="strength"):
        combinatorial_coverage([], _BINARY_3, strength=0)


# ---------------------------------------------------------------------------
# Tests: assert_combinatorial_coverage (assertion wrapper)
# ---------------------------------------------------------------------------


def test_assert_full_coverage_passes():
    # SCENARIO: full Cartesian product → coverage 1.0 >= min_coverage 1.0
    # WHY: the assertion must return a passing TestResult and not raise
    # EXPECTED: result.passed is True, name is correct
    params = {"a": [0, 1], "b": [0, 1]}
    test_cases = _full_cartesian(params)
    result = assert_combinatorial_coverage(test_cases, params)

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.name == "testing.combinatorial_coverage"


def test_assert_empty_tests_critical_raises():
    # SCENARIO: no test cases, default CRITICAL severity, min_coverage=1.0
    # WHY: coverage 0.0 < 1.0 with CRITICAL must raise MltkAssertionError
    # EXPECTED: MltkAssertionError is raised
    with pytest.raises(MltkAssertionError):
        assert_combinatorial_coverage([], _BINARY_3)


def test_assert_warning_severity_no_raise():
    # SCENARIO: coverage below min_coverage, severity=WARNING
    # WHY: WARNING must not raise — it returns a failing TestResult instead
    # EXPECTED: result.passed is False, no exception raised
    result = assert_combinatorial_coverage([], _BINARY_3, severity=Severity.WARNING)

    assert isinstance(result, TestResult)
    assert result.passed is False


def test_assert_partial_coverage_above_threshold():
    # SCENARIO: one test case achieves 1/4 pairwise coverage, min_coverage=0.25
    # WHY: callers may accept less than 100 % coverage; the assertion must honour it
    # EXPECTED: result.passed is True (0.25 >= 0.25)
    params = {"a": [0, 1], "b": [0, 1]}
    # Covers (a=0,b=0) only → 1 of 4 pairwise pairs → coverage=0.25
    test_cases = [{"a": 0, "b": 0}]
    result = assert_combinatorial_coverage(
        test_cases, params, min_coverage=0.25, severity=Severity.WARNING
    )

    assert result.passed is True


def test_assert_result_details_keys():
    # SCENARIO: successful assertion — all documented detail keys must be present
    # WHY: callers read details to build coverage reports
    # EXPECTED: details contains covered, total, coverage, strength, uncovered
    params = {"x": [1, 2], "y": ["a", "b"]}
    test_cases = _full_cartesian(params)
    result = assert_combinatorial_coverage(test_cases, params, strength=2)

    for key in ("covered", "total", "coverage", "strength", "uncovered"):
        assert key in result.details


def test_assert_strength_1_passes():
    # SCENARIO: strength=1 (value coverage), every allowed value appears
    # WHY: t=1 is a supported mode; the assertion must accept the strength kwarg
    # EXPECTED: result.passed is True when each value appears at least once
    params = {"color": ["red", "blue"], "size": ["S", "M", "L"]}
    test_cases = [
        {"color": "red", "size": "S"},
        {"color": "blue", "size": "M"},
        {"color": "red", "size": "L"},
    ]
    result = assert_combinatorial_coverage(test_cases, params, strength=1)

    assert result.passed is True


def test_assert_duration_ms_populated():
    # SCENARIO: timed_assertion decorator wraps the function
    # WHY: TestResult.duration_ms feeds into reports; must be non-negative
    # EXPECTED: duration_ms >= 0.0
    params = {"a": [0, 1]}
    test_cases = _full_cartesian(params)
    result = assert_combinatorial_coverage(test_cases, params, strength=1)

    assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Review-fix regression: missing-key vs None, dedupe, empty value lists
# ---------------------------------------------------------------------------


def test_missing_key_not_confused_with_none_value():
    # SCENARIO: parameter 'a' allows None; the test case omits 'a' entirely
    # WHY: tc.get('a') would return None and falsely "cover" the (a=None) combo
    # EXPECTED: neither (a=None) nor (a=1) is covered -> covered == 0
    result = combinatorial_coverage([{"b": 5}], {"a": [None, 1]}, strength=1)
    assert result["covered"] == 0
    assert result["total"] == 2


def test_explicit_none_value_is_covered():
    # SCENARIO: a test case explicitly sets a=None
    # EXPECTED: the (a=None) combination is genuinely covered
    result = combinatorial_coverage([{"a": None}], {"a": [None, 1]}, strength=1)
    assert result["covered"] == 1


def test_duplicate_values_deduped_before_counting():
    # SCENARIO: a parameter value list contains a duplicate (0 twice)
    # WHY: duplicates would inflate total/covered and skew the coverage ratio
    # EXPECTED: deduped to {0, 1} -> total == 2, coverage == 0.5 (not 0.667)
    result = combinatorial_coverage([{"a": 0}], {"a": [0, 0, 1]}, strength=1)
    assert result["total"] == 2
    assert result["coverage"] == 0.5


def test_empty_value_list_raises():
    # SCENARIO: a parameter has an empty value list (misconfigured matrix)
    # EXPECTED: ValueError instead of a silent vacuous 1.0 / pass
    with pytest.raises(ValueError, match="empty value lists"):
        assert_combinatorial_coverage([], {"a": [], "b": [0, 1]}, strength=2)
