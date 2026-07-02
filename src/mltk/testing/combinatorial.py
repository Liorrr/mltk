"""NIST-style combinatorial (covering-array) test coverage measurement."""
from __future__ import annotations

import itertools
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

_UNCOVERED_SAMPLE = 10

# Sentinel distinguishing "test case omits this parameter" from "value is None".
_MISSING = object()


def _dedupe(values: list[Any]) -> list[Any]:
    """Return *values* with duplicates removed, order preserved.

    Uses equality membership (not hashing) so unhashable parameter values are
    tolerated.
    """
    seen: list[Any] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def combinatorial_coverage(
    test_cases: list[dict[str, Any]],
    parameters: dict[str, list[Any]],
    strength: int = 2,
) -> dict[str, Any]:
    """Compute t-way (combinatorial) coverage of *test_cases* over *parameters*.

    For each size-*strength* subset of parameter names, every Cartesian product of
    the corresponding allowed values is a required value-combination. A test case
    covers a combination when it maps every (param, value) pair in that combination
    to the same value.

    Args:
        test_cases: List of dicts mapping parameter name to its chosen value.
        parameters: Mapping from parameter name to its list of allowed values.
        strength: Coverage strength *t* (default 2 = pairwise). Must satisfy
            ``1 <= strength <= len(parameters)``.

    Returns:
        Dict with keys:
        - ``covered`` (int): number of required value-combinations covered.
        - ``total`` (int): total number of required value-combinations.
        - ``coverage`` (float): fraction covered; 1.0 when total is 0.
        - ``uncovered`` (list[dict]): sample of up to 10 uncovered combinations.

    Raises:
        ValueError: If *parameters* is empty, any value list is empty, or
            *strength* is outside [1, len(parameters)].

    Example:
        >>> params = {"a": [0, 1], "b": [0, 1]}
        >>> cases = [{"a": 0, "b": 0}, {"a": 1, "b": 1}]
        >>> combinatorial_coverage(cases, params, strength=2)["total"]
        4
    """
    if not parameters:
        raise ValueError("parameters must be non-empty")
    if strength < 1 or strength > len(parameters):
        raise ValueError(
            f"strength must be between 1 and len(parameters)={len(parameters)}, got {strength}"
        )

    empty_params = [name for name, vals in parameters.items() if not vals]
    if empty_params:
        raise ValueError(f"parameters with empty value lists: {empty_params}")

    # Dedupe per-parameter values so repeats do not skew total/covered counts.
    parameters = {name: _dedupe(vals) for name, vals in parameters.items()}

    param_names = list(parameters.keys())
    covered = 0
    total = 0
    uncovered: list[dict[str, Any]] = []

    for name_subset in itertools.combinations(param_names, strength):
        value_lists = [parameters[n] for n in name_subset]
        for value_combo in itertools.product(*value_lists):
            required: dict[str, Any] = dict(zip(name_subset, value_combo, strict=True))
            total += 1
            is_covered = any(
                all(tc.get(p, _MISSING) == v for p, v in required.items())
                for tc in test_cases
            )
            if is_covered:
                covered += 1
            elif len(uncovered) < _UNCOVERED_SAMPLE:
                uncovered.append(required)

    coverage = covered / total if total > 0 else 1.0

    return {
        "covered": covered,
        "total": total,
        "coverage": coverage,
        "uncovered": uncovered,
    }


@timed_assertion
def assert_combinatorial_coverage(
    test_cases: list[dict[str, Any]],
    parameters: dict[str, list[Any]],
    *,
    strength: int = 2,
    min_coverage: float = 1.0,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that test_cases achieve at least min_coverage t-way coverage.

    Enumerates every size-*strength* subset of parameter names and checks that
    the corresponding Cartesian-product value-combinations are covered by at
    least one entry in *test_cases*.

    Args:
        test_cases: List of dicts mapping parameter name to its chosen value.
        parameters: Mapping from parameter name to its list of allowed values.
        strength: Coverage strength *t*; default 2 (pairwise).
        min_coverage: Minimum required coverage fraction in [0, 1]; default 1.0.
        severity: Severity level; CRITICAL (default) raises on failure.

    Returns:
        TestResult with coverage details (covered, total, coverage, strength,
        uncovered) in ``result.details``.

    Raises:
        ValueError: If *parameters* is empty, any value list is empty, or
            *strength* is outside valid range.
        MltkAssertionError: If coverage < min_coverage and severity is CRITICAL.

    Example:
        >>> params = {"a": [0, 1], "b": [0, 1]}
        >>> cases = [{"a": 0, "b": 0}, {"a": 0, "b": 1}, {"a": 1, "b": 0}, {"a": 1, "b": 1}]
        >>> assert_combinatorial_coverage(cases, params).passed
        True
    """
    result = combinatorial_coverage(test_cases, parameters, strength)
    cov: float = result["coverage"]
    covered: int = result["covered"]
    total: int = result["total"]
    uncovered: list[dict[str, Any]] = result["uncovered"]
    n_uncovered = total - covered

    passed = cov >= min_coverage
    message = (
        f"{strength}-way coverage {cov:.2f} ({covered}/{total} combinations)"
        + (f"; {n_uncovered} uncovered" if n_uncovered > 0 else "")
    )

    return assert_true(
        passed,
        name="testing.combinatorial_coverage",
        message=message,
        severity=severity,
        covered=covered,
        total=total,
        coverage=cov,
        strength=strength,
        uncovered=uncovered,
    )
