"""YAML test definitions — declare and run ML data tests from YAML files.

This module provides a declarative interface for mltk assertions.
Instead of writing Python, define tests in YAML and run them against
any CSV or Parquet data source.

Example YAML::

    data_source: data/features.csv
    tests:
      - name: Check schema
        assertion: schema
        params:
          expected:
            id: int64
            score: float64

      - name: No nulls
        assertion: no_nulls

      - name: Score in range
        assertion: range
        params:
          column: score
          min_val: 0.0
          max_val: 1.0

Usage::

    from mltk.testdefs import load_test_suite, run_test_suite

    suite = load_test_suite("tests.yaml")
    results = run_test_suite(suite)
    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} passed")
"""

from mltk.testdefs.runner import run_red_team_suite, run_test_suite
from mltk.testdefs.schema import (
    CustomAttack,
    RedTeamDefaults,
    RedTeamSuiteYaml,
    TestDef,
    TestSuiteYaml,
    load_test_suite,
)

__all__ = [
    "CustomAttack",
    "RedTeamDefaults",
    "RedTeamSuiteYaml",
    "TestDef",
    "TestSuiteYaml",
    "load_test_suite",
    "run_red_team_suite",
    "run_test_suite",
]
