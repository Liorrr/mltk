"""Contract validator — run all checks from a contract against a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mltk.contracts.schema import Contract
from mltk.core.result import TestSuite


def validate_data(
    df: pd.DataFrame,
    contract_path: str | Path,
) -> TestSuite:
    """Validate a DataFrame against a data contract.

    Runs schema, null, range, unique, row count, and freshness checks
    based on the contract YAML spec.

    Args:
        df: DataFrame to validate.
        contract_path: Path to the contract YAML file.

    Returns:
        TestSuite with all check results.

    Example:
        >>> suite = validate_data(df, "contract.yaml")
        >>> print(f"{suite.passed_count}/{suite.total} passed")
    """
    contract = Contract.from_yaml(contract_path)
    suite = TestSuite()

    # Schema check: expected columns and types
    from mltk.data.schema import assert_schema

    expected_schema = {col.name: col.type for col in contract.columns}
    try:
        result = assert_schema(df, expected_schema)
        suite.add(result)
    except Exception as e:
        from mltk.core.result import Severity, TestResult

        suite.add(TestResult(
            name="contract.schema", passed=False,
            severity=Severity.CRITICAL, message=str(e),
        ))

    # Per-column checks
    for col in contract.columns:
        if col.name not in df.columns:
            continue

        # Null check
        if not col.nullable:
            from mltk.data.schema import assert_no_nulls

            try:
                result = assert_no_nulls(df, columns=[col.name])
                suite.add(result)
            except Exception as e:
                from mltk.core.result import Severity, TestResult

                suite.add(TestResult(
                    name=f"contract.no_nulls.{col.name}", passed=False,
                    severity=Severity.CRITICAL, message=str(e),
                ))

        # Range check
        if col.range is not None:
            from mltk.data.distribution import assert_range

            try:
                result = assert_range(df[col.name], min_val=col.range[0], max_val=col.range[1])
                suite.add(result)
            except Exception as e:
                from mltk.core.result import Severity, TestResult

                suite.add(TestResult(
                    name=f"contract.range.{col.name}", passed=False,
                    severity=Severity.CRITICAL, message=str(e),
                ))

        # Unique check
        if col.unique:
            from mltk.data.distribution import assert_unique

            try:
                result = assert_unique(df, columns=[col.name])
                suite.add(result)
            except Exception as e:
                from mltk.core.result import Severity, TestResult

                suite.add(TestResult(
                    name=f"contract.unique.{col.name}", passed=False,
                    severity=Severity.CRITICAL, message=str(e),
                ))

    # Quality checks
    q = contract.quality

    if q.min_rows is not None or q.max_rows is not None:
        from mltk.data.freshness import assert_row_count

        try:
            result = assert_row_count(df, min_rows=q.min_rows, max_rows=q.max_rows)
            suite.add(result)
        except Exception as e:
            from mltk.core.result import Severity, TestResult

            suite.add(TestResult(
                name="contract.row_count", passed=False,
                severity=Severity.CRITICAL, message=str(e),
            ))

    if q.freshness_days is not None and q.freshness_column is not None:
        from mltk.data.freshness import assert_freshness

        try:
            result = assert_freshness(
                df, date_column=q.freshness_column, max_age_days=q.freshness_days
            )
            suite.add(result)
        except Exception as e:
            from mltk.core.result import Severity, TestResult

            suite.add(TestResult(
                name="contract.freshness", passed=False,
                severity=Severity.CRITICAL, message=str(e),
            ))

    return suite
