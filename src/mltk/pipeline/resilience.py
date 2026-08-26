"""Pipeline resilience testing -- ML chaos engineering via fault injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

DEFAULT_FAULTS: tuple[str, ...] = (
    "null_injection",
    "empty_input",
    "dropped_column",
    "dtype_corruption",
    "scale_shift",
    "duplicate_rows",
    "single_row",
)


def apply_fault(
    df: pd.DataFrame,
    fault: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply a named fault to a copy of *df*; never mutates the original.

    Args:
        df: Source DataFrame (not modified).
        fault: Fault name; must be one of DEFAULT_FAULTS.
        rng: NumPy random Generator for deterministic stochastic faults.

    Returns:
        A new DataFrame with the requested fault applied.

    Raises:
        ValueError: If *fault* is not a recognised fault name.

    Example:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"a": [1.0, 2.0]})
        >>> apply_fault(df, "empty_input", np.random.default_rng(42))
        Empty DataFrame
        Columns: [a]
        Index: []
    """
    if fault not in DEFAULT_FAULTS:
        raise ValueError(
            f"Unknown fault {fault!r}. Valid faults: {list(DEFAULT_FAULTS)}"
        )

    out = df.copy()

    if fault == "null_injection":
        # Set ~20% of cells to NaN via a deterministic boolean mask.
        out = out.mask(rng.random(out.shape) < 0.2)

    elif fault == "empty_input":
        out = pd.DataFrame(columns=df.columns)

    elif fault == "dropped_column":
        if out.shape[1] > 0:
            out = out.iloc[:, :-1]

    elif fault == "dtype_corruption":
        numeric_cols = out.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            # Direct column assignment avoids pandas inplace-cast restriction
            # when converting a numeric column to string dtype.
            out[col] = out[col].astype(str)

    elif fault == "scale_shift":
        numeric_cols = out.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                out[col] = out[col] * 1e6

    elif fault == "duplicate_rows":
        out = pd.concat([df.copy(), df.copy()], ignore_index=True)

    elif fault == "single_row":
        out = df.iloc[[0]].copy() if len(df) > 0 else df.copy()

    return out


@timed_assertion
def assert_pipeline_resilient(
    pipeline_fn: Callable[[pd.DataFrame], Any],
    baseline_input: pd.DataFrame,
    *,
    faults: list[str] | None = None,
    max_failure_rate: float = 0.0,
    validate_output: Callable[[Any], bool] | None = None,
    severity: Severity = Severity.CRITICAL,
    seed: int = 42,
) -> TestResult:
    """Assert that *pipeline_fn* degrades gracefully under fault injection.

    Injects chaos faults into copies of *baseline_input*, calls *pipeline_fn*
    on each faulted frame, and asserts the crash rate stays within
    *max_failure_rate*. The original *baseline_input* is never mutated.

    Args:
        pipeline_fn: Callable accepting a pd.DataFrame; any return value is fine.
        baseline_input: Reference DataFrame; NEVER mutated by this function.
        faults: Fault names to inject; defaults to all entries in DEFAULT_FAULTS.
        max_failure_rate: Maximum tolerated crash fraction (0.0 = zero crashes allowed).
        validate_output: Optional predicate on the pipeline return value. When
            provided, a return that is not truthy under this callable counts as
            a failure (silent ``None``/garbage). Omitted: any non-raising call
            is treated as survived (legacy).
        severity: Severity level; CRITICAL raises on failure, WARNING/INFO only report.
        seed: Random seed for deterministic fault generation.

    Returns:
        TestResult with per-fault crash details and aggregate failure_rate.

    Raises:
        ValueError: If any entry in *faults* is not a recognised fault name.

    Example:
        >>> import pandas as pd
        >>> pipeline = lambda df: df.select_dtypes("number").sum().sum()
        >>> result = assert_pipeline_resilient(pipeline, pd.DataFrame({"x": [1.0, 2.0]}))
        >>> result.passed
        True
    """
    fault_names: list[str] = list(faults) if faults is not None else list(DEFAULT_FAULTS)

    unknown = [f for f in fault_names if f not in DEFAULT_FAULTS]
    if unknown:
        raise ValueError(
            f"Unknown fault(s): {unknown}. Valid faults: {list(DEFAULT_FAULTS)}"
        )

    rng = np.random.default_rng(seed)
    run_results: list[dict[str, Any]] = []

    for fault_name in fault_names:
        faulted = apply_fault(baseline_input, fault_name, rng)
        try:
            output = pipeline_fn(faulted)
        except Exception as exc:
            run_results.append({
                "fault": fault_name,
                "crashed": True,
                "invalid_output": False,
                "error": str(exc),
            })
            continue
        invalid = (
            validate_output is not None and not bool(validate_output(output))
        )
        run_results.append({
            "fault": fault_name,
            "crashed": False,
            "invalid_output": invalid,
            "error": None,
        })

    n_faults = len(fault_names)
    crashes = sum(1 for r in run_results if r["crashed"] or r["invalid_output"])
    survived = n_faults - crashes
    failure_rate = crashes / n_faults if n_faults > 0 else 0.0
    passed = failure_rate <= max_failure_rate

    message = (
        f"Pipeline survived {survived}/{n_faults} fault injections "
        f"(failure_rate={failure_rate:.2f})"
    )

    return assert_true(
        passed,
        name="pipeline.resilient",
        message=message,
        severity=severity,
        results=run_results,
        failure_rate=failure_rate,
        faults_run=n_faults,
    )
