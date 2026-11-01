"""Pipeline stage compatibility testing -- validate inter-stage schema flow in ML pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@dataclass
class StageSpec:
    """Specification for a single pipeline stage's schema contract.

    Attributes:
        name: Human-readable name for the stage (e.g. ``"feature_engineering"``).
        produces: Mapping of column name to dtype string this stage outputs.
            Example: ``{"x": "float64", "label": "int64"}``.
        requires: Mapping of column name to dtype string this stage expects as input.
            For the **first** stage, ``requires`` describes raw/external input and is
            never validated against upstream produces.
    """

    name: str
    produces: dict[str, str] = field(default_factory=dict)
    requires: dict[str, str] = field(default_factory=dict)


def _canonical_dtype(dtype: str) -> str:
    """Normalize common dtype spellings for compatibility checks."""
    dtype_text = str(dtype).strip()
    lowered = dtype_text.lower()
    aliases = {
        "int": "int64",
        "integer": "int64",
        "float": "float64",
    }
    if lowered in aliases:
        return aliases[lowered]

    try:
        return np.dtype(dtype_text).name.lower()
    except (TypeError, ValueError):
        return lowered


def _dtype_compatible(produced: str, required: str, *, allow_widening: bool) -> bool:
    """Return True if *produced* satisfies *required*.

    Exact match after canonicalization always succeeds. When
    ``allow_widening`` is True, a produced dtype that NumPy can safely
    cast to the required dtype also succeeds (e.g. int32 → int64).
    """
    left = _canonical_dtype(produced)
    right = _canonical_dtype(required)
    if left == right:
        return True
    if not allow_widening:
        return False
    try:
        return bool(np.can_cast(np.dtype(left), np.dtype(right), casting="safe"))
    except (TypeError, ValueError):
        return False


@timed_assertion
def assert_pipeline_stages_compatible(
    stages: list[StageSpec],
    *,
    check_dtypes: bool = True,
    allow_widening: bool = False,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that consecutive pipeline stages are schema-compatible.

    Validates inter-stage schema flow: each stage's required columns must be
    produced by at least one upstream stage. The first stage's requirements
    represent raw/external input and are **never** validated against upstream
    output (document this in ``StageSpec.requires`` for clarity, but it will
    not cause failures).

    A cumulative available schema is built as stages are visited; a stage may
    consume any column produced by *any* earlier stage -- not only the
    immediately preceding one. Later ``produces`` override earlier ones when
    the same column name appears in multiple stages.

    Args:
        stages: Ordered list of StageSpec objects from first to last stage.
        check_dtypes: When True, the required dtype must match the produced
            dtype after canonicalizing equivalent spellings. When False, only
            column presence is verified and dtype differences are silently ignored.
        allow_widening: When True (and ``check_dtypes`` is True), a produced
            dtype that NumPy can safely cast to the required dtype is accepted
            (e.g. ``int32`` satisfies ``int64``). Default False preserves
            exact-match behaviour.
        severity: Severity level applied on failure. CRITICAL raises
            ``MltkAssertionError``; WARNING and INFO return a failed
            ``TestResult`` without raising.

    Returns:
        TestResult with ``n_stages``, ``check_dtypes``, ``allow_widening``, and
        ``incompatibilities`` (list of per-stage dicts with keys ``from``,
        ``to``, ``missing``, and ``dtype_mismatch``) in ``details``.

    Example:
        >>> ingest = StageSpec("ingest", produces={"x": "float64"})
        >>> model = StageSpec("model", requires={"x": "float64"})
        >>> result = assert_pipeline_stages_compatible([ingest, model])
        >>> result.passed
        True
    """
    n_stages = len(stages)

    # Empty or single-stage list -- nothing to validate between stages
    if n_stages <= 1:
        note = "no stages" if n_stages == 0 else f"single stage '{stages[0].name}'"
        return assert_true(
            True,
            name="pipeline.stages_compatible",
            message=f"Stage compatibility not applicable: {note}",
            severity=severity,
            n_stages=n_stages,
            check_dtypes=check_dtypes,
            allow_widening=allow_widening,
            incompatibilities=[],
        )

    cumulative: dict[str, str] = {}
    incompatibilities: list[dict] = []

    for idx, stage in enumerate(stages):
        if idx > 0:
            # Validate this stage's requirements against all upstream produces
            missing: list[str] = []
            dtype_mismatch: list[dict[str, str]] = []

            for col, required_dtype in stage.requires.items():
                if col not in cumulative:
                    missing.append(col)
                elif check_dtypes:
                    available_dtype = cumulative[col]
                    if not _dtype_compatible(
                        available_dtype,
                        required_dtype,
                        allow_widening=allow_widening,
                    ):
                        dtype_mismatch.append(
                            {
                                "column": col,
                                "required": required_dtype,
                                "available": available_dtype,
                            }
                        )

            if missing or dtype_mismatch:
                incompatibilities.append(
                    {
                        "from": stages[idx - 1].name,
                        "to": stage.name,
                        "missing": missing,
                        "dtype_mismatch": dtype_mismatch,
                    }
                )

        # Merge this stage's produces into the cumulative schema
        cumulative.update(stage.produces)

    passed = len(incompatibilities) == 0

    if passed:
        message = f"{n_stages} stages compatible"
    else:
        n = len(incompatibilities)
        first = incompatibilities[0]
        if first["missing"]:
            col_desc = ", ".join(f"'{c}'" for c in first["missing"][:2])
            message = (
                f"{n} incompatibilit{'y' if n == 1 else 'ies'}: "
                f"stage '{first['to']}' requires column {col_desc} not produced upstream"
            )
        else:
            col_name = first["dtype_mismatch"][0]["column"]
            message = (
                f"{n} incompatibilit{'y' if n == 1 else 'ies'}: "
                f"stage '{first['to']}' dtype mismatch on column '{col_name}'"
            )

    return assert_true(
        passed,
        name="pipeline.stages_compatible",
        message=message,
        severity=severity,
        n_stages=n_stages,
        check_dtypes=check_dtypes,
        allow_widening=allow_widening,
        incompatibilities=incompatibilities,
    )
