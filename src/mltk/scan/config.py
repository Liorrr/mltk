"""Scan configuration and context dataclasses.

Two dataclasses live here:

:class:`ScanConfig`
    User-facing knobs that control how the scan behaves --
    row limits, timeouts, thresholds, and per-scanner
    overrides.  Immutable after construction (frozen).

:class:`ScanContext`
    Internal struct built by the engine before scanners run.
    Contains the model function, data, auto-detected column
    types, model type, and the config.  Scanners declare
    which fields they need via their ``requires`` set and
    the engine skips scanners whose requirements are not met.

Design notes:

- ``ScanConfig`` defaults are intentionally conservative:
  10K rows, 60s total budget, 30s per scanner.  These keep
  first-run experience fast on a laptop.
- ``categorical_threshold`` is the single source of truth
  for the numeric-vs-categorical split.  Both config and
  the detection logic use this value.
- ``scanner_config`` lets power users override individual
  scanner parameters without touching the scanner code::

      config = ScanConfig(
          scanner_config={
              "slice": {"metric": "f1"},
              "bias": {"threshold": 0.05},
          }
      )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["ScanConfig", "ScanContext"]


@dataclass(frozen=True)
class ScanConfig:
    """User-facing configuration for ``mltk scan``.

    All fields have sensible defaults so you can call
    ``scan(model, X, y)`` with zero config and get useful
    results.  Override individual fields for your use case.

    Attributes:
        max_scan_rows: Maximum rows to scan.  Larger datasets
            are sampled down to this size (preserving label
            distribution for classifiers).
        sample_strategy: How to sample when rows exceed
            ``max_scan_rows``.  ``"stratified"`` preserves
            class proportions; regressors use quantile-binned
            y for stratification.
        time_budget_seconds: Total wall-clock budget for the
            entire scan.  The engine stops launching new
            scanners once this is exceeded.
        per_scanner_timeout: Maximum seconds any single
            scanner may run before being interrupted.
        seed: Random seed for reproducibility (sampling,
            noise generation).
        categorical_threshold: A column with this many or
            fewer unique values is treated as categorical.
        max_slices_per_column: Maximum number of unique
            values to slice on per categorical column.
        min_slice_samples: Minimum samples in a slice for
            it to be tested (avoids noisy small-sample
            results).
        critical_drop: Performance drop (absolute) from
            overall that triggers a CRITICAL finding.
        warning_drop: Performance drop (absolute) from
            overall that triggers a WARNING finding.
        scanner_config: Per-scanner overrides as a dict of
            ``{scanner_name: {param: value}}``.
        enabled_scanners: If set, ONLY these scanners run.
            Mutually exclusive with ``disabled_scanners``.
        disabled_scanners: If set, these scanners are
            skipped.  Mutually exclusive with
            ``enabled_scanners``.
    """

    max_scan_rows: int = 10_000
    sample_strategy: str = "stratified"
    time_budget_seconds: float = 60.0
    per_scanner_timeout: float = 30.0
    seed: int = 42
    categorical_threshold: int = 20
    max_slices_per_column: int = 50
    min_slice_samples: int = 30
    critical_drop: float = 0.20
    warning_drop: float = 0.10
    scanner_config: dict[str, dict[str, Any]] = field(
        default_factory=dict,
    )
    enabled_scanners: list[str] | None = None
    disabled_scanners: list[str] | None = None


@dataclass
class ScanContext:
    """Everything a scanner might need, pre-computed by the engine.

    The engine builds this once from the user's inputs before any
    scanner runs.  Each scanner declares a ``requires`` set of
    field names (e.g., ``{"model_fn", "X", "y"}``).  If any
    required field is ``None``, the scanner is skipped and its
    name goes into ``scanners_skipped``.

    This design means scanners never do detection logic
    themselves -- they just read from the context.

    Attributes:
        model_fn: The prediction function ``f(X) -> y_pred``,
            or ``None`` if no model was provided.
        predict_proba_fn: Probability prediction function
            ``f(X) -> probabilities``, or ``None``.  Auto-
            detected from ``model_fn`` if it has a
            ``predict_proba`` attribute.
        X: Feature DataFrame (possibly sampled down).
        y: Ground-truth labels/values, or ``None``.
        y_train: Training labels (for overfitting checks),
            or ``None``.
        X_train: Training features (for overfitting checks),
            or ``None``.
        sensitive_columns: Column names flagged as sensitive
            for bias testing (e.g., ``["gender", "age"]``).
        numeric_columns: Auto-detected numeric column names.
        categorical_columns: Auto-detected categorical column
            names.
        model_type: ``"classifier"``, ``"regressor"``, or
            ``"unknown"``.  Auto-detected from prediction
            output.
        config: The :class:`ScanConfig` governing this scan.
        seed: Random seed (copied from config for
            convenience).
    """

    model_fn: Callable[..., Any] | None
    predict_proba_fn: Callable[..., Any] | None
    X: pd.DataFrame
    y: np.ndarray | None
    y_train: np.ndarray | None
    X_train: pd.DataFrame | None
    sensitive_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_type: str
    config: ScanConfig
    seed: int

    @property
    def available_fields(self) -> set[str]:
        """Return names of fields that are not ``None``.

        Scanners declare ``requires = {"model_fn", "X"}``.
        The engine checks
        ``scanner.requires <= ctx.available_fields``
        before running it.

        Returns:
            Set of attribute names whose values are not
            ``None``.
        """
        present: set[str] = set()
        for attr in (
            "model_fn",
            "predict_proba_fn",
            "X",
            "y",
            "y_train",
            "X_train",
            "sensitive_columns",
        ):
            value = getattr(self, attr)
            if value is not None:
                present.add(attr)
        # X is always present (never None)
        present.add("X")
        return present
