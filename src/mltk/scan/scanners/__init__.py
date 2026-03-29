"""Built-in scanners for ``mltk scan``.

This package contains the eight default scanners that ship with
mltk.  They run in a fixed order designed to maximize value per
second:

1. **DataScanner** -- data quality (nulls, PII, outliers).
   Runs first because bad data invalidates all downstream
   checks.  Requires only ``X`` (no model needed).

2. **DriftScanner** -- feature distribution anomalies.
   Checks column distributions for unusual patterns.
   Requires only ``X``.

3. **LeakageScanner** -- feature-target correlation.
   Flags features suspiciously correlated with the target.
   Requires ``X`` and ``y`` (no model needed).

4. **SliceScanner** -- subgroup performance drops.
   Tests model accuracy on every categorical slice.
   Requires ``model_fn``, ``X``, ``y``.

5. **BiasScanner** -- fairness violations.
   Compares prediction rates across sensitive groups.
   Requires ``model_fn``, ``X``, ``y``,
   ``sensitive_columns``.

6. **CalibrationScanner** -- confidence miscalibration.
   Checks whether predicted probabilities match reality.
   Requires ``predict_proba_fn``, ``X``, ``y``.

7. **RobustnessScanner** -- prediction instability.
   Adds small noise and checks if predictions flip.
   Requires ``model_fn``, ``X``.

8. **OverfitScanner** -- train-test performance gap.
   Compares model metrics on training vs test data.
   Requires ``model_fn``, ``X``, ``y``, ``X_train``,
   ``y_train``.

Custom scanners can be registered via
:func:`mltk.scan.register_scanner`.
"""

from __future__ import annotations

from mltk.scan.scanners.base import Scanner
from mltk.scan.scanners.bias import BiasScanner
from mltk.scan.scanners.leakage import LeakageScanner
from mltk.scan.scanners.slice import SliceScanner

__all__ = [
    "Scanner",
    "BUILTIN_SCANNERS",
    "LeakageScanner",
    "SliceScanner",
    "BiasScanner",
]

# Ordered list of built-in scanner classes.
# The engine instantiates and runs them in this order.
# Data-only scanners first (fast, no model), then model
# scanners by ascending computational cost.
# MVP: 3 scanners. Planned: DataScanner, DriftScanner,
# CalibrationScanner, RobustnessScanner, OverfitScanner.
BUILTIN_SCANNERS: list[type[Scanner]] = [
    LeakageScanner,
    SliceScanner,
    BiasScanner,
]
