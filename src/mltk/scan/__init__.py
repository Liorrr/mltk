"""``mltk scan`` -- auto-discover model issues and generate tests.

This is the public API for mltk's scanning engine.  The simplest
usage is the :func:`scan` function, which takes a model and data
and returns a :class:`ScanReport` with all discovered issues::

    from mltk.scan import scan

    report = scan(
        model.predict, X_test, y_test,
        sensitive_columns=["age", "gender"],
    )

    # Console summary
    print(report.summary())

    # Generate a pytest file with tests for every issue
    report.to_test_file("tests/test_scan_results.py")

    # Run findings as an MltkSuite
    suite_result = report.to_suite().run()

    # Export for CI/CD
    report.to_junit("scan-results.xml")

For more control, use :class:`ScanEngine` directly::

    from mltk.scan import ScanEngine, ScanConfig

    config = ScanConfig(
        max_scan_rows=5_000,
        per_scanner_timeout=10.0,
        enabled_scanners=["slice", "bias"],
    )
    engine = ScanEngine(config)
    report = engine.scan(model.predict, X_test, y_test)

Custom scanners can be registered with
:func:`register_scanner`::

    from mltk.scan import register_scanner
    from mltk.scan.scanners.base import Scanner

    @register_scanner
    class MyScanner(Scanner):
        name = "my_check"
        category = "custom"
        requires = {"model_fn", "X"}

        def scan(self, ctx):
            return []

Registered scanners run after all built-in scanners, in
registration order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from mltk.scan.config import ScanConfig, ScanContext
from mltk.scan.engine import ScanEngine, ScanReport
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = [
    "scan",
    "register_scanner",
    "ScanEngine",
    "ScanReport",
    "ScanFinding",
    "ScanConfig",
    "ScanContext",
    "Scanner",
]

# Module-level registry for custom scanners.
# Use register_scanner() to add your own.
_SCANNER_REGISTRY: list[type[Scanner]] = []


def register_scanner(
    cls: type[Scanner],
) -> type[Scanner]:
    """Register a custom scanner class.

    Registered scanners are included alongside built-in
    scanners whenever :func:`scan` or
    :class:`ScanEngine` is used.

    Can be used as a decorator::

        @register_scanner
        class MyScanner(Scanner):
            name = "my_check"
            category = "custom"
            requires = {"model_fn", "X"}

            def scan(self, ctx):
                return []

    Or called directly::

        register_scanner(MyScanner)

    Args:
        cls: A subclass of :class:`Scanner` with ``name``,
            ``category``, ``requires``, and ``scan()``
            defined.

    Returns:
        The same class (unmodified), so it works as a
        decorator.
    """
    _SCANNER_REGISTRY.append(cls)
    return cls


def scan(
    model_fn: Any,
    X: pd.DataFrame,
    y: Any | None = None,
    sensitive_columns: Sequence[str] | None = None,
    config: ScanConfig | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: Any | None = None,
) -> ScanReport:
    """One-line scan API -- finds issues and generates tests.

    Creates a :class:`ScanEngine` with the given config (or
    defaults), includes all built-in and registered custom
    scanners, runs the scan, and returns a
    :class:`ScanReport`.

    Args:
        model_fn: Prediction function ``f(X) -> y_pred``.
            Pass ``None`` to run data-only scanners
            (DataScanner, DriftScanner, LeakageScanner).
        X: Feature DataFrame.
        y: Ground-truth labels/values.  Required for most
            scanners.
        sensitive_columns: Column names for bias testing
            (e.g., ``["gender", "age"]``).  If ``None``,
            auto-detected from column names.
        config: Scan configuration.  If ``None``, uses
            :class:`ScanConfig` defaults.
        X_train: Training features (for overfitting checks).
        y_train: Training labels (for overfitting checks).

    Returns:
        :class:`ScanReport` with all findings and export
        methods.

    Example::

        from mltk.scan import scan

        report = scan(
            model.predict, X_test, y_test,
            sensitive_columns=["gender"],
        )
        print(report.summary())
        report.to_test_file("tests/test_model.py")
    """
    engine = ScanEngine(
        config=config,
        extra_scanners=list(_SCANNER_REGISTRY),
    )
    return engine.scan(
        model_fn=model_fn,
        X=X,
        y=y,
        sensitive_columns=sensitive_columns,
        X_train=X_train,
        y_train=y_train,
    )
