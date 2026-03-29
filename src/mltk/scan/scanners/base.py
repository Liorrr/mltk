"""Scanner abstract base class.

Every built-in and custom scanner inherits from :class:`Scanner`.
The contract is minimal:

1. Declare ``name``, ``category``, and ``requires``.
2. Implement ``scan(ctx) -> list[ScanFinding]``.

The engine handles everything else -- timeout enforcement,
error isolation, requirement checking, and ordering.

**name** is a short lowercase identifier (e.g., ``"slice"``,
``"bias"``).  It is used in console output, per-scanner config
overrides, and the ``enabled_scanners`` / ``disabled_scanners``
filter lists.

**category** groups scanners for reporting (e.g.,
``"performance"``, ``"fairness"``, ``"data_quality"``).

**requires** is a set of :class:`ScanContext` field names that
must be non-None for this scanner to run.  If any required
field is missing, the engine skips the scanner gracefully and
records it in ``scanners_skipped``.

Example custom scanner::

    from mltk.scan.scanners.base import Scanner
    from mltk.scan.config import ScanContext
    from mltk.scan.finding import ScanFinding

    class LatencyScanner(Scanner):
        name = "latency"
        category = "performance"
        requires = {"model_fn", "X"}

        def scan(self, ctx: ScanContext) -> list[ScanFinding]:
            # Measure prediction latency
            import time
            start = time.perf_counter()
            ctx.model_fn(ctx.X.iloc[:100])
            elapsed = time.perf_counter() - start
            # ... build ScanFinding if too slow ...
            return findings
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mltk.scan.config import ScanContext
from mltk.scan.finding import ScanFinding

__all__ = ["Scanner"]


class Scanner(ABC):
    """Abstract base class for all mltk scanners.

    Subclasses must set three class attributes and implement
    one method:

    Attributes:
        name: Short lowercase identifier (e.g., ``"slice"``).
            Must be unique across all registered scanners.
            Used for config lookups, filtering, and display.
        category: Grouping label for reports (e.g.,
            ``"performance"``, ``"fairness"``,
            ``"data_quality"``).
        requires: Set of :class:`ScanContext` field names
            that must be non-None for this scanner to run.
            Common values: ``"model_fn"``, ``"X"``, ``"y"``,
            ``"predict_proba_fn"``, ``"sensitive_columns"``,
            ``"X_train"``, ``"y_train"``.

    The engine calls ``scan(ctx)`` only if all requirements
    are satisfied.  If a scanner raises an exception, the
    engine catches it, logs the error, and continues with
    the next scanner.  Scanners should NOT catch broad
    exceptions internally -- let the engine handle isolation.

    Example::

        class MyScanner(Scanner):
            name = "my_check"
            category = "custom"
            requires = {"model_fn", "X", "y"}

            def scan(self, ctx):
                findings = []
                # ... run checks, append ScanFindings ...
                return findings
    """

    name: str = ""
    category: str = ""
    requires: set[str] = frozenset({"model_fn", "X", "y"})

    @abstractmethod
    def scan(
        self, ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run this scanner and return discovered issues.

        Each issue is wrapped in a :class:`ScanFinding` that
        carries both the evidence (TestResult) and the
        reproduction recipe (assertion_fn + args).

        Args:
            ctx: Pre-built scan context with model, data,
                detected column types, and configuration.

        Returns:
            List of :class:`ScanFinding` objects.  Return
            an empty list if no issues are found.
        """
        ...  # pragma: no cover
