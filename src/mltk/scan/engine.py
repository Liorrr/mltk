"""Scan engine -- orchestrates scanners and produces reports.

The engine is the core of ``mltk scan``.  It:

1. **Builds context** -- auto-detects model type, column types,
   sensitive columns, and samples data down to a manageable
   size.
2. **Filters scanners** -- checks each scanner's ``requires``
   set against the context.  Scanners whose requirements are
   not met are skipped gracefully.
3. **Runs scanners** -- executes each scanner with per-scanner
   timeout and error isolation.  A crashing scanner never
   takes down the scan.
4. **Collects findings** -- aggregates all
   :class:`~mltk.scan.finding.ScanFinding` objects into a
   :class:`ScanReport`.

The report can then be exported to pytest code, HTML, JUnit
XML, or run directly as an :class:`~mltk.core.suite.MltkSuite`.

Usage::

    from mltk.scan.engine import ScanEngine
    from mltk.scan.detect import _SENSITIVE_KEYWORDS
    from mltk.scan.config import ScanConfig

    engine = ScanEngine(ScanConfig(seed=42))
    report = engine.scan(model.predict, X_test, y_test)
    print(report.summary())
    report.to_test_file("tests/test_scan.py")
"""

from __future__ import annotations

import ast
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mltk.core.result import Severity
from mltk.core.suite import MltkSuite
from mltk.scan.config import ScanConfig, ScanContext
from mltk.scan.detect import (
    _SENSITIVE_KEYWORDS,
)
from mltk.scan.finding import ScanFinding
from mltk.scan.scanners import BUILTIN_SCANNERS
from mltk.scan.scanners.base import Scanner

__all__ = ["ScanEngine", "ScanReport"]

logger = logging.getLogger(__name__)

# Sensitive column name keywords for auto-detection.


# ---------------------------------------------------------------
# ScanReport
# ---------------------------------------------------------------


@dataclass
class ScanReport:
    """Aggregated output of a scan run.

    Contains every finding from every scanner that ran,
    plus metadata about which scanners ran, which were
    skipped, which errored, and how long the scan took.

    The report is the gateway to all output formats:

    - ``to_suite()`` -- runnable :class:`MltkSuite`
    - ``to_test_file()`` -- self-contained pytest file
    - ``to_html()`` -- interactive HTML report
    - ``to_junit()`` -- CI/CD-compatible XML
    - ``summary()`` -- console-friendly text

    Attributes:
        findings: All issues found across all scanners.
        scanners_run: Names of scanners that executed
            successfully.
        scanners_skipped: Names of scanners skipped due
            to missing requirements.
        scanners_errored: Mapping of scanner name to
            error message for scanners that crashed.
        duration_ms: Total wall-clock time in milliseconds.
        model_type: Detected model type (``"classifier"``,
            ``"regressor"``, or ``"unknown"``).
        n_samples: Number of rows scanned (after sampling).
        n_features: Number of feature columns.
        config: The :class:`ScanConfig` used for this run.
    """

    findings: list[ScanFinding] = field(
        default_factory=list,
    )
    scanners_run: list[str] = field(default_factory=list)
    scanners_skipped: list[str] = field(
        default_factory=list,
    )
    scanners_errored: dict[str, str] = field(
        default_factory=dict,
    )
    duration_ms: float = 0.0
    model_type: str = "unknown"
    n_samples: int = 0
    n_features: int = 0
    config: ScanConfig = field(default_factory=ScanConfig)

    # -- output methods ------------------------------------

    def to_suite(self) -> MltkSuite:
        """Build a runnable MltkSuite from findings.

        Each finding's ``assertion_fn`` + ``assertion_args``
        + ``assertion_kwargs`` are registered as pending
        assertions.  Calling ``suite.run()`` replays every
        finding.

        Returns:
            An :class:`MltkSuite` ready to ``.run()``.
        """
        suite = MltkSuite("scan")
        for f in self.findings:
            suite.add(
                f.assertion_fn,
                *f.assertion_args,
                **f.assertion_kwargs,
            )
        return suite

    def to_test_file(self, path: str) -> str:
        """Write a self-contained pytest file from findings.

        Each finding's ``suggested_test`` is collected into
        a single Python file with proper imports and
        fixtures.  The generated code is validated with
        ``ast.parse()`` before writing.

        Args:
            path: Destination file path (e.g.,
                ``"tests/test_scan_results.py"``).

        Returns:
            Absolute path to the written file.

        Raises:
            SyntaxError: If the generated code is invalid
                Python (should not happen with well-formed
                scanners).
        """
        lines: list[str] = [
            '"""Auto-generated by mltk scan.',
            "",
            "These tests reproduce issues found during a",
            "model scan.  Each test is self-contained and",
            "can be run with pytest.",
            '"""',
            "",
            "import os",
            "",
            "import numpy as np",
            "import pandas as pd",
            "import pytest",
            "",
            "import mltk",
            "",
            "",
        ]

        for _i, f in enumerate(self.findings):
            if not f.suggested_test:
                continue
            lines.append(f.suggested_test)
            lines.append("")
            lines.append("")

        code = "\n".join(lines)
        ast.parse(code)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code, encoding="utf-8")
        return str(out.resolve())

    def to_html(self, path: str) -> str:
        """Generate an HTML report from findings.

        Delegates to :func:`mltk.report.generate_report`.

        Args:
            path: Destination HTML file path.

        Returns:
            Absolute path to the generated HTML file.
        """
        from mltk.report.generator import generate_report

        out = Path(path)
        report_results: list[dict[str, Any]] = []
        for f in self.findings:
            report_results.append({
                "nodeid": f.result.name,
                "outcome": (
                    "passed" if f.result.passed
                    else "failed"
                ),
                "duration": f.result.duration_ms / 1000.0,
                "ml_result": f.result,
            })

        generated = generate_report(
            results=report_results,
            output_dir=str(out.parent),
            title="mltk scan",
        )
        return str(generated.resolve())

    def to_junit(self, path: str) -> str:
        """Export findings as JUnit XML for CI/CD systems.

        Delegates to :func:`mltk.report.junit.export_junit_xml`.

        Args:
            path: Destination XML file path.

        Returns:
            Absolute path to the written XML file.
        """
        from mltk.report.junit import export_junit_xml

        records: list[dict[str, Any]] = []
        for f in self.findings:
            r = f.result
            records.append({
                "name": r.name,
                "passed": r.passed,
                "severity": r.severity.value,
                "message": r.message,
                "details": r.details,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp.isoformat(),
            })

        return export_junit_xml(
            results=records,
            output_path=path,
            suite_name="mltk-scan",
        )

    def to_json(
        self, path: str | None = None,
    ) -> str:
        """Serialize report as JSON.

        Produces a machine-readable representation of the
        scan report including all findings, scanner stats,
        and metadata.

        Args:
            path: Optional file path.  When provided the
                JSON is written to disk in addition to being
                returned.

        Returns:
            JSON string of the full report.
        """

        class _Enc(json.JSONEncoder):
            """Handle numpy / enum types."""

            def default(self, o: Any) -> Any:
                if isinstance(o, Severity):
                    return o.value
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        findings_list: list[dict[str, Any]] = []
        for f in self.findings:
            r = f.result
            finding: dict[str, Any] = {
                "name": r.name,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
                "scanner_name": f.scanner_name,
                "duration_ms": r.duration_ms,
                "details": r.details,
                "suggested_test": f.suggested_test,
            }
            findings_list.append(finding)

        payload: dict[str, Any] = {
            "findings": findings_list,
            "scanners_run": self.scanners_run,
            "scanners_skipped": self.scanners_skipped,
            "scanners_errored": self.scanners_errored,
            "model_type": self.model_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
        }

        text = json.dumps(
            payload, cls=_Enc, indent=2,
        )

        if path is not None:
            out = Path(path)
            out.parent.mkdir(
                parents=True, exist_ok=True,
            )
            out.write_text(text, encoding="utf-8")

        return text

    def summary(self) -> str:
        """Return a console-friendly text summary.

        Includes scanner stats, each finding with severity,
        and a footer with counts and a pytest hint.

        Returns:
            Multi-line string suitable for ``print()``.
        """
        lines: list[str] = []

        # Header
        total_scanners = (
            len(self.scanners_run)
            + len(self.scanners_skipped)
            + len(self.scanners_errored)
        )
        skipped_note = ""
        if self.scanners_skipped:
            skipped_note = (
                f" ({', '.join(self.scanners_skipped)}"
                f" skipped)"
            )

        lines.append(
            f"mltk scan | {self.model_type} | "
            f"{self.n_samples:,} samples | "
            f"{self.n_features} features"
        )
        lines.append(
            f"Scanners: {len(self.scanners_run)}"
            f"/{total_scanners} run"
            f"{skipped_note} | "
            f"{self.duration_ms:.1f}ms"
        )
        lines.append("")

        # Findings
        critical = 0
        warnings = 0
        info = 0

        for f in self.findings:
            sev = f.result.severity
            if sev == Severity.CRITICAL:
                marker = "CRITICAL"
                critical += 1
            elif sev == Severity.WARNING:
                marker = "WARNING "
                warnings += 1
            else:
                marker = "INFO    "
                info += 1

            passed_mark = "  " if f.result.passed else "X "
            lines.append(
                f"  {passed_mark}{marker}  "
                f"{f.result.message}  "
                f"[{f.scanner_name}]"
            )

        # Errored scanners
        for name, err in self.scanners_errored.items():
            lines.append(
                f"  !  ERROR    {name}: {err}"
            )

        lines.append("")
        lines.append(
            f"Summary: {critical} critical, "
            f"{warnings} warnings, {info} info"
        )

        if self.findings:
            lines.append(
                "-> Run: pytest tests/"
                "test_scan_results.py"
            )

        return "\n".join(lines)

    @property
    def exit_code(self) -> int:
        """Process exit code for CLI usage.

        Returns:
            - ``0`` -- no findings (clean scan).
            - ``1`` -- findings were discovered.
            - ``2`` -- at least one scanner errored.
        """
        if self.scanners_errored:
            return 2
        if self.findings:
            return 1
        return 0


# ---------------------------------------------------------------
# ScanEngine
# ---------------------------------------------------------------


class ScanEngine:
    """Orchestrates scanners and produces a ScanReport.

    The engine is stateless between scans -- you can call
    ``scan()`` multiple times with different models/data.

    Args:
        config: Scan configuration.  If ``None``, uses
            :class:`ScanConfig` defaults.
        extra_scanners: Additional scanner classes to run
            alongside the built-in ones.

    Example::

        engine = ScanEngine()
        report = engine.scan(model.predict, X_test, y_test)
        print(report.summary())
    """

    def __init__(
        self,
        config: ScanConfig | None = None,
        extra_scanners: Sequence[type[Scanner]] | None = None,
    ) -> None:
        self._config = config or ScanConfig()
        self._scanner_classes: list[type[Scanner]] = list(
            BUILTIN_SCANNERS
        )
        if extra_scanners:
            self._scanner_classes.extend(extra_scanners)

    def scan(
        self,
        model_fn: Callable[..., Any] | None,
        X: pd.DataFrame,
        y: Any | None = None,
        sensitive_columns: Sequence[str] | None = None,
        X_train: pd.DataFrame | None = None,
        y_train: Any | None = None,
    ) -> ScanReport:
        """Run all applicable scanners and return a report.

        Args:
            model_fn: Prediction function ``f(X) -> y_pred``.
                Pass ``None`` to run data-only scanners.
            X: Feature DataFrame.
            y: Ground-truth labels/values (optional).
            sensitive_columns: Column names for bias testing.
                If ``None``, auto-detected from column names.
            X_train: Training features for overfitting checks.
            y_train: Training labels for overfitting checks.

        Returns:
            :class:`ScanReport` with all findings, stats, and
            export methods.
        """
        start = time.perf_counter()

        ctx = self._build_context(
            model_fn=model_fn,
            X=X,
            y=y,
            sensitive_columns=sensitive_columns,
            X_train=X_train,
            y_train=y_train,
        )

        scanners = self._resolve_scanners()
        available = ctx.available_fields

        all_findings: list[ScanFinding] = []
        scanners_run: list[str] = []
        scanners_skipped: list[str] = []
        scanners_errored: dict[str, str] = {}

        budget_start = time.perf_counter()

        for scanner in scanners:
            # Check time budget
            elapsed = time.perf_counter() - budget_start
            if elapsed >= self._config.time_budget_seconds:
                logger.info(
                    "Time budget exhausted (%.1fs); "
                    "stopping scan.",
                    elapsed,
                )
                remaining = [
                    s.name for s in scanners
                    if s.name not in scanners_run
                    and s.name not in scanners_skipped
                    and s.name not in scanners_errored
                ]
                scanners_skipped.extend(remaining)
                break

            # Check requirements
            missing = scanner.requires - available
            if missing:
                logger.debug(
                    "Skipping %s: missing %s",
                    scanner.name,
                    missing,
                )
                scanners_skipped.append(scanner.name)
                continue

            # Check sensitive_columns requirement
            if (
                "sensitive_columns" in scanner.requires
                and not ctx.sensitive_columns
            ):
                logger.debug(
                    "Skipping %s: no sensitive columns",
                    scanner.name,
                )
                scanners_skipped.append(scanner.name)
                continue

            # Run with timeout and error isolation
            findings = self._run_scanner(
                scanner, ctx, scanners_errored,
            )
            if findings is not None:
                all_findings.extend(findings)
                scanners_run.append(scanner.name)

        duration_ms = (
            (time.perf_counter() - start) * 1000
        )

        return ScanReport(
            findings=all_findings,
            scanners_run=scanners_run,
            scanners_skipped=scanners_skipped,
            scanners_errored=scanners_errored,
            duration_ms=duration_ms,
            model_type=ctx.model_type,
            n_samples=len(ctx.X),
            n_features=len(ctx.X.columns),
            config=self._config,
        )

    # -----------------------------------------------------------
    # Internal: context building
    # -----------------------------------------------------------

    def _build_context(
        self,
        model_fn: Callable[..., Any] | None,
        X: pd.DataFrame,
        y: Any | None,
        sensitive_columns: Sequence[str] | None,
        X_train: pd.DataFrame | None,
        y_train: Any | None,
    ) -> ScanContext:
        """Build a ScanContext from raw user inputs.

        Auto-detects:
        - Model type (classifier vs regressor)
        - Column types (numeric vs categorical)
        - Sensitive columns (if not provided)
        - predict_proba function (if available)
        - Samples data if too large
        """
        cfg = self._config
        rng = np.random.RandomState(cfg.seed)

        # Convert y to numpy
        y_arr: np.ndarray | None = None
        if y is not None:
            y_arr = np.asarray(y)

        y_train_arr: np.ndarray | None = None
        if y_train is not None:
            y_train_arr = np.asarray(y_train)

        # Sample if needed
        X_sampled, y_sampled = self._maybe_sample(
            X, y_arr, rng,
        )

        # Detect column types
        numeric_cols = self._detect_numeric(
            X_sampled,
        )
        categorical_cols = self._detect_categorical(
            X_sampled,
        )

        # Detect sensitive columns
        if sensitive_columns is not None:
            sens = list(sensitive_columns)
        else:
            sens = self._detect_sensitive(X_sampled)

        # Detect model type
        model_type = self._detect_model_type(
            model_fn, X_sampled, y_sampled,
        )

        # Detect predict_proba
        proba_fn = self._detect_proba(model_fn)

        return ScanContext(
            model_fn=model_fn,
            predict_proba_fn=proba_fn,
            X=X_sampled,
            y=y_sampled,
            y_train=y_train_arr,
            X_train=X_train,
            sensitive_columns=sens,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            model_type=model_type,
            config=cfg,
            seed=cfg.seed,
        )

    def _maybe_sample(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None,
        rng: np.random.RandomState,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Sample data down to max_scan_rows if needed.

        Uses stratified sampling for classifiers
        (preserving class proportions) and quantile-binned
        stratification for regressors.
        """
        cfg = self._config
        n = len(X)
        if n <= cfg.max_scan_rows:
            return X, y

        if (
            y is not None
            and cfg.sample_strategy == "stratified"
        ):
            indices = self._stratified_sample(
                y, cfg.max_scan_rows, rng,
            )
        else:
            indices = rng.choice(
                n, size=cfg.max_scan_rows, replace=False,
            )

        indices = np.sort(indices)
        X_sampled = X.iloc[indices].reset_index(drop=True)
        y_sampled = y[indices] if y is not None else None
        return X_sampled, y_sampled

    @staticmethod
    def _stratified_sample(
        y: np.ndarray,
        n_samples: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Stratified sampling preserving label distribution.

        For continuous targets (regressors), bins y into
        quantiles first to approximate stratification.
        """
        unique_vals = np.unique(y)
        # If many unique values, treat as regression target
        # and bin into quantile groups.
        if len(unique_vals) > 20:
            try:
                bins = np.quantile(
                    y,
                    np.linspace(0, 1, 11),
                )
                bins = np.unique(bins)
                labels = np.digitize(y, bins[1:-1])
            except Exception:
                return rng.choice(
                    len(y),
                    size=n_samples,
                    replace=False,
                )
        else:
            labels = y

        # Per-class proportional sampling
        unique_labels = np.unique(labels)
        indices: list[int] = []
        for label in unique_labels:
            label_idx = np.where(labels == label)[0]
            n_label = max(
                1,
                int(
                    len(label_idx)
                    / len(y)
                    * n_samples
                ),
            )
            n_label = min(n_label, len(label_idx))
            chosen = rng.choice(
                label_idx, size=n_label, replace=False,
            )
            indices.extend(chosen.tolist())

        return np.array(indices[:n_samples])

    def _detect_numeric(
        self, X: pd.DataFrame,
    ) -> list[str]:
        """Identify numeric columns in the DataFrame."""
        return [
            col for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]

    def _detect_categorical(
        self, X: pd.DataFrame,
    ) -> list[str]:
        """Identify categorical columns in the DataFrame.

        A column is categorical if:
        - It is not numeric, OR
        - It is numeric but has <= categorical_threshold
          unique values.
        """
        threshold = self._config.categorical_threshold
        result: list[str] = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                result.append(col)
            elif X[col].nunique() <= threshold:
                result.append(col)
        return result

    @staticmethod
    def _detect_sensitive(
        X: pd.DataFrame,
    ) -> list[str]:
        """Auto-detect sensitive columns by name keywords."""
        found: list[str] = []
        for col in X.columns:
            lower = str(col).lower().replace("-", "_")
            for keyword in _SENSITIVE_KEYWORDS:
                if keyword in lower:
                    found.append(str(col))
                    break
        return found

    @staticmethod
    def _detect_model_type(
        model_fn: Callable[..., Any] | None,
        X: pd.DataFrame,
        y: np.ndarray | None,
    ) -> str:
        """Detect whether the model is a classifier or regressor.

        Heuristic: if y has few unique values relative to
        its size, it is classification.  Otherwise regression.
        Falls back to probing model output if y is absent.
        """
        if model_fn is None:
            return "unknown"

        if y is not None:
            unique = np.unique(y)
            if len(unique) <= 20:
                return "classifier"
            # Float labels with many unique -> regression
            if np.issubdtype(y.dtype, np.floating):
                return "regressor"
            return "classifier"

        # Probe model output on a small sample
        try:
            sample = X.head(min(10, len(X)))
            preds = np.asarray(model_fn(sample))
            if preds.ndim == 1:
                unique_preds = np.unique(preds)
                if len(unique_preds) <= 20:
                    return "classifier"
                return "regressor"
        except Exception:
            pass

        return "unknown"

    @staticmethod
    def _detect_proba(
        model_fn: Callable[..., Any] | None,
    ) -> Callable[..., Any] | None:
        """Extract predict_proba if model_fn has one."""
        if model_fn is None:
            return None

        # Check if model_fn is a bound method and the
        # underlying object has predict_proba
        obj = getattr(model_fn, "__self__", None)
        if obj is not None:
            proba = getattr(obj, "predict_proba", None)
            if callable(proba):
                return proba

        return None

    # -----------------------------------------------------------
    # Internal: scanner resolution and execution
    # -----------------------------------------------------------

    def _resolve_scanners(self) -> list[Scanner]:
        """Instantiate and filter scanner classes.

        Applies ``enabled_scanners`` and ``disabled_scanners``
        filters from the config.
        """
        cfg = self._config
        instances: list[Scanner] = []

        for cls in self._scanner_classes:
            scanner = cls()
            name = scanner.name

            if cfg.enabled_scanners is not None:
                if name not in cfg.enabled_scanners:
                    continue

            if cfg.disabled_scanners is not None:
                if name in cfg.disabled_scanners:
                    continue

            instances.append(scanner)

        return instances

    def _run_scanner(
        self,
        scanner: Scanner,
        ctx: ScanContext,
        errors: dict[str, str],
    ) -> list[ScanFinding] | None:
        """Run a single scanner with timeout and error isolation.

        Returns:
            List of findings on success, or None if the
            scanner errored (error is recorded in *errors*).
        """
        timeout = self._config.per_scanner_timeout
        result_holder: list[list[ScanFinding] | None] = [
            None,
        ]
        error_holder: list[BaseException | None] = [None]

        def _target() -> None:
            try:
                result_holder[0] = scanner.scan(ctx)
            except Exception as exc:
                error_holder[0] = exc

        thread = threading.Thread(
            target=_target, daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning(
                "Scanner %s timed out after %.1fs",
                scanner.name,
                timeout,
            )
            errors[scanner.name] = (
                f"Timed out after {timeout:.1f}s"
            )
            return None

        if error_holder[0] is not None:
            exc = error_holder[0]
            logger.warning(
                "Scanner %s errored: %s",
                scanner.name,
                exc,
                exc_info=exc,
            )
            errors[scanner.name] = str(exc)
            return None

        return result_holder[0]
