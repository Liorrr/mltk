"""MLflow integration — log mltk test results as metrics and artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _sanitize_metric_name(name: str) -> str:
    """Sanitize a string into a valid MLflow metric name.

    MLflow metric names allow letters, digits, underscores, dashes, and
    forward slashes. Dots are replaced with underscores; all other
    non-alphanumeric characters (except ``-`` and ``/``) are stripped.

    Args:
        name: Raw metric name (e.g., a pytest node ID or test name).

    Returns:
        Sanitized metric name safe for MLflow.
    """
    # Replace dots with underscores (common in node IDs like "tests/test_model.py")
    name = name.replace(".", "_")
    # Strip any character that is not alphanumeric, underscore, dash, or slash
    name = re.sub(r"[^\w\-/]", "_", name)
    # Collapse consecutive underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


class MlflowLogger:
    """Log mltk test results to an MLflow tracking server.

    Example::

        from mltk.integrations.mlflow_logger import MlflowLogger
        from mltk.core.result import TestSuite

        logger = MlflowLogger(experiment_name="my-model-tests")
        suite = run_my_tests()
        logger.log_results(suite)

    MLflow is an optional dependency. An :class:`ImportError` with a clear
    installation hint is raised if ``mlflow`` is not available.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize the logger.

        MLflow is imported lazily here so that importing this module does
        not require ``mlflow`` to be installed.

        Args:
            experiment_name: MLflow experiment to log into. Uses the
                MLflow default experiment when ``None``.
            tracking_uri: MLflow tracking server URI
                (e.g. ``"http://localhost:5000"``). Uses the value of
                ``MLFLOW_TRACKING_URI`` env-var or local ``./mlruns``
                when ``None``.
        """
        self._mlflow = self._import_mlflow()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)

        if experiment_name is not None:
            self._mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_mlflow() -> Any:
        """Lazy-import mlflow, raising a helpful error if not installed."""
        try:
            import mlflow  # noqa: PLC0415
            return mlflow
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MlflowLogger but is not installed. "
                "Install it with: pip install mlflow"
            ) from exc

    def _ensure_active_run(self, run_id: str | None) -> Any:
        """Return context that guarantees an active MLflow run.

        If *run_id* is given, the existing run is resumed.
        If there is already an active run, it is reused.
        Otherwise a new run is started.

        Returns the MLflow run object (used in tests for assertions).
        """
        mlflow = self._mlflow

        if run_id is not None:
            return mlflow.start_run(run_id=run_id)

        active = mlflow.active_run()
        if active is not None:
            return active

        return mlflow.start_run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_results(self, suite: TestSuite, run_id: str | None = None) -> None:  # noqa: F821
        """Log :class:`~mltk.core.result.TestSuite` metrics to MLflow.

        The following metrics are recorded:

        * ``mltk_total_tests`` — total number of test results in the suite
        * ``mltk_passed``      — number of passing tests
        * ``mltk_failed``      — number of failing tests
        * ``mltk_score``       — pass rate as a percentage (0–100)
        * ``mltk_duration_ms`` — combined duration across all tests (ms)
        * ``mltk_{name}``      — per-test result: ``1.0`` (passed) or ``0.0``

        Args:
            suite: :class:`~mltk.core.result.TestSuite` produced by a test run.
            run_id: Optional existing MLflow run ID to log into. When
                ``None``, the active run is reused or a new one is started.
        """
        mlflow = self._mlflow

        with self._ensure_active_run(run_id):
            total_duration = sum(r.duration_ms for r in suite.results)

            mlflow.log_metric("mltk_total_tests", suite.total)
            mlflow.log_metric("mltk_passed", suite.passed_count)
            mlflow.log_metric("mltk_failed", suite.failed_count)
            mlflow.log_metric("mltk_score", suite.score)
            mlflow.log_metric("mltk_duration_ms", total_duration)

            for result in suite.results:
                metric_name = "mltk_" + _sanitize_metric_name(result.name)
                mlflow.log_metric(metric_name, 1.0 if result.passed else 0.0)

    def log_report(self, report_path: str | Path, run_id: str | None = None) -> None:
        """Attach an HTML report file as an MLflow artifact.

        Args:
            report_path: Path to the HTML report file to upload.
            run_id: Optional existing MLflow run ID. Behaves the same as
                in :meth:`log_results`.
        """
        mlflow = self._mlflow
        report_path = Path(report_path)

        with self._ensure_active_run(run_id):
            mlflow.log_artifact(str(report_path))

    def log_test_result(self, result: TestResult, run_id: str | None = None) -> None:  # noqa: F821
        """Log a single :class:`~mltk.core.result.TestResult` as an MLflow metric.

        The metric name is ``mltk_{sanitized_test_name}`` and the value is
        ``1.0`` if the test passed or ``0.0`` if it failed.

        Args:
            result: A single :class:`~mltk.core.result.TestResult`.
            run_id: Optional existing MLflow run ID. Behaves the same as
                in :meth:`log_results`.
        """
        mlflow = self._mlflow
        metric_name = "mltk_" + _sanitize_metric_name(result.name)

        with self._ensure_active_run(run_id):
            mlflow.log_metric(metric_name, 1.0 if result.passed else 0.0)
