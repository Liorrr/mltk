"""Weights & Biases integration -- log mltk test results to W&B dashboards.

WHY W&B for ML testing:
W&B (Weights & Biases) is the most popular experiment tracking tool, used by
70%+ of ML teams for logging training metrics, hyperparameters, and artifacts.
Most teams already have W&B dashboards for training runs.

The problem: test results from mltk live in pytest output (ephemeral), while
training metrics live in W&B (persistent). When a model passes training but
fails quality gates, there is no single place to see both signals.

This adapter bridges the gap by logging mltk TestResult objects as W&B metrics,
tables, and summaries. After integration, your W&B dashboard shows:
"Did the model train well AND pass quality gates?" -- in one view.

Usage::

    from mltk.integrations.wandb_adapter import WandbLogger

    logger = WandbLogger(project="my-model-tests", tags=["nightly"])
    logger.log_result({"name": "accuracy", "passed": True, ...})
    logger.log_suite(results)
    url = logger.finish()  # returns the W&B run URL

wandb is an optional dependency. A clear ImportError with installation
instructions is raised if ``wandb`` is not available.
"""

from __future__ import annotations

from typing import Any


class WandbLogger:
    """Log mltk test results to Weights & Biases.

    WHY W&B for ML testing:
    W&B is the most popular experiment tracking tool (used by 70%+ of ML teams).
    Most teams already have W&B dashboards for training metrics. By logging
    mltk test results alongside training runs, you get a unified view:
    "Did the model train well AND pass quality gates?"

    Without this integration, test results live in pytest output (ephemeral)
    while training metrics live in W&B (persistent). This bridges the gap.

    Example::

        logger = WandbLogger(project="model-qa", tags=["nightly", "v2.1"])
        logger.log_result({"name": "drift_psi", "passed": True, ...})
        logger.log_suite(all_results)
        run_url = logger.finish()
    """

    def __init__(
        self,
        project: str = "mltk-tests",
        entity: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize W&B logger. Requires wandb to be installed.

        Creates a new W&B run with the given project, entity, and tags.
        The run stays open until :meth:`finish` is called, allowing multiple
        ``log_result`` and ``log_suite`` calls to accumulate in the same run.

        Args:
            project: W&B project name. Defaults to ``"mltk-tests"``.
            entity: W&B team or user entity. Uses the default entity when None.
            run_name: Human-readable name for the run (shown in the dashboard).
                When None, W&B auto-generates a random name.
            tags: List of string tags for filtering runs in the W&B UI
                (e.g., ``["nightly", "v2.1", "regression"]``).

        Raises:
            ImportError: If wandb is not installed, with a pip install hint.
        """
        self._wandb = self._import_wandb()
        self._run = self._wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags or [],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_wandb() -> Any:
        """Lazy-import wandb, raising a helpful error if not installed.

        WHY lazy import:
        mltk is an ML testing toolkit with many optional integrations. Users
        who do not use W&B should not need it installed. The import happens
        at __init__ time (not module load time) so that ``import mltk`` never
        fails due to a missing optional dependency.

        Raises:
            ImportError: Clear message with ``pip install wandb`` hint.
        """
        try:
            import wandb  # noqa: PLC0415

            return wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandbLogger but is not installed. "
                "Install it with: pip install wandb"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_result(self, result: dict[str, Any]) -> None:
        """Log a single test result as W&B metrics.

        Each result becomes a set of flat metrics in the current W&B step:

        - ``mltk/{name}/passed`` -- 1 if passed, 0 if failed
        - ``mltk/{name}/duration_ms`` -- how long the assertion took
        - ``mltk/{name}/severity`` -- numeric severity (critical=2, warning=1, info=0)

        Plus any numeric values found in the result's ``details`` dict.

        WHY flat metrics (not nested dicts):
        W&B metrics are time-series scalars. Flat keys with slash separators
        create auto-grouped panels in the dashboard (e.g., all ``mltk/drift_*``
        metrics cluster together). This makes it trivial to build monitoring
        dashboards without manual configuration.

        Args:
            result: Dict with at least ``name`` and ``passed`` keys.
                Optional: ``duration_ms``, ``severity``, ``message``, ``details``.
        """
        wandb = self._wandb
        name = result.get("name", "unknown")
        prefix = f"mltk/{name}"

        severity_map = {"critical": 2, "warning": 1, "info": 0}

        metrics: dict[str, Any] = {
            f"{prefix}/passed": 1 if result.get("passed", False) else 0,
            f"{prefix}/duration_ms": result.get("duration_ms", 0.0),
            f"{prefix}/severity": severity_map.get(
                result.get("severity", "info"), 0
            ),
        }

        # Flatten numeric details into metrics
        details = result.get("details", {})
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}/{key}"] = value

        wandb.log(metrics)

    def log_suite(self, results: list[dict[str, Any]]) -> None:
        """Log an entire test suite as a W&B Table + summary metrics.

        This method does two things:

        1. **Table**: Creates a ``wandb.Table`` with columns
           (name, passed, severity, duration_ms, message) and logs it as
           ``"mltk/test_results"``. Tables are browsable in the W&B UI with
           sorting, filtering, and grouping -- ideal for suite-level review.

        2. **Summary metrics**: Logs aggregate numbers (total, passed_count,
           failed_count, pass_rate, total_duration_ms) as ``mltk/summary/*``
           metrics. These power dashboard widgets like pass-rate trend lines
           and duration histograms.

        WHY both Table and metrics:
        Tables give the full picture (every test with its message), while
        summary metrics enable time-series monitoring across runs. A dropping
        pass_rate trend triggers alerts; the table shows which tests broke.

        Args:
            results: List of result dicts, each with at least ``name`` and
                ``passed``. Optional: ``severity``, ``duration_ms``, ``message``.
        """
        wandb = self._wandb

        # Build W&B Table
        columns = ["name", "passed", "severity", "duration_ms", "message"]
        table = wandb.Table(columns=columns)

        for r in results:
            table.add_data(
                r.get("name", "unknown"),
                r.get("passed", False),
                r.get("severity", "info"),
                r.get("duration_ms", 0.0),
                r.get("message", ""),
            )

        wandb.log({"mltk/test_results": table})

        # Compute and log summary metrics
        total = len(results)
        passed_count = sum(1 for r in results if r.get("passed", False))
        failed_count = total - passed_count
        pass_rate = (passed_count / total * 100) if total > 0 else 0.0
        total_duration = sum(r.get("duration_ms", 0.0) for r in results)

        wandb.log(
            {
                "mltk/summary/total": total,
                "mltk/summary/passed_count": passed_count,
                "mltk/summary/failed_count": failed_count,
                "mltk/summary/pass_rate": pass_rate,
                "mltk/summary/total_duration_ms": total_duration,
            }
        )

    def finish(self) -> str:
        """Finish the W&B run and return the run URL.

        Must be called when all logging is complete. After this, no more
        metrics can be logged to this run. The returned URL points to the
        run page in the W&B web UI.

        WHY explicit finish:
        W&B runs hold network connections and background threads for async
        metric upload. Not finishing leaves orphaned runs in the "running"
        state on the dashboard, which confuses team members and wastes quota.

        Returns:
            The W&B run URL (e.g., ``"https://wandb.ai/team/project/runs/abc123"``).
        """
        url = self._run.get_url() if self._run else ""
        self._wandb.finish()
        return url
