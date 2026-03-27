"""Tests for mltk.integrations.wandb_adapter -- Weights & Biases integration.

All tests mock the wandb module so that an actual wandb installation is NOT
required. This mirrors the pattern used in test_mlflow.py: we inject a mock
into sys.modules before importing WandbLogger, so the class binds to our
controlled mock instead of the real library.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str = "accuracy",
    passed: bool = True,
    severity: str = "info",
    message: str = "ok",
    duration_ms: float = 10.0,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal result dict matching the format expected by WandbLogger."""
    result: dict[str, Any] = {
        "name": name,
        "passed": passed,
        "severity": severity,
        "message": message,
        "duration_ms": duration_ms,
    }
    if details is not None:
        result["details"] = details
    return result


def _make_mixed_results() -> list[dict[str, Any]]:
    """Build a list of results with a mix of passed and failed."""
    return [
        _make_result("accuracy", passed=True, severity="critical", duration_ms=5.0),
        _make_result("drift_psi", passed=True, severity="warning", duration_ms=15.0),
        _make_result(
            "latency_p99",
            passed=False,
            severity="critical",
            message="450ms > 200ms threshold",
            duration_ms=8.0,
        ),
    ]


def _make_wandb_mock() -> MagicMock:
    """Return a MagicMock that behaves like the wandb module.

    Key behaviours:
    - ``wandb.init()`` returns a mock run with ``get_url()``
    - ``wandb.Table`` is a callable mock that returns a table with ``add_data``
    - ``wandb.log`` and ``wandb.finish`` are plain mocks
    """
    mock = MagicMock(name="wandb")

    # wandb.init returns a run object with get_url
    run_mock = MagicMock(name="wandb_run")
    run_mock.get_url.return_value = "https://wandb.ai/team/project/runs/abc123"
    mock.init.return_value = run_mock

    # wandb.Table must be callable and return a table mock
    table_mock = MagicMock(name="wandb_table")
    mock.Table.return_value = table_mock

    return mock


# ---------------------------------------------------------------------------
# Fixture: inject mock wandb before each test
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_wandb() -> Generator[MagicMock, None, None]:
    """Patch wandb in sys.modules so WandbLogger uses the mock.

    This fixture temporarily replaces the ``wandb`` entry in ``sys.modules``
    with our controlled mock. After the test, the original state is restored
    (whether wandb was installed or not).
    """
    wandb_mock = _make_wandb_mock()
    original = sys.modules.get("wandb", None)
    sys.modules["wandb"] = wandb_mock  # type: ignore[assignment]
    try:
        yield wandb_mock
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

class TestWandbLoggerInit:
    """WandbLogger.__init__ -- run creation and configuration."""

    def test_init_calls_wandb_init(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: WandbLogger is created with project, entity, run_name, tags.
        # WHY: The constructor must call wandb.init with the correct arguments
        #      so the run appears in the right project/entity with proper metadata.
        # EXPECTED: wandb.init called once with matching kwargs.

        from mltk.integrations.wandb_adapter import WandbLogger

        WandbLogger(
            project="my-tests",
            entity="my-team",
            run_name="nightly-run",
            tags=["nightly", "v2"],
        )

        mock_wandb.init.assert_called_once_with(
            project="my-tests",
            entity="my-team",
            name="nightly-run",
            tags=["nightly", "v2"],
        )

    def test_init_default_args(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: WandbLogger is created with no arguments (all defaults).
        # WHY: The most common usage is just ``WandbLogger()`` with defaults.
        #      Default project should be "mltk-tests", entity and name should
        #      be None, tags should be an empty list.
        # EXPECTED: wandb.init called with default values.

        from mltk.integrations.wandb_adapter import WandbLogger

        WandbLogger()

        mock_wandb.init.assert_called_once_with(
            project="mltk-tests",
            entity=None,
            name=None,
            tags=[],
        )


# ---------------------------------------------------------------------------
# Tests: log_result
# ---------------------------------------------------------------------------

class TestLogResult:
    """WandbLogger.log_result -- single-result metric logging."""

    def test_log_result_metrics(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: A single passing test result is logged.
        # WHY: log_result must translate a result dict into flat W&B metrics
        #      with the correct prefix (mltk/{name}/...) and values. This is
        #      the core contract for streaming result logging.
        # EXPECTED: wandb.log called with passed=1, duration_ms, severity.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        result = _make_result("bias_check", passed=True, duration_ms=42.5, severity="warning")
        logger.log_result(result)

        # wandb.log should have been called (init call + log call)
        logged = mock_wandb.log.call_args[0][0]

        assert logged["mltk/bias_check/passed"] == 1
        assert logged["mltk/bias_check/duration_ms"] == 42.5
        assert logged["mltk/bias_check/severity"] == 1  # warning = 1

    def test_log_result_failed(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: A single failed test result is logged.
        # WHY: Failed results must log passed=0 so W&B dashboards can
        #      differentiate failures from successes in metric charts.
        # EXPECTED: mltk/{name}/passed == 0.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        result = _make_result("latency_p99", passed=False, severity="critical")
        logger.log_result(result)

        logged = mock_wandb.log.call_args[0][0]
        assert logged["mltk/latency_p99/passed"] == 0
        assert logged["mltk/latency_p99/severity"] == 2  # critical = 2

    def test_log_result_with_numeric_details(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: A result includes numeric values in the details dict.
        # WHY: Many mltk assertions produce numeric details (e.g., PSI score,
        #      p-value, threshold). These should be logged as additional metrics
        #      for fine-grained monitoring.
        # EXPECTED: Numeric detail values appear as mltk/{name}/{key} metrics.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        result = _make_result(
            "drift_psi",
            passed=False,
            details={"psi_score": 0.35, "threshold": 0.25, "note": "above limit"},
        )
        logger.log_result(result)

        logged = mock_wandb.log.call_args[0][0]
        assert logged["mltk/drift_psi/psi_score"] == 0.35
        assert logged["mltk/drift_psi/threshold"] == 0.25
        # Non-numeric "note" should NOT be in logged metrics
        assert "mltk/drift_psi/note" not in logged


# ---------------------------------------------------------------------------
# Tests: log_suite
# ---------------------------------------------------------------------------

class TestLogSuite:
    """WandbLogger.log_suite -- suite-level table and summary logging."""

    def test_log_suite_creates_table(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: A mixed suite of 3 results is logged via log_suite.
        # WHY: log_suite must create a wandb.Table with the correct columns
        #      and one row per result. The Table is the primary way users
        #      browse individual test results in the W&B UI.
        # EXPECTED: wandb.Table created with 5 columns, add_data called 3 times.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        results = _make_mixed_results()
        logger.log_suite(results)

        # Table was created with correct columns
        mock_wandb.Table.assert_called_once_with(
            columns=["name", "passed", "severity", "duration_ms", "message"]
        )

        # add_data called once per result
        table_mock = mock_wandb.Table.return_value
        assert table_mock.add_data.call_count == 3

    def test_log_suite_summary_metrics(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: A mixed suite (2 passed, 1 failed) is logged.
        # WHY: Summary metrics (total, passed_count, failed_count, pass_rate,
        #      total_duration_ms) power dashboard trend charts. If pass_rate
        #      is wrong, alerting thresholds trigger false positives/negatives.
        # EXPECTED: Summary metrics have correct values.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        results = _make_mixed_results()
        logger.log_suite(results)

        # wandb.log is called twice: once for table, once for summary
        assert mock_wandb.log.call_count == 2

        summary_call = mock_wandb.log.call_args_list[1][0][0]
        assert summary_call["mltk/summary/total"] == 3
        assert summary_call["mltk/summary/passed_count"] == 2
        assert summary_call["mltk/summary/failed_count"] == 1
        assert abs(summary_call["mltk/summary/pass_rate"] - (2 / 3 * 100)) < 1e-6
        assert summary_call["mltk/summary/total_duration_ms"] == pytest.approx(28.0)


# ---------------------------------------------------------------------------
# Tests: finish
# ---------------------------------------------------------------------------

class TestFinish:
    """WandbLogger.finish -- run completion and URL retrieval."""

    def test_finish_returns_url(self, mock_wandb: MagicMock) -> None:
        # SCENARIO: finish() is called after logging is complete.
        # WHY: finish() must call wandb.finish() to close the run cleanly
        #      and return the run URL for CI artifact tracking or Slack
        #      notifications ("see results at <url>").
        # EXPECTED: wandb.finish called; return value is the run URL string.

        from mltk.integrations.wandb_adapter import WandbLogger

        logger = WandbLogger()
        url = logger.finish()

        mock_wandb.finish.assert_called_once()
        assert url == "https://wandb.ai/team/project/runs/abc123"


# ---------------------------------------------------------------------------
# Tests: wandb not installed
# ---------------------------------------------------------------------------

class TestWandbNotInstalled:
    """WandbLogger gracefully fails when wandb is absent."""

    def test_wandb_not_installed(self) -> None:
        # SCENARIO: wandb is not installed in the environment (ImportError).
        # WHY: mltk is an ML testing toolkit; wandb is an *optional* extra.
        #      Users who do not need W&B integration must get a clear error
        #      message with installation instructions, not a cryptic traceback.
        # EXPECTED: ImportError raised with "pip install wandb" in the message.

        original = sys.modules.pop("wandb", None)
        wandb_adapter_mod = sys.modules.pop(
            "mltk.integrations.wandb_adapter", None
        )
        try:
            with patch.dict(sys.modules, {"wandb": None}):  # type: ignore[dict-item]
                import importlib  # noqa: PLC0415

                import mltk.integrations.wandb_adapter as mod  # noqa: PLC0415

                importlib.reload(mod)
                with pytest.raises(ImportError, match="pip install wandb"):
                    mod.WandbLogger()
        finally:
            if original is not None:
                sys.modules["wandb"] = original
            if wandb_adapter_mod is not None:
                sys.modules["mltk.integrations.wandb_adapter"] = wandb_adapter_mod
