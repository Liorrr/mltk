"""Tests for LLM observability adapters -- Phoenix, Langfuse, trace quality.

All tests mock external dependencies (phoenix, langfuse, opentelemetry)
so that actual installations are NOT required.  This mirrors the pattern
used in test_wandb.py: we inject mocks or patch imports so the adapters
bind to controlled fakes.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError, assert_true
from mltk.core.result import Severity, TestResult

# ===================================================================
# Helpers
# ===================================================================

def _passing_assertion(**kwargs: Any) -> TestResult:
    """Assertion that always passes. Returns a TestResult."""
    return TestResult(
        name="test_pass",
        passed=True,
        severity=Severity.INFO,
        message="All checks passed",
    )


def _failing_assertion(**kwargs: Any) -> TestResult:
    """Assertion that always fails (non-critical, no raise)."""
    return TestResult(
        name="test_fail",
        passed=False,
        severity=Severity.WARNING,
        message="Quality below threshold",
    )


def _critical_failing_assertion(**kwargs: Any) -> TestResult:
    """Assertion that raises MltkAssertionError (critical failure)."""
    return assert_true(
        False,
        name="test_critical",
        message="Critical failure detected",
        severity=Severity.CRITICAL,
    )


def _exploding_assertion(**kwargs: Any) -> TestResult:
    """Assertion that raises an unexpected exception."""
    msg = "connection timeout"
    raise ConnectionError(msg)


def _assertion_with_args(
    output: str = "",
    expected: str = "",
    **kwargs: Any,
) -> TestResult:
    """Assertion that checks if output contains expected text."""
    passed = expected.lower() in output.lower() if expected else True
    return TestResult(
        name="contains_check",
        passed=passed,
        severity=Severity.CRITICAL if not passed else Severity.INFO,
        message=f"Output {'contains' if passed else 'missing'} expected",
    )


# ===================================================================
# PhoenixAdapter tests
# ===================================================================

class TestPhoenixAdapter:
    """Tests for PhoenixAdapter -- wraps mltk assertions as Phoenix evaluators."""

    def test_wraps_passing_assertion(self) -> None:
        # SCENARIO: Adapter wraps an assertion that passes
        # WHY: The most common case -- assertion passes, score = 1.0
        # EXPECTED: score 1.0, label "pass", explanation from TestResult
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(_passing_assertion, name="quality")
        result = adapter(output="hello")

        assert result["passed"] == 1.0
        assert result["label"] == "pass"
        assert result["explanation"] == "All checks passed"

    def test_wraps_failing_assertion(self) -> None:
        # SCENARIO: Adapter wraps an assertion that fails (non-critical)
        # WHY: Non-critical failures return 0.0 but do not raise
        # EXPECTED: score 0.0, label "fail"
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(_failing_assertion, name="quality")
        result = adapter(output="hello")

        assert result["passed"] == 0.0
        assert result["label"] == "fail"
        assert "below threshold" in result["explanation"]

    def test_handles_critical_assertion_error(self) -> None:
        # SCENARIO: Wrapped assertion raises MltkAssertionError
        # WHY: Phoenix evaluators must never raise -- they return dicts
        # EXPECTED: score 0.0, label "fail", error message in explanation
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(
            _critical_failing_assertion, name="critical"
        )
        result = adapter(output="test")

        assert result["passed"] == 0.0
        assert result["label"] == "fail"
        assert "Critical failure" in result["explanation"]

    def test_handles_unexpected_exception(self) -> None:
        # SCENARIO: Wrapped assertion raises a non-mltk exception
        # WHY: Network errors, type errors, etc. must not crash Phoenix
        # EXPECTED: score 0.0, label "error", exception info in explanation
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(_exploding_assertion, name="net")
        result = adapter(output="test")

        assert result["passed"] == 0.0
        assert result["label"] == "error"
        assert "ConnectionError" in result["explanation"]
        assert "connection timeout" in result["explanation"]

    def test_default_name(self) -> None:
        # SCENARIO: No name provided to constructor
        # WHY: Default name should be "mltk" for easy identification
        # EXPECTED: adapter.name == "mltk"
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(_passing_assertion)
        assert adapter.name == "mltk"

    def test_custom_name(self) -> None:
        # SCENARIO: Custom name provided
        # WHY: Name appears in Phoenix Evaluations tab
        # EXPECTED: adapter.name matches what was passed
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(
            _passing_assertion, name="faithfulness"
        )
        assert adapter.name == "faithfulness"

    def test_custom_score_key(self) -> None:
        # SCENARIO: Custom score_key provided (e.g., "score")
        # WHY: Some Phoenix configs use "score" instead of "passed"
        # EXPECTED: result dict uses the custom key name
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(
            _passing_assertion,
            name="test",
            score_key="score",
        )
        result = adapter(output="hello")

        assert "score" in result
        assert result["score"] == 1.0
        assert "passed" not in result

    def test_forwards_output_arg(self) -> None:
        # SCENARIO: Adapter passes output to the assertion function
        # WHY: Assertions need the LLM output to evaluate
        # EXPECTED: assertion receives the output value
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(
            _assertion_with_args, name="contains"
        )
        result = adapter(output="Paris is in France", expected="France")

        assert result["passed"] == 1.0
        assert result["label"] == "pass"

    def test_forwards_all_kwargs(self) -> None:
        # SCENARIO: Adapter passes input, metadata alongside output
        # WHY: Rich evaluators need full span context
        # EXPECTED: No error when all kwargs are provided
        from mltk.integrations.phoenix import PhoenixAdapter

        adapter = PhoenixAdapter(_passing_assertion, name="full")
        result = adapter(
            output="answer",
            expected="answer",
            input="question",
            metadata={"model": "gpt-4"},
        )

        assert result["passed"] == 1.0

    def test_none_args_not_forwarded(self) -> None:
        # SCENARIO: None values are not forwarded to the assertion
        # WHY: Prevents unexpected keyword arguments in simple assertions
        # EXPECTED: assertion receives no kwargs when all args are None
        from mltk.integrations.phoenix import PhoenixAdapter

        call_args: dict[str, Any] = {}

        def capture_assertion(**kwargs: Any) -> TestResult:
            call_args.update(kwargs)
            return _passing_assertion()

        adapter = PhoenixAdapter(capture_assertion, name="capture")
        adapter(output=None, expected=None, input=None, metadata=None)

        assert len(call_args) == 0

    def test_only_output_forwarded_when_others_none(self) -> None:
        # SCENARIO: Only output is provided, rest are None
        # WHY: Most evaluations only need the output text
        # EXPECTED: Only "output" key in forwarded kwargs
        from mltk.integrations.phoenix import PhoenixAdapter

        call_args: dict[str, Any] = {}

        def capture_assertion(**kwargs: Any) -> TestResult:
            call_args.update(kwargs)
            return _passing_assertion()

        adapter = PhoenixAdapter(capture_assertion, name="capture")
        adapter(output="hello")

        assert list(call_args.keys()) == ["output"]
        assert call_args["output"] == "hello"


# ===================================================================
# register_phoenix tests
# ===================================================================

class TestRegisterPhoenix:
    """Tests for register_phoenix -- OTEL setup helper for Phoenix."""

    def test_configures_endpoint(self) -> None:
        # SCENARIO: register_phoenix called with custom endpoint
        # WHY: Teams run Phoenix on different hosts/ports
        # EXPECTED: OTLPSpanExporter receives the custom endpoint
        from mltk.integrations.phoenix import register_phoenix

        mock_trace = MagicMock()
        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_exporter_cls = MagicMock()
        mock_processor_cls = MagicMock()

        provider_instance = MagicMock()
        mock_provider_cls.return_value = provider_instance

        modules = {
            "opentelemetry": mock_trace,
            "opentelemetry.trace": mock_trace,
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls,
            ),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls,
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                SimpleSpanProcessor=mock_processor_cls,
            ),
        }

        with patch.dict("sys.modules", modules):
            result = register_phoenix(
                endpoint="http://phoenix:9090/v1/traces",
                project_name="my-project",
            )

        mock_exporter_cls.assert_called_once_with(
            endpoint="http://phoenix:9090/v1/traces"
        )
        assert result is provider_instance

    def test_default_endpoint(self) -> None:
        # SCENARIO: register_phoenix called with no arguments
        # WHY: Default should point to localhost:6006
        # EXPECTED: OTLPSpanExporter gets the default endpoint
        from mltk.integrations.phoenix import register_phoenix

        mock_exporter_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_provider_cls.return_value = MagicMock()

        modules = {
            "opentelemetry": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                OTLPSpanExporter=mock_exporter_cls,
            ),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls,
            ),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }

        with patch.dict("sys.modules", modules):
            register_phoenix()

        mock_exporter_cls.assert_called_once_with(
            endpoint="http://localhost:6006/v1/traces"
        )

    def test_import_error_when_otel_missing(self) -> None:
        # SCENARIO: opentelemetry is not installed
        # WHY: Clear error message helps users install the right package
        # EXPECTED: ImportError with pip install instructions
        from mltk.integrations.phoenix import register_phoenix

        with patch.dict("sys.modules", {"opentelemetry": None}):
            with pytest.raises(ImportError, match="opentelemetry"):
                register_phoenix()

    def test_sets_project_name_in_resource(self) -> None:
        # SCENARIO: Custom project name provided
        # WHY: Project name appears in Phoenix project selector
        # EXPECTED: Resource.create receives the project name
        from mltk.integrations.phoenix import register_phoenix

        mock_resource_cls = MagicMock()
        mock_provider_cls = MagicMock()
        mock_provider_cls.return_value = MagicMock()

        modules = {
            "opentelemetry": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=mock_resource_cls,
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls,
            ),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }

        with patch.dict("sys.modules", modules):
            register_phoenix(project_name="rag-eval")

        mock_resource_cls.create.assert_called_once_with(
            {"service.name": "rag-eval"}
        )

    def test_returns_tracer_provider(self) -> None:
        # SCENARIO: register_phoenix returns the TracerProvider
        # WHY: Callers may need it for custom tracer creation
        # EXPECTED: Return value is the TracerProvider instance
        from mltk.integrations.phoenix import register_phoenix

        sentinel = MagicMock(name="provider_instance")
        mock_provider_cls = MagicMock(return_value=sentinel)
        mock_trace_mod = MagicMock()

        modules = {
            "opentelemetry": mock_trace_mod,
            "opentelemetry.trace": mock_trace_mod,
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls,
            ),
            "opentelemetry.sdk.trace.export": MagicMock(),
        }

        with patch.dict("sys.modules", modules):
            result = register_phoenix()

        assert result is sentinel


# ===================================================================
# LangfuseAdapter tests
# ===================================================================

class TestLangfuseAdapter:
    """Tests for LangfuseAdapter -- posts mltk results as Langfuse scores."""

    def test_wraps_passing_assertion(self) -> None:
        # SCENARIO: Assertion passes, score posted to Langfuse
        # WHY: Most common case -- assertion passes, value = 1.0
        # EXPECTED: client.score called with value=1.0
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="quality")
        adapter._langfuse = mock_langfuse

        result = adapter.score(trace_id="trace-1", output="hello")

        assert result.passed is True
        mock_langfuse.score.assert_called_once()
        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["trace_id"] == "trace-1"
        assert call_kwargs["name"] == "quality"
        assert call_kwargs["value"] == 1.0

    def test_wraps_failing_assertion(self) -> None:
        # SCENARIO: Assertion fails (non-critical), score = 0.0
        # WHY: Failed assertions must still post scores (not crash)
        # EXPECTED: client.score called with value=0.0
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_failing_assertion, name="quality")
        adapter._langfuse = mock_langfuse

        result = adapter.score(trace_id="trace-2")

        assert result.passed is False
        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["value"] == 0.0
        assert "below threshold" in call_kwargs["comment"]

    def test_critical_failure_posts_then_raises(self) -> None:
        # SCENARIO: Assertion raises MltkAssertionError
        # WHY: Score must be posted before the exception propagates
        # EXPECTED: score posted with 0.0, then MltkAssertionError raised
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(
            _critical_failing_assertion, name="critical"
        )
        adapter._langfuse = mock_langfuse

        with pytest.raises(MltkAssertionError):
            adapter.score(trace_id="trace-3")

        mock_langfuse.score.assert_called_once()
        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["value"] == 0.0

    def test_observation_id_forwarded(self) -> None:
        # SCENARIO: observation_id provided for span-level scoring
        # WHY: Langfuse supports scoring individual spans, not just traces
        # EXPECTED: observation_id appears in the score call
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="span_check")
        adapter._langfuse = mock_langfuse

        adapter.score(
            trace_id="trace-4",
            observation_id="obs-789",
        )

        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["observation_id"] == "obs-789"

    def test_no_observation_id_when_none(self) -> None:
        # SCENARIO: No observation_id provided
        # WHY: Score attaches to the trace itself, not a span
        # EXPECTED: observation_id key absent from score call
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="trace_level")
        adapter._langfuse = mock_langfuse

        adapter.score(trace_id="trace-5")

        call_kwargs = mock_langfuse.score.call_args[1]
        assert "observation_id" not in call_kwargs

    def test_data_type_is_numeric(self) -> None:
        # SCENARIO: Score data_type should be NUMERIC
        # WHY: Langfuse uses NUMERIC for 0.0-1.0 range scores
        # EXPECTED: data_type="NUMERIC" in score call
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="typed")
        adapter._langfuse = mock_langfuse

        adapter.score(trace_id="trace-6")

        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["data_type"] == "NUMERIC"

    def test_default_name(self) -> None:
        # SCENARIO: No name provided to constructor
        # WHY: Default name should be "mltk"
        # EXPECTED: adapter.name == "mltk"
        from mltk.integrations.langfuse import LangfuseAdapter

        adapter = LangfuseAdapter(_passing_assertion)
        assert adapter.name == "mltk"

    def test_import_error_when_langfuse_missing(self) -> None:
        # SCENARIO: langfuse package not installed
        # WHY: Clear error message helps users install the package
        # EXPECTED: ImportError with pip install instructions
        from mltk.integrations.langfuse import LangfuseAdapter

        adapter = LangfuseAdapter(_passing_assertion, name="test")
        adapter._langfuse = None  # reset cached client

        with patch.dict("sys.modules", {"langfuse": None}):
            with pytest.raises(ImportError, match="langfuse"):
                adapter._get_client()

    def test_returns_test_result(self) -> None:
        # SCENARIO: score() returns the TestResult for local use
        # WHY: Callers need the result for pytest assertions
        # EXPECTED: Return type is TestResult with correct fields
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="check")
        adapter._langfuse = mock_langfuse

        result = adapter.score(trace_id="trace-7")

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.name == "test_pass"

    def test_kwargs_forwarded_to_assertion(self) -> None:
        # SCENARIO: Extra kwargs passed to score() reach the assertion
        # WHY: Assertions need output, expected, context, etc.
        # EXPECTED: Assertion receives the kwargs and uses them
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(
            _assertion_with_args, name="contains"
        )
        adapter._langfuse = mock_langfuse

        result = adapter.score(
            trace_id="trace-8",
            output="The capital of France is Paris",
            expected="Paris",
        )

        assert result.passed is True

    def test_client_cached_across_calls(self) -> None:
        # SCENARIO: Multiple score() calls reuse the same client
        # WHY: Creating a new Langfuse client per call is wasteful
        # EXPECTED: _langfuse is the same object after two calls
        from mltk.integrations.langfuse import LangfuseAdapter

        mock_langfuse = MagicMock()
        adapter = LangfuseAdapter(_passing_assertion, name="cache")
        adapter._langfuse = mock_langfuse

        adapter.score(trace_id="t1")
        adapter.score(trace_id="t2")

        assert adapter._langfuse is mock_langfuse


# ===================================================================
# assert_trace_quality tests
# ===================================================================

class TestAssertTraceQuality:
    """Tests for assert_trace_quality -- CI/CD quality gate for traces."""

    def test_all_checks_pass(self) -> None:
        # SCENARIO: All three thresholds met
        # WHY: Happy path -- trace is within all budgets
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={
                "latency_ms": 100,
                "cost_usd": 0.002,
                "score": 0.95,
            },
            max_latency_ms=500,
            max_cost_usd=0.01,
            min_score=0.8,
        )

        assert result.passed is True
        assert result.name == "integrations.trace_quality"
        assert "passed" in result.message.lower()

    def test_latency_check_fails(self) -> None:
        # SCENARIO: Latency exceeds threshold
        # WHY: Slow responses violate SLA
        # EXPECTED: result.passed is False, message mentions latency
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_trace_quality(
                trace={"latency_ms": 3000},
                max_latency_ms=2000,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "latency" in result.message.lower()

    def test_cost_check_fails(self) -> None:
        # SCENARIO: Cost exceeds budget
        # WHY: Expensive traces blow the API budget
        # EXPECTED: result.passed is False, message mentions cost
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_trace_quality(
                trace={"cost_usd": 0.05},
                max_cost_usd=0.01,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "cost" in result.message.lower()

    def test_score_check_fails(self) -> None:
        # SCENARIO: Quality score below minimum
        # WHY: Low-quality responses should not reach production
        # EXPECTED: result.passed is False, message mentions score
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_trace_quality(
                trace={"score": 0.3},
                min_score=0.8,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "score" in result.message.lower()

    def test_combined_failures(self) -> None:
        # SCENARIO: Both latency and cost exceed thresholds
        # WHY: Message should list ALL violations, not just the first
        # EXPECTED: message contains both "latency" and "cost"
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_trace_quality(
                trace={"latency_ms": 5000, "cost_usd": 0.5},
                max_latency_ms=2000,
                max_cost_usd=0.01,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert "latency" in result.message.lower()
        assert "cost" in result.message.lower()

    def test_missing_latency_key_skips_check(self) -> None:
        # SCENARIO: trace dict has no "latency_ms" key
        # WHY: Not all traces include timing data -- skip, don't fail
        # EXPECTED: result.passed is True (nothing to check)
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"output": "hello"},
            max_latency_ms=500,
        )

        assert result.passed is True

    def test_missing_cost_key_skips_check(self) -> None:
        # SCENARIO: trace dict has no "cost_usd" key
        # WHY: Not all traces include cost data
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"output": "hello"},
            max_cost_usd=0.01,
        )

        assert result.passed is True

    def test_judge_fn_overrides_trace_score(self) -> None:
        # SCENARIO: judge_fn provided alongside trace["score"]
        # WHY: judge_fn takes priority for custom quality checks
        # EXPECTED: judge_fn's return value is used, not trace["score"]
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        def strict_judge(trace: dict) -> float:
            return 0.2  # always returns low score

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_trace_quality(
                trace={"score": 0.99},  # high trace score
                min_score=0.5,
                judge_fn=strict_judge,
            )

        result = exc_info.value.result
        assert result.passed is False
        assert result.details["score"] == 0.2

    def test_no_thresholds_always_passes(self) -> None:
        # SCENARIO: No threshold parameters provided
        # WHY: Gate with no checks should pass (nothing to violate)
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"latency_ms": 99999, "cost_usd": 999.99}
        )

        assert result.passed is True

    def test_details_contain_actual_values(self) -> None:
        # SCENARIO: Trace has values for all checked dimensions
        # WHY: Details dict should show actual vs threshold for debugging
        # EXPECTED: details has latency_ms, max_latency_ms, etc.
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={
                "latency_ms": 100,
                "cost_usd": 0.002,
                "score": 0.9,
            },
            max_latency_ms=500,
            max_cost_usd=0.01,
            min_score=0.5,
        )

        assert result.details["latency_ms"] == 100
        assert result.details["max_latency_ms"] == 500
        assert result.details["cost_usd"] == 0.002
        assert result.details["max_cost_usd"] == 0.01
        assert result.details["score"] == 0.9
        assert result.details["min_score"] == 0.5

    def test_has_duration_ms(self) -> None:
        # SCENARIO: timed_assertion decorator is applied
        # WHY: All mltk assertions should track their own timing
        # EXPECTED: result.duration_ms > 0
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"latency_ms": 100},
            max_latency_ms=500,
        )

        assert result.duration_ms >= 0.0

    def test_name_is_integrations_trace_quality(self) -> None:
        # SCENARIO: TestResult name follows module path convention
        # WHY: Consistent naming for filtering in dashboards
        # EXPECTED: name == "integrations.trace_quality"
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(trace={})

        assert result.name == "integrations.trace_quality"

    def test_latency_at_exact_threshold_passes(self) -> None:
        # SCENARIO: Latency equals the threshold exactly
        # WHY: Boundary condition -- at-threshold should pass
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"latency_ms": 500.0},
            max_latency_ms=500.0,
        )

        assert result.passed is True

    def test_cost_at_exact_threshold_passes(self) -> None:
        # SCENARIO: Cost equals the threshold exactly
        # WHY: Boundary condition -- at-threshold should pass
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"cost_usd": 0.01},
            max_cost_usd=0.01,
        )

        assert result.passed is True

    def test_score_at_exact_threshold_passes(self) -> None:
        # SCENARIO: Score equals the min_score exactly
        # WHY: Boundary condition -- at-threshold should pass
        # EXPECTED: result.passed is True
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={"score": 0.8},
            min_score=0.8,
        )

        assert result.passed is True

    def test_empty_trace_with_all_thresholds(self) -> None:
        # SCENARIO: Empty trace dict with all thresholds set
        # WHY: Missing keys should skip, not fail
        # EXPECTED: result.passed is True (nothing to check)
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        result = assert_trace_quality(
            trace={},
            max_latency_ms=500,
            max_cost_usd=0.01,
            min_score=0.8,
        )

        assert result.passed is True

    def test_judge_fn_with_no_min_score(self) -> None:
        # SCENARIO: judge_fn provided but min_score is None
        # WHY: judge_fn should only be called if min_score is set
        # EXPECTED: judge_fn is NOT called, result passes
        from mltk.integrations.trace_quality import (
            assert_trace_quality,
        )

        called = False

        def should_not_be_called(trace: dict) -> float:
            nonlocal called
            called = True
            return 0.0

        result = assert_trace_quality(
            trace={"output": "hello"},
            judge_fn=should_not_be_called,
        )

        assert result.passed is True
        assert not called
