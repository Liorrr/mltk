"""Arize Phoenix integration -- use mltk assertions as Phoenix evaluators.

Arize Phoenix is an open-source LLM observability platform built entirely
on OpenTelemetry.  It provides trace visualization, span debugging, and
an evaluation framework where custom evaluators score individual spans.

**Why this matters for ML teams:**

Most teams evaluating LLM systems already live inside Phoenix for trace
visualization.  Without this adapter, mltk assertions produce TestResult
objects that exist only in pytest output.  With it, every mltk assertion
becomes a named evaluator inside Phoenix -- visible in the Evaluations
tab alongside built-in metrics like relevance and hallucination.

**Architecture:**

``PhoenixAdapter`` wraps any mltk assertion function as a Phoenix-compatible
callable evaluator.  Phoenix calls the adapter with span attributes (a dict
containing ``output``, ``expected``, ``input``, ``metadata``), and the
adapter returns a score dict with ``score``, ``label``, and ``explanation``.

``register_phoenix`` is a one-line setup helper that configures the
OpenTelemetry SDK to send spans to a running Phoenix instance.  This
means existing ``MltkTracer`` spans appear in Phoenix without code changes.

**Dependencies:**

Both ``arize-phoenix`` and ``opentelemetry-sdk`` are optional.  Import
errors are caught and re-raised with clear pip install instructions.
If neither is installed, you can still import this module -- errors only
occur when you call the functions.

Typical usage::

    from mltk.integrations.phoenix import PhoenixAdapter
    from mltk.domains.llm import assert_faithfulness

    # Wrap any mltk assertion as a Phoenix evaluator
    evaluator = PhoenixAdapter(assert_faithfulness, name="faithfulness")

    # Phoenix calls this with span attributes
    score = evaluator(output="Paris is in France", expected="France")
    # => {"score": 1.0, "label": "pass", "explanation": "..."}

    # One-line setup to send MltkTracer spans to Phoenix
    from mltk.integrations.phoenix import register_phoenix
    tracer = register_phoenix(endpoint="http://localhost:6006/v1/traces")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.result import TestResult


class PhoenixAdapter:
    """Wrap any mltk assertion as a Phoenix evaluator callable.

    Phoenix evaluators follow a simple protocol: they receive a dict
    of span attributes and return a dict with ``score``, ``label``,
    and ``explanation`` keys.  This adapter bridges mltk's TestResult
    model to that protocol.

    **How it works:**

    1. Phoenix passes span attributes as keyword arguments.
    2. The adapter forwards them to the wrapped mltk assertion.
    3. The assertion returns a ``TestResult``.
    4. The adapter maps ``TestResult.passed`` to a numeric score
       (1.0 for pass, 0.0 for fail) and ``TestResult.message`` to
       the explanation string.

    **Score key:**

    The ``score_key`` parameter controls which field name holds the
    numeric score in the returned dict.  Phoenix expects ``"score"``
    by default, but some custom dashboards use different keys.

    Example::

        from mltk.integrations.phoenix import PhoenixAdapter

        def my_assertion(output, **kwargs):
            from mltk.core.assertion import assert_true
            return assert_true(
                len(output) > 0,
                name="non_empty",
                message="Output is not empty",
            )

        adapter = PhoenixAdapter(my_assertion, name="non_empty")
        result = adapter(output="hello")
        assert result["score"] == 1.0
        assert result["label"] == "pass"

    Args:
        assertion_fn: Any mltk assertion function that returns a
            ``TestResult``.  The function must accept keyword arguments
            matching the span attribute names (typically ``output``,
            ``expected``, ``input``, ``metadata``).
        name: Human-readable name shown in the Phoenix Evaluations tab.
            Defaults to ``"mltk"``.
        score_key: Key name for the numeric score in the returned dict.
            Defaults to ``"passed"``.
    """

    def __init__(
        self,
        assertion_fn: Callable[..., TestResult],
        name: str = "mltk",
        score_key: str = "passed",
    ) -> None:
        self._fn = assertion_fn
        self.name = name
        self.score_key = score_key

    def __call__(
        self,
        output: Any = None,
        expected: Any = None,
        input: Any = None,  # noqa: A002
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a span using the wrapped mltk assertion.

        This method is the Phoenix evaluator protocol: Phoenix calls
        it with span attributes extracted from a trace, and expects a
        dict back with score information.

        **Error handling:**

        If the assertion function raises an ``MltkAssertionError``
        (which happens on critical failures), the adapter catches it
        and returns a score of 0.0 with the error message as the
        explanation.  Any other exception is also caught and mapped
        to a 0.0 score so that Phoenix never sees an unhandled error
        from an evaluator.

        Args:
            output: The LLM output text from the span.
            expected: The expected/reference answer, if available.
            input: The original input/prompt to the LLM.
            metadata: Additional span metadata as a dict.

        Returns:
            A dict with three keys:

            - ``score`` (or the configured ``score_key``): ``1.0`` if
              the assertion passed, ``0.0`` if it failed.
            - ``label``: ``"pass"`` or ``"fail"``.
            - ``explanation``: The human-readable message from the
              ``TestResult``.
        """
        from mltk.core.assertion import MltkAssertionError

        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if expected is not None:
            kwargs["expected"] = expected
        if input is not None:
            kwargs["input"] = input
        if metadata is not None:
            kwargs["metadata"] = metadata

        try:
            result = self._fn(**kwargs)
        except MltkAssertionError as exc:
            return {
                self.score_key: 0.0,
                "label": "fail",
                "explanation": str(exc),
            }
        except Exception as exc:
            return {
                self.score_key: 0.0,
                "label": "error",
                "explanation": f"Assertion raised {type(exc).__name__}: {exc}",
            }

        score = 1.0 if result.passed else 0.0
        label = "pass" if result.passed else "fail"
        explanation = result.message

        return {
            self.score_key: score,
            "label": label,
            "explanation": explanation,
        }


def register_phoenix(
    endpoint: str = "http://localhost:6006/v1/traces",
    project_name: str = "mltk",
) -> Any:
    """Configure OpenTelemetry to send spans to a Phoenix instance.

    This is a one-line setup helper for teams already running Arize
    Phoenix.  It creates an OTLP HTTP exporter pointed at the Phoenix
    collector endpoint and registers it as the global tracer provider.

    After calling this function, any code using ``MltkTracer`` (or any
    other OpenTelemetry-instrumented library) will automatically send
    spans to Phoenix for visualization.

    **How Phoenix receives spans:**

    Phoenix runs an OTLP-compatible collector on ``/v1/traces``.  The
    OpenTelemetry SDK sends spans via HTTP POST to that endpoint.
    Phoenix then indexes them by project and makes them queryable
    through its UI and REST API.

    **Why a helper function:**

    Setting up OpenTelemetry correctly requires importing three
    packages (``opentelemetry-sdk``, ``opentelemetry-api``, and
    ``opentelemetry-exporter-otlp-proto-http``), creating a resource,
    building a tracer provider, attaching a span processor, and
    registering it globally.  This function does all of that in one
    call.

    Args:
        endpoint: The OTLP HTTP endpoint of the Phoenix instance.
            Defaults to ``"http://localhost:6006/v1/traces"`` (the
            standard Phoenix local development port).
        project_name: Project name to tag spans with.  Appears as the
            project selector in the Phoenix UI.  Defaults to
            ``"mltk"``.

    Returns:
        The configured ``TracerProvider`` instance.  Callers can use
        this to create additional tracers or to shut down the provider
        when done.

    Raises:
        ImportError: If ``opentelemetry-sdk`` or
            ``opentelemetry-exporter-otlp-proto-http`` is not
            installed, with pip install instructions.

    Example::

        from mltk.integrations.phoenix import register_phoenix

        provider = register_phoenix(
            endpoint="http://phoenix.internal:6006/v1/traces",
            project_name="my-llm-app",
        )
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry SDK is required for register_phoenix. "
            "Install it with: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        ) from exc

    resource = Resource.create({"service.name": project_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return provider
