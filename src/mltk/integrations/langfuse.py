"""Langfuse integration -- post mltk assertion results as Langfuse scores.

Langfuse is the most popular open-source LLM observability platform
(24K+ GitHub stars, MIT license).  It provides trace visualization,
prompt management, and a scoring API for attaching evaluation results
to individual traces.

**Why this matters for ML teams:**

Teams using Langfuse for production LLM monitoring need their quality
gates visible in the same dashboard as latency, cost, and prompt
analytics.  Without this adapter, mltk assertions produce TestResult
objects trapped in pytest output.  With it, every assertion result
becomes a numeric score attached to a Langfuse trace -- filterable,
chartable, and alertable.

**Architecture:**

Langfuse evaluations are *scores on traces*, not evaluators that run
inside the platform.  The workflow is:

1. Your LLM app creates traces in Langfuse (via decorator or SDK).
2. mltk assertions run on the LLM outputs (in CI or locally).
3. ``LangfuseAdapter`` posts the assertion results back to Langfuse
   as scores, linked to the original trace by ``trace_id``.

This is Pattern C from the research spec: mltk runs first, then
pushes results.  It contrasts with Phoenix's Pattern B where the
platform invokes mltk as an evaluator.

**Dependencies:**

``langfuse`` is an optional dependency.  A clear ImportError with
installation instructions is raised if it is not available.

Typical usage::

    from mltk.integrations.langfuse import LangfuseAdapter

    adapter = LangfuseAdapter(my_assertion, name="faithfulness")
    adapter.score(trace_id="abc-123", output="Paris is in France")

    # Or with observation-level granularity:
    adapter.score(
        trace_id="abc-123",
        observation_id="span-456",
        output="Paris is in France",
        expected="France",
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.result import TestResult


class LangfuseAdapter:
    """Wrap an mltk assertion as a Langfuse score function.

    Each call to :meth:`score` runs the wrapped assertion and posts
    the result as a numeric score to a Langfuse trace.  This bridges
    mltk's TestResult model to Langfuse's scoring API.

    **How scoring works in Langfuse:**

    Langfuse scores are numeric values (0.0--1.0) or boolean (0/1)
    attached to a trace or observation (span).  Each score has a
    ``name``, ``value``, and optional ``comment``.  Scores appear in
    the Langfuse dashboard as filterable columns, enabling queries
    like "show all traces where faithfulness < 0.8".

    **Why wrap assertions instead of posting raw results:**

    The adapter runs the assertion at scoring time, so you get both
    the TestResult (for local pytest assertions) and the Langfuse
    score (for dashboard visibility) from a single call.  You do not
    need to run assertions separately and then manually map results
    to scores.

    Example::

        from mltk.integrations.langfuse import LangfuseAdapter
        from mltk.core.assertion import assert_true

        def check_length(output, **kwargs):
            return assert_true(
                len(output) > 10,
                name="min_length",
                message=f"Output has {len(output)} chars",
            )

        adapter = LangfuseAdapter(check_length, name="min_length")
        result = adapter.score(
            trace_id="trace-abc",
            output="Hello, world! This is a test.",
        )

    Args:
        assertion_fn: Any mltk assertion function that returns a
            ``TestResult``.  Must accept keyword arguments matching
            the data you will pass to :meth:`score`.
        name: Score name shown in the Langfuse dashboard.  Defaults
            to ``"mltk"``.  Langfuse prefixes this in its UI, so
            a name like ``"faithfulness"`` becomes clearly visible.
    """

    def __init__(
        self,
        assertion_fn: Callable[..., TestResult],
        name: str = "mltk",
    ) -> None:
        self._fn = assertion_fn
        self.name = name
        self._langfuse: Any = None

    def _get_client(self) -> Any:
        """Lazy-import and cache the Langfuse client.

        **Why lazy import:**

        mltk is an ML testing toolkit with many optional integrations.
        Users who do not use Langfuse should not need it installed.
        The import happens at first use (not module load time) so that
        ``import mltk`` never fails due to a missing dependency.

        Returns:
            A ``Langfuse`` client instance.

        Raises:
            ImportError: If ``langfuse`` is not installed, with a
                pip install hint.
        """
        if self._langfuse is not None:
            return self._langfuse
        try:
            from langfuse import Langfuse  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langfuse is required for LangfuseAdapter but is "
                "not installed. "
                "Install it with: pip install langfuse"
            ) from exc
        self._langfuse = Langfuse()
        return self._langfuse

    def score(
        self,
        trace_id: str,
        observation_id: str | None = None,
        **kwargs: Any,
    ) -> TestResult:
        """Run the assertion and post the result as a Langfuse score.

        This method does two things:

        1. Calls the wrapped assertion function with ``**kwargs``,
           producing a ``TestResult``.
        2. Posts the result to Langfuse as a numeric score attached
           to the given ``trace_id`` (and optionally an
           ``observation_id`` for span-level granularity).

        **Score mapping:**

        - ``TestResult.passed = True`` maps to ``value = 1.0``
        - ``TestResult.passed = False`` maps to ``value = 0.0``
        - ``TestResult.message`` becomes the score ``comment``
        - ``data_type`` is always ``"NUMERIC"`` (0.0--1.0 range)

        **Error handling:**

        If the assertion raises ``MltkAssertionError`` (critical
        failure), the adapter catches it, posts a score of 0.0
        with the error message, and re-raises the exception so
        pytest still sees the failure.

        Args:
            trace_id: The Langfuse trace ID to attach the score to.
                This is the ID returned when creating a trace via
                the Langfuse SDK or decorator.
            observation_id: Optional observation (span) ID for
                span-level scoring.  When None, the score is
                attached to the trace itself.
            **kwargs: Arguments forwarded to the assertion function
                (e.g., ``output``, ``expected``, ``context``).

        Returns:
            The ``TestResult`` from the assertion function, so callers
            can use it for local assertions as well.

        Raises:
            MltkAssertionError: If the assertion fails with critical
                severity (re-raised after posting the score).
        """
        from mltk.core.assertion import MltkAssertionError

        client = self._get_client()

        try:
            result = self._fn(**kwargs)
        except MltkAssertionError as exc:
            score_kwargs: dict[str, Any] = {
                "trace_id": trace_id,
                "name": self.name,
                "value": 0.0,
                "comment": str(exc),
                "data_type": "NUMERIC",
            }
            if observation_id is not None:
                score_kwargs["observation_id"] = observation_id
            client.score(**score_kwargs)
            raise

        value = 1.0 if result.passed else 0.0

        score_kwargs = {
            "trace_id": trace_id,
            "name": self.name,
            "value": value,
            "comment": result.message,
            "data_type": "NUMERIC",
        }
        if observation_id is not None:
            score_kwargs["observation_id"] = observation_id

        client.score(**score_kwargs)

        return result
