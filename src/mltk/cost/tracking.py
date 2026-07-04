"""Accumulate LLM token/cost usage across a run or suite, and assert on budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.cost.pricing import estimate_cost


@dataclass
class UsageRecord:
    """Single LLM call record with token counts and computed cost."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    label: str | None = None


class CostTracker:
    """Accumulate LLM token and cost usage across a run or test suite.

    Call :meth:`record` after each LLM call; query :attr:`total_cost_usd` or
    :attr:`total_tokens` at any time, then pass the tracker to
    :func:`assert_cost_within` or :func:`assert_token_usage`.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record("gpt-4o", 1000, 200)
        UsageRecord(...)
        >>> tracker.total_cost_usd
        0.004500000000000001
    """

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        label: str | None = None,
    ) -> UsageRecord:
        """Record one LLM call; cost is computed via :func:`estimate_cost`.

        Args:
            model: Model identifier (must exist in pricing table or overrides).
            input_tokens: Prompt token count.
            output_tokens: Completion token count.
            label: Optional human-readable tag for this call (e.g. "step-3").

        Returns:
            The :class:`UsageRecord` that was appended.

        Raises:
            ValueError: Propagated from :func:`estimate_cost` on unknown model.
        """
        cost = estimate_cost(model, input_tokens, output_tokens)
        rec = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            label=label,
        )
        self.records.append(rec)
        return rec

    @property
    def total_cost_usd(self) -> float:
        """Sum of cost_usd across all recorded calls."""
        return sum((r.cost_usd for r in self.records), 0.0)

    @property
    def total_tokens(self) -> int:
        """Total input + output tokens across all recorded calls."""
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def by_model(self) -> dict[str, dict[str, float]]:
        """Return a per-model usage summary.

        Returns:
            Dict mapping model_id -> {"tokens": float, "cost_usd": float, "calls": float}.
        """
        result: dict[str, dict[str, float]] = {}
        for rec in self.records:
            entry = result.setdefault(
                rec.model, {"tokens": 0, "cost_usd": 0.0, "calls": 0}
            )
            entry["tokens"] += rec.input_tokens + rec.output_tokens
            entry["cost_usd"] += rec.cost_usd
            entry["calls"] += 1
        return result

    def reset(self) -> None:
        """Clear all records, resetting totals to zero."""
        self.records.clear()


@timed_assertion
def assert_cost_within(
    usage: CostTracker | float,
    max_usd: float,
    *,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that total LLM spend does not exceed *max_usd*.

    Args:
        usage: A :class:`CostTracker` (uses :attr:`~CostTracker.total_cost_usd`)
               or a raw float USD amount.
        max_usd: Budget ceiling in USD.
        severity: ``CRITICAL`` (default) raises on failure; ``WARNING`` records it.

    Returns:
        :class:`~mltk.core.result.TestResult` with name ``"cost.within_budget"``.

    Raises:
        MltkAssertionError: When *usage* > *max_usd* and severity is ``CRITICAL``.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record("gpt-4o-mini", 1000, 500)
        UsageRecord(...)
        >>> assert_cost_within(tracker, 1.00)
    """
    if isinstance(usage, CostTracker):
        actual_cost = usage.total_cost_usd
        by_model_summary: dict[str, Any] = usage.by_model()
    else:
        actual_cost = float(usage)
        by_model_summary = {}

    passed = actual_cost <= max_usd
    message = (
        f"Cost within budget: ${actual_cost:.6f} <= ${max_usd:.6f}"
        if passed
        else f"Cost exceeds budget: ${actual_cost:.6f} > ${max_usd:.6f}"
    )

    return assert_true(
        passed,
        name="cost.within_budget",
        message=message,
        severity=severity,
        total_cost_usd=actual_cost,
        max_usd=max_usd,
        by_model=by_model_summary,
    )


@timed_assertion
def assert_token_usage(
    usage: CostTracker | int,
    max_tokens: int,
    *,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that total token consumption does not exceed *max_tokens*.

    Args:
        usage: A :class:`CostTracker` (uses :attr:`~CostTracker.total_tokens`)
               or a raw int token count.
        max_tokens: Token budget ceiling.
        severity: ``CRITICAL`` (default) raises on failure; ``WARNING`` records it.

    Returns:
        :class:`~mltk.core.result.TestResult` with name ``"cost.token_usage"``.

    Raises:
        MltkAssertionError: When *usage* > *max_tokens* and severity is ``CRITICAL``.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record("gpt-4o-mini", 1000, 500)
        UsageRecord(...)
        >>> assert_token_usage(tracker, 10_000)
    """
    if isinstance(usage, CostTracker):
        actual_tokens = usage.total_tokens
    else:
        actual_tokens = int(usage)

    passed = actual_tokens <= max_tokens
    message = (
        f"Token usage within limit: {actual_tokens} <= {max_tokens}"
        if passed
        else f"Token usage exceeds limit: {actual_tokens} > {max_tokens}"
    )

    return assert_true(
        passed,
        name="cost.token_usage",
        message=message,
        severity=severity,
        total_tokens=actual_tokens,
        max_tokens=max_tokens,
    )
