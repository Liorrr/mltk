"""Tests for mltk.cost.tracking -- CostTracker accumulator and budget assertions."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.cost.tracking import (
    CostTracker,
    UsageRecord,
    assert_cost_within,
    assert_token_usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_with_two_calls() -> CostTracker:
    """Return a CostTracker with two recorded calls on the same model.

    gpt-4.1-mini: (0.40/1M in, 1.60/1M out)
    Call 1 — 500 in + 200 out:  500/1e6*0.40 + 200/1e6*1.60 = 0.0002 + 0.00032 = 0.00052
    Call 2 — 300 in + 100 out:  300/1e6*0.40 + 100/1e6*1.60 = 0.00012 + 0.00016 = 0.00028
    Total cost: 0.00080   Total tokens: 1100
    """
    t = CostTracker()
    t.record("gpt-4.1-mini", 500, 200)
    t.record("gpt-4.1-mini", 300, 100)
    return t


# ---------------------------------------------------------------------------
# CostTracker unit tests
# ---------------------------------------------------------------------------

class TestUsageRecord:
    """record() return value shape."""

    def test_record_returns_usage_record(self) -> None:
        """record() returns a UsageRecord instance."""
        t = CostTracker()
        rec = t.record("gpt-4o-mini", 1_000, 500)
        assert isinstance(rec, UsageRecord)

    def test_record_fields_populated(self) -> None:
        """UsageRecord has correct model, token counts, and cost.

        gpt-4o-mini: (0.15/1M, 0.60/1M)
        1000 in + 500 out: 1000/1e6*0.15 + 500/1e6*0.60 = 0.00015 + 0.0003 = 0.00045
        """
        t = CostTracker()
        rec = t.record("gpt-4o-mini", 1_000, 500)
        assert rec.model == "gpt-4o-mini"
        assert rec.input_tokens == 1_000
        assert rec.output_tokens == 500
        assert rec.cost_usd == pytest.approx(0.00045)
        assert rec.label is None

    def test_record_label_stored(self) -> None:
        """label kwarg is preserved on the returned UsageRecord."""
        t = CostTracker()
        rec = t.record("gpt-4o-mini", 100, 50, label="summarise-step")
        assert rec.label == "summarise-step"


class TestCostTrackerAccumulators:
    """total_cost_usd, total_tokens, by_model, reset."""

    def test_total_cost_usd_empty(self) -> None:
        """Empty tracker has zero total cost."""
        assert CostTracker().total_cost_usd == pytest.approx(0.0)

    def test_total_tokens_empty(self) -> None:
        """Empty tracker has zero total tokens."""
        assert CostTracker().total_tokens == 0

    def test_total_cost_usd_sums_across_records(self) -> None:
        """total_cost_usd is the sum of all UsageRecord costs.

        Two gpt-4.1-mini calls → 0.00052 + 0.00028 = 0.00080
        """
        t = _tracker_with_two_calls()
        assert t.total_cost_usd == pytest.approx(0.00080)

    def test_total_tokens_sums_input_and_output(self) -> None:
        """total_tokens sums input + output across all records.

        (500+200) + (300+100) = 700 + 400 = 1100
        """
        t = _tracker_with_two_calls()
        assert t.total_tokens == 1_100

    def test_by_model_groups_same_model(self) -> None:
        """by_model returns a single key when all records share one model."""
        t = _tracker_with_two_calls()
        summary = t.by_model()
        assert list(summary.keys()) == ["gpt-4.1-mini"]
        entry = summary["gpt-4.1-mini"]
        assert entry["calls"] == pytest.approx(2)
        assert entry["tokens"] == pytest.approx(1_100)
        assert entry["cost_usd"] == pytest.approx(0.00080)

    def test_by_model_groups_multiple_models(self) -> None:
        """by_model returns one entry per distinct model.

        claude-sonnet-4-6: 1000 in + 500 out = 1500 tok, $0.0105
        gpt-4o:            2000 in + 1000 out = 3000 tok, $0.015
        """
        t = CostTracker()
        t.record("claude-sonnet-4-6", 1_000, 500)
        t.record("gpt-4o", 2_000, 1_000)
        summary = t.by_model()
        assert set(summary.keys()) == {"claude-sonnet-4-6", "gpt-4o"}
        assert summary["claude-sonnet-4-6"]["tokens"] == pytest.approx(1_500)
        assert summary["claude-sonnet-4-6"]["cost_usd"] == pytest.approx(0.0105)
        assert summary["gpt-4o"]["tokens"] == pytest.approx(3_000)
        assert summary["gpt-4o"]["cost_usd"] == pytest.approx(0.015)

    def test_reset_clears_records(self) -> None:
        """reset() empties the tracker; subsequent totals are zero."""
        t = _tracker_with_two_calls()
        t.reset()
        assert t.total_cost_usd == pytest.approx(0.0)
        assert t.total_tokens == 0
        assert t.records == []

    def test_reset_allows_fresh_accumulation(self) -> None:
        """After reset, new records accumulate from zero."""
        t = _tracker_with_two_calls()
        t.reset()
        t.record("gpt-4o-mini", 100, 50)
        # gpt-4o-mini: 100/1e6*0.15 + 50/1e6*0.60 = 0.000015 + 0.00003 = 0.000045
        assert t.total_cost_usd == pytest.approx(0.000045)


# ---------------------------------------------------------------------------
# assert_cost_within
# ---------------------------------------------------------------------------

class TestAssertCostWithin:
    """assert_cost_within — CostTracker and float paths, CRITICAL + WARNING."""

    def test_passes_under_budget_with_tracker(self) -> None:
        """PASS: total cost well below max_usd."""
        t = _tracker_with_two_calls()  # $0.00080
        result = assert_cost_within(t, 1.00)
        assert result.passed is True
        assert result.name == "cost.within_budget"

    def test_passes_at_exact_budget(self) -> None:
        """PASS: total cost equal to max_usd is within budget."""
        t = _tracker_with_two_calls()  # $0.00080
        result = assert_cost_within(t, 0.00080)
        assert result.passed is True

    def test_raises_over_budget_critical(self) -> None:
        """FAIL + raise: total cost exceeds max_usd at CRITICAL severity."""
        t = _tracker_with_two_calls()  # $0.00080
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_cost_within(t, 0.0005)
        assert exc_info.value.result.passed is False
        assert exc_info.value.result.name == "cost.within_budget"

    def test_float_input_passes(self) -> None:
        """PASS: float USD amount below max_usd."""
        result = assert_cost_within(0.50, 1.00)
        assert result.passed is True

    def test_float_input_raises_over_budget(self) -> None:
        """FAIL + raise: float USD amount above max_usd."""
        with pytest.raises(MltkAssertionError):
            assert_cost_within(1.50, 1.00)

    def test_warning_severity_does_not_raise(self) -> None:
        """WARNING severity: returns result with passed=False, does NOT raise."""
        t = _tracker_with_two_calls()  # $0.00080
        result = assert_cost_within(t, 0.0001, severity=Severity.WARNING)
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_details_include_cost_and_max(self) -> None:
        """TestResult.details carry total_cost_usd and max_usd."""
        t = _tracker_with_two_calls()
        result = assert_cost_within(t, 1.00)
        assert "total_cost_usd" in result.details
        assert "max_usd" in result.details
        assert result.details["max_usd"] == pytest.approx(1.00)

    def test_duration_ms_populated(self) -> None:
        """duration_ms is set (>= 0) by @timed_assertion."""
        t = _tracker_with_two_calls()
        result = assert_cost_within(t, 1.00)
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# assert_token_usage
# ---------------------------------------------------------------------------

class TestAssertTokenUsage:
    """assert_token_usage — CostTracker and int paths."""

    def test_passes_under_limit_with_tracker(self) -> None:
        """PASS: total tokens well below max_tokens."""
        t = _tracker_with_two_calls()  # 1100 tokens
        result = assert_token_usage(t, 10_000)
        assert result.passed is True
        assert result.name == "cost.token_usage"

    def test_passes_at_exact_limit(self) -> None:
        """PASS: total tokens equal to max_tokens."""
        t = _tracker_with_two_calls()  # 1100 tokens
        result = assert_token_usage(t, 1_100)
        assert result.passed is True

    def test_raises_over_limit_critical(self) -> None:
        """FAIL + raise: total tokens exceed max_tokens at CRITICAL severity."""
        t = _tracker_with_two_calls()  # 1100 tokens
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_token_usage(t, 1_000)
        assert exc_info.value.result.passed is False
        assert exc_info.value.result.name == "cost.token_usage"

    def test_int_input_passes(self) -> None:
        """PASS: raw int token count below max_tokens."""
        result = assert_token_usage(500, 1_000)
        assert result.passed is True

    def test_int_input_raises_over_limit(self) -> None:
        """FAIL + raise: raw int token count above max_tokens."""
        with pytest.raises(MltkAssertionError):
            assert_token_usage(2_000, 1_000)

    def test_warning_severity_does_not_raise(self) -> None:
        """WARNING severity: returns result with passed=False, does NOT raise."""
        t = _tracker_with_two_calls()  # 1100 tokens
        result = assert_token_usage(t, 100, severity=Severity.WARNING)
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_details_include_tokens_and_max(self) -> None:
        """TestResult.details carry total_tokens and max_tokens."""
        t = _tracker_with_two_calls()
        result = assert_token_usage(t, 10_000)
        assert "total_tokens" in result.details
        assert "max_tokens" in result.details
        assert result.details["max_tokens"] == 10_000

    def test_duration_ms_populated(self) -> None:
        """duration_ms is set (>= 0) by @timed_assertion."""
        t = _tracker_with_two_calls()
        result = assert_token_usage(t, 10_000)
        assert result.duration_ms >= 0.0


def test_record_unknown_model_raises_and_leaves_records_empty() -> None:
    """record() propagates ValueError for an unknown model and adds no record."""
    tracker = CostTracker()
    with pytest.raises(ValueError, match="register_pricing"):
        tracker.record("ghost-model-9000", 100, 50)
    assert tracker.records == []
