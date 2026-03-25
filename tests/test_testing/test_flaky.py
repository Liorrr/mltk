"""Tests for mltk.testing.flaky — flaky test detection."""
from __future__ import annotations

import pytest

from mltk.testing.flaky import FlakySummary, detect_flaky

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_pass() -> None:
    pass


def _always_fail() -> None:
    raise AssertionError("always fails")


class _SometimesFailCounter:
    """Fails on every N-th call so we have deterministic control."""

    def __init__(self, fail_every: int) -> None:
        self._fail_every = fail_every
        self._calls = 0

    def __call__(self) -> None:
        self._calls += 1
        if self._calls % self._fail_every == 0:
            raise AssertionError("periodic failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_stable_pass():
    # SCENARIO: function that never raises
    # WHY: an always-passing test is stable, not flaky
    # EXPECTED: is_flaky=False, pass_rate=1.0, fail_count=0
    summary = detect_flaky(_always_pass, runs=5)

    assert isinstance(summary, FlakySummary)
    assert summary.pass_count == 5
    assert summary.fail_count == 0
    assert summary.pass_rate == pytest.approx(1.0)
    assert summary.is_flaky is False


def test_detect_stable_fail():
    # SCENARIO: function that always raises
    # WHY: an always-failing test is broken, not flaky
    # EXPECTED: is_flaky=False, pass_rate=0.0, pass_count=0
    summary = detect_flaky(_always_fail, runs=5)

    assert summary.pass_count == 0
    assert summary.fail_count == 5
    assert summary.pass_rate == pytest.approx(0.0)
    assert summary.is_flaky is False


def test_detect_flaky():
    # SCENARIO: function that fails on every 2nd call (50% pass rate)
    # WHY: 0 < pass_rate < threshold(0.8) => flaky
    # EXPECTED: is_flaky=True, pass_rate == 0.5
    counter = _SometimesFailCounter(fail_every=2)
    summary = detect_flaky(counter, runs=10, threshold=0.8)

    assert summary.pass_count == 5
    assert summary.fail_count == 5
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.is_flaky is True


def test_custom_threshold():
    # SCENARIO: pass_rate=0.5 with threshold=0.4
    # WHY: pass_rate(0.5) >= threshold(0.4) so is_flaky should be False
    # EXPECTED: is_flaky=False (passes the threshold)
    counter = _SometimesFailCounter(fail_every=2)
    # runs=10 → 5 pass / 5 fail → pass_rate=0.5
    summary = detect_flaky(counter, runs=10, threshold=0.4)

    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.is_flaky is False


def test_summary_fields():
    # SCENARIO: verify every FlakySummary field is populated
    # WHY: callers depend on all fields being present and typed correctly
    # EXPECTED: all fields have expected types and consistent values
    counter = _SometimesFailCounter(fail_every=3)
    summary = detect_flaky(counter, runs=9, test_name="my_test")

    assert summary.test_name == "my_test"
    assert isinstance(summary.pass_count, int)
    assert isinstance(summary.fail_count, int)
    assert isinstance(summary.pass_rate, float)
    assert isinstance(summary.is_flaky, bool)
    assert summary.pass_count + summary.fail_count == 9
    assert 0.0 <= summary.pass_rate <= 1.0


def test_custom_test_name():
    # SCENARIO: test_name kwarg overrides func.__name__
    # WHY: callers want to label tests independently from the callable name
    # EXPECTED: summary.test_name matches the supplied name
    summary = detect_flaky(_always_pass, runs=3, test_name="custom_label")

    assert summary.test_name == "custom_label"


def test_single_run():
    # SCENARIO: runs=1, function passes
    # WHY: edge case — minimal run count should still produce valid summary
    # EXPECTED: pass_rate=1.0, is_flaky=False
    summary = detect_flaky(_always_pass, runs=1)

    assert summary.pass_count == 1
    assert summary.pass_rate == pytest.approx(1.0)
    assert summary.is_flaky is False
