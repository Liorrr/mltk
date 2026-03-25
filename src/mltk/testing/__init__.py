"""Testing patterns — flaky detection, golden sets, retry, smart selection."""

from mltk.testing.flaky import FlakySummary, detect_flaky
from mltk.testing.golden import assert_matches_golden, load_golden, save_golden
from mltk.testing.retry import RetryResult, retry_until_confident
from mltk.testing.selection import build_test_map, select_affected_tests

__all__ = [
    # flaky
    "FlakySummary",
    "detect_flaky",
    # golden
    "save_golden",
    "load_golden",
    "assert_matches_golden",
    # retry
    "RetryResult",
    "retry_until_confident",
    # selection
    "build_test_map",
    "select_affected_tests",
]
