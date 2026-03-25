"""Tests for mltk.domains.llm.agentic — agentic evaluation assertions."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.agentic import (
    assert_task_completion,
    assert_tool_call_correctness,
    assert_tool_selection,
)


class TestTaskCompletion:
    """Task completion — token overlap between expected and actual agent output."""

    def test_task_completion_pass(self) -> None:
        # SCENARIO: Agent output closely matches expected output, sharing most tokens.
        # WHY: Jaccard overlap of highly similar texts is well above 0.7.
        # EXPECTED: passes with score >= 0.7.
        expected = "Sorted list: 1 2 3 4 5"
        actual = "The sorted list is 1 2 3 4 5."
        result = assert_task_completion(expected, actual, min_score=0.5)
        assert result.passed is True
        assert result.details["score"] >= 0.5

    def test_task_completion_fail(self) -> None:
        # SCENARIO: Agent output is completely unrelated to what was expected.
        # WHY: Two texts with no token overlap have Jaccard score of 0.
        # EXPECTED: raises MltkAssertionError with score < 0.7.
        expected = "Sorted list: 1 2 3 4 5"
        actual = "Jupiter has large storms visible from Earth."
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_task_completion(expected, actual, min_score=0.7)
        assert exc_info.value.result.details["score"] < 0.7

    def test_task_completion_identical(self) -> None:
        # SCENARIO: Expected and actual outputs are identical strings.
        # WHY: Jaccard similarity of identical strings = 1.0.
        # EXPECTED: passes with score = 1.0.
        text = "calculate the sum of the first ten prime numbers"
        result = assert_task_completion(text, text, min_score=0.9)
        assert result.passed is True
        assert abs(result.details["score"] - 1.0) < 1e-9

    def test_task_completion_has_timing(self) -> None:
        # SCENARIO: Assertion is wrapped with @timed_assertion.
        # WHY: duration_ms must be populated for performance tracking in all assertions.
        # EXPECTED: duration_ms > 0.
        result = assert_task_completion("hello world", "hello world", min_score=0.5)
        assert result.duration_ms > 0


class TestToolSelection:
    """Tool selection — correct tools called, no missing or extra tools."""

    def test_tool_selection_correct(self) -> None:
        # SCENARIO: Agent called exactly the expected tools, nothing more, nothing less.
        # WHY: Perfect tool selection — all expected present, no unexpected tools.
        # EXPECTED: passes with precision=1.0, recall=1.0, no missing/extra tools.
        expected = ["search", "calculator"]
        actual = ["search", "calculator"]
        result = assert_tool_selection(expected, actual)
        assert result.passed is True
        assert abs(result.details["precision"] - 1.0) < 1e-9
        assert abs(result.details["recall"] - 1.0) < 1e-9
        assert result.details["missing_tools"] == []
        assert result.details["extra_tools"] == []

    def test_tool_selection_missing(self) -> None:
        # SCENARIO: Agent forgot to call the "calculator" tool.
        # WHY: Missing tool means the agent did not complete all required steps.
        # EXPECTED: raises MltkAssertionError; "calculator" appears in missing_tools.
        expected = ["search", "calculator"]
        actual = ["search"]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_selection(expected, actual)
        result = exc_info.value.result
        assert "calculator" in result.details["missing_tools"]
        assert result.details["extra_tools"] == []

    def test_tool_selection_extra(self) -> None:
        # SCENARIO: Agent called an unexpected "database_query" tool not in the plan.
        # WHY: Extra tool calls may indicate the agent is over-querying or hallucinating actions.
        # EXPECTED: raises MltkAssertionError; "database_query" appears in extra_tools.
        expected = ["search"]
        actual = ["search", "database_query"]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_selection(expected, actual)
        result = exc_info.value.result
        assert "database_query" in result.details["extra_tools"]
        assert result.details["missing_tools"] == []

    def test_tool_selection_both_missing_and_extra(self) -> None:
        # SCENARIO: Agent skipped one expected tool and called one unexpected tool.
        # WHY: Combined failure mode; both missing and extra should be reported.
        # EXPECTED: raises MltkAssertionError with both lists populated.
        expected = ["search", "calculator"]
        actual = ["search", "code_executor"]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_selection(expected, actual)
        result = exc_info.value.result
        assert "calculator" in result.details["missing_tools"]
        assert "code_executor" in result.details["extra_tools"]

    def test_tool_selection_empty_both(self) -> None:
        # SCENARIO: No tools expected and none called (pure reasoning task).
        # WHY: Empty expected and actual should be treated as correct selection.
        # EXPECTED: passes.
        result = assert_tool_selection([], [])
        assert result.passed is True


class TestToolCallCorrectness:
    """Tool call correctness — args match expected within tolerance."""

    def test_tool_call_correctness_pass(self) -> None:
        # SCENARIO: All args match — one numeric within tolerance, one string exact.
        # WHY: Numeric arg 0.705 is within tolerance=0.01 of 0.7; string matches exactly.
        # EXPECTED: passes with zero mismatches.
        expected = {"temperature": 0.7, "model": "gpt-4"}
        actual = {"temperature": 0.705, "model": "gpt-4"}
        result = assert_tool_call_correctness(expected, actual, tolerance=0.01)
        assert result.passed is True
        assert result.details["mismatch_count"] == 0

    def test_tool_call_correctness_numeric_out_of_tolerance(self) -> None:
        # SCENARIO: Numeric arg differs by 0.05, beyond the allowed tolerance of 0.01.
        # WHY: Tight numeric tolerance catches significant parameter drift in tool calls.
        # EXPECTED: raises MltkAssertionError; mismatch reported for "temperature".
        expected = {"temperature": 0.7, "model": "gpt-4"}
        actual = {"temperature": 0.75, "model": "gpt-4"}
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_call_correctness(expected, actual, tolerance=0.01)
        result = exc_info.value.result
        assert result.details["mismatch_count"] > 0
        assert any("temperature" in m for m in result.details["mismatches"])

    def test_tool_call_correctness_string_mismatch(self) -> None:
        # SCENARIO: String arg "gpt-3.5" differs from expected "gpt-4".
        # WHY: String args use exact match — a different model name is a hard failure.
        # EXPECTED: raises MltkAssertionError; mismatch reported for "model".
        expected = {"model": "gpt-4"}
        actual = {"model": "gpt-3.5"}
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_call_correctness(expected, actual, tolerance=0.01)
        result = exc_info.value.result
        assert any("model" in m for m in result.details["mismatches"])

    def test_tool_call_correctness_missing_key(self) -> None:
        # SCENARIO: Actual args are missing the "temperature" key entirely.
        # WHY: A tool called without a required argument is a correctness failure.
        # EXPECTED: raises MltkAssertionError; mismatch mentions missing key.
        expected = {"temperature": 0.7, "model": "gpt-4"}
        actual = {"model": "gpt-4"}
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_call_correctness(expected, actual, tolerance=0.01)
        result = exc_info.value.result
        assert any("temperature" in m for m in result.details["mismatches"])

    def test_tool_call_correctness_extra_key(self) -> None:
        # SCENARIO: Actual args include "stream=True" which was not expected.
        # WHY: Unexpected args can alter tool behavior in production systems.
        # EXPECTED: raises MltkAssertionError; mismatch mentions extra key "stream".
        expected = {"temperature": 0.7}
        actual = {"temperature": 0.7, "stream": True}
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_call_correctness(expected, actual, tolerance=0.01)
        result = exc_info.value.result
        assert any("stream" in m for m in result.details["mismatches"])

    def test_tool_call_correctness_all_exact(self) -> None:
        # SCENARIO: Multiple args of different types, all matching exactly.
        # WHY: Verifies correctness across mixed-type arg dicts.
        # EXPECTED: passes with zero mismatches.
        expected = {"endpoint": "https://api.example.com", "retries": 3, "timeout": 30.0}
        actual = {"endpoint": "https://api.example.com", "retries": 3, "timeout": 30.0}
        result = assert_tool_call_correctness(expected, actual, tolerance=0.001)
        assert result.passed is True
        assert result.details["mismatch_count"] == 0
