"""Tests for LLM-as-Judge default configuration and convenience wrappers."""

from __future__ import annotations

import threading

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.judge_defaults import (
    assert_with_judge,
    configure_default_judge,
    get_default_judge,
    resolve_judge,
)

# -------------------------------------------------------------------
# Fixtures -- ensure clean state between tests
# -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_default_judge():
    """Reset the module-level default judge before and after each test."""
    configure_default_judge(None)
    yield
    configure_default_judge(None)


# -------------------------------------------------------------------
# Mock judge functions
# -------------------------------------------------------------------

def _echo_judge(prompt: str) -> str:
    """Returns the prompt back (for testing wiring)."""
    return prompt


def _score_high(prompt: str) -> str:
    """Judge that always returns a high score string."""
    return "0.95"


def _score_low(prompt: str) -> str:
    """Judge that always returns a low score string."""
    return "0.1"


def _two_arg_scorer(text_a: str, text_b: str) -> float:
    """Two-arg judge that returns high score."""
    return 0.9


def _two_arg_scorer_low(text_a: str, text_b: str) -> float:
    """Two-arg judge that returns low score."""
    return 0.1


def _two_arg_error(text_a: str, text_b: str) -> float:
    """Two-arg judge that raises an error."""
    raise RuntimeError("Model unavailable")


def _alternative_judge(prompt: str) -> str:
    """Alternative judge returning a different score."""
    return "0.75"


# ===================================================================
# configure_default_judge / get_default_judge round-trip
# ===================================================================

class TestConfigureGetRoundTrip:
    """Tests for configure/get default judge."""

    def test_default_is_none_initially(self) -> None:
        """No default judge is set at startup."""
        assert get_default_judge() is None

    def test_set_and_get(self) -> None:
        """Set a judge, get it back."""
        configure_default_judge(_echo_judge)
        assert get_default_judge() is _echo_judge

    def test_set_to_none_clears(self) -> None:
        """Setting None clears the default."""
        configure_default_judge(_echo_judge)
        configure_default_judge(None)
        assert get_default_judge() is None

    def test_replace_default(self) -> None:
        """Replacing the default replaces the old one."""
        configure_default_judge(_echo_judge)
        configure_default_judge(_alternative_judge)
        assert get_default_judge() is _alternative_judge

    def test_set_lambda(self) -> None:
        """Lambda judge functions work."""
        fn = lambda p: "0.5"  # noqa: E731
        configure_default_judge(fn)
        assert get_default_judge() is fn


# ===================================================================
# resolve_judge
# ===================================================================

class TestResolveJudge:
    """Tests for the resolve_judge helper."""

    def test_explicit_judge_used(self) -> None:
        """Explicit judge takes priority over default."""
        configure_default_judge(_echo_judge)
        judge, method = resolve_judge(
            explicit_judge=_alternative_judge,
            method="auto",
        )
        assert judge is _alternative_judge
        assert method == "llm"

    def test_explicit_judge_preserves_method(self) -> None:
        """Explicit judge with non-auto method keeps method."""
        judge, method = resolve_judge(
            explicit_judge=_echo_judge,
            method="nli",
        )
        assert judge is _echo_judge
        assert method == "nli"

    def test_auto_uses_default_when_set(self) -> None:
        """Auto method picks up the module default."""
        configure_default_judge(_score_high)
        judge, method = resolve_judge(method="auto")
        assert judge is _score_high
        assert method == "llm"

    def test_auto_falls_back_to_lexical(self) -> None:
        """Auto with no default falls back to lexical."""
        judge, method = resolve_judge(method="auto")
        assert judge is None
        assert method == "lexical"

    def test_auto_falls_back_to_custom(self) -> None:
        """Auto with no default uses custom fallback."""
        judge, method = resolve_judge(
            method="auto",
            fallback_method="embedding",
        )
        assert judge is None
        assert method == "embedding"

    def test_explicit_method_no_judge(self) -> None:
        """Non-auto method without judge passes through."""
        judge, method = resolve_judge(method="embedding")
        assert judge is None
        assert method == "embedding"

    def test_explicit_none_judge_auto(self) -> None:
        """Passing explicit_judge=None with auto uses default."""
        configure_default_judge(_score_high)
        judge, method = resolve_judge(
            explicit_judge=None,
            method="auto",
        )
        assert judge is _score_high
        assert method == "llm"


# ===================================================================
# assert_with_judge -- using explicit judge
# ===================================================================

class TestAssertWithJudgeExplicit:
    """Tests for assert_with_judge with explicit judge_fn."""

    def test_explicit_judge_passes(self) -> None:
        """PASS: Explicit judge returns high score."""
        result = assert_with_judge(
            assertion_name="test_quality",
            text_a="The answer is correct.",
            text_b="The reference confirms this.",
            judge_fn=_two_arg_scorer,
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["score"] == 0.9
        assert result.details["method"] == "llm"

    def test_explicit_judge_fails(self) -> None:
        """FAIL: Explicit judge returns low score."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="test_quality",
                text_a="Bad answer",
                text_b="Good reference",
                judge_fn=_two_arg_scorer_low,
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["score"] == 0.1

    def test_explicit_judge_overrides_default(self) -> None:
        """Explicit judge used even when default is set."""
        configure_default_judge(_score_low)
        result = assert_with_judge(
            assertion_name="override_test",
            text_a="text a",
            text_b="text b",
            judge_fn=_two_arg_scorer,
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["score"] == 0.9


# ===================================================================
# assert_with_judge -- using default judge
# ===================================================================

class TestAssertWithJudgeDefault:
    """Tests for assert_with_judge using module default."""

    def test_default_judge_used(self) -> None:
        """Default judge is invoked when no explicit one."""
        configure_default_judge(_score_high)
        result = assert_with_judge(
            assertion_name="default_test",
            text_a="answer text",
            text_b="reference text",
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "llm"


# ===================================================================
# assert_with_judge -- lexical fallback
# ===================================================================

class TestAssertWithJudgeFallback:
    """Tests for assert_with_judge lexical fallback."""

    def test_no_judge_falls_back_to_lexical(self) -> None:
        """Lexical fallback when no judge available."""
        result = assert_with_judge(
            assertion_name="lexical_test",
            text_a="the quick brown fox",
            text_b="the quick brown fox jumps",
            min_score=0.3,
        )
        assert result.passed is True
        assert result.details["method"] == "lexical"

    def test_lexical_low_overlap_fails(self) -> None:
        """FAIL: Lexical fallback with low overlap."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="overlap_test",
                text_a="alpha beta gamma",
                text_b="delta epsilon zeta",
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["method"] == "lexical"

    def test_lexical_identical_texts(self) -> None:
        """PASS: Identical texts get score 1.0."""
        result = assert_with_judge(
            assertion_name="identical_test",
            text_a="hello world",
            text_b="hello world",
            min_score=0.9,
        )
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_unsupported_fallback_method_fails(self) -> None:
        """FAIL: Non-lexical fallback without judge errors."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="bad_fallback",
                text_a="text",
                text_b="text",
                fallback_method="embedding",
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert "No judge available" in result.message


# ===================================================================
# assert_with_judge -- error handling
# ===================================================================

class TestAssertWithJudgeErrors:
    """Tests for assert_with_judge error handling."""

    def test_judge_exception_handled(self) -> None:
        """FAIL: Judge raising exception is caught gracefully."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="error_test",
                text_a="input",
                text_b="reference",
                judge_fn=_two_arg_error,
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert "Judge error" in result.message
        assert "RuntimeError" in result.message


# ===================================================================
# assert_with_judge -- result metadata
# ===================================================================

class TestAssertWithJudgeMetadata:
    """Tests for result metadata from assert_with_judge."""

    def test_result_name_includes_assertion(self) -> None:
        """Result name follows llm.judge_defaults.{name}."""
        result = assert_with_judge(
            assertion_name="my_check",
            text_a="a",
            text_b="a",
            min_score=0.1,
        )
        assert result.name == "llm.judge_defaults.my_check"

    def test_has_duration_ms(self) -> None:
        """timed_assertion decorator populates duration_ms."""
        result = assert_with_judge(
            assertion_name="timed",
            text_a="some text",
            text_b="some text",
            min_score=0.1,
        )
        assert result.duration_ms >= 0.0

    def test_details_contain_score_and_method(self) -> None:
        """Details always include score, min_score, method."""
        result = assert_with_judge(
            assertion_name="details_check",
            text_a="hello world",
            text_b="hello world",
            min_score=0.1,
        )
        assert "score" in result.details
        assert "min_score" in result.details
        assert "method" in result.details

    def test_exact_threshold_passes(self) -> None:
        """PASS: Score exactly at min_score passes."""
        def exact_scorer(a: str, b: str) -> float:
            return 0.5

        result = assert_with_judge(
            assertion_name="exact",
            text_a="a",
            text_b="b",
            judge_fn=exact_scorer,
            min_score=0.5,
        )
        assert result.passed is True


# ===================================================================
# Thread safety
# ===================================================================

class TestThreadSafety:
    """Basic thread-safety tests for default judge config."""

    def test_concurrent_set_get(self) -> None:
        """Multiple threads setting/getting do not crash."""
        errors: list[Exception] = []

        def setter(fn: object) -> None:
            try:
                for _ in range(100):
                    configure_default_judge(fn)
            except Exception as exc:
                errors.append(exc)

        def getter() -> None:
            try:
                for _ in range(100):
                    get_default_judge()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=setter, args=(_echo_judge,)),
            threading.Thread(target=setter, args=(_score_high,)),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0

    def test_concurrent_resolve(self) -> None:
        """resolve_judge is safe under concurrent access."""
        configure_default_judge(_score_high)
        errors: list[Exception] = []
        results: list[tuple] = []

        def resolver() -> None:
            try:
                for _ in range(50):
                    result = resolve_judge(method="auto")
                    results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=resolver)
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(results) == 200
        for _judge, method in results:
            assert method == "llm"

    def test_set_none_during_reads(self) -> None:
        """Setting None while reads happen is safe."""
        configure_default_judge(_echo_judge)
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = get_default_judge()
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            try:
                for _ in range(100):
                    configure_default_judge(None)
                    configure_default_judge(_echo_judge)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_empty_text_a_lexical(self) -> None:
        """Empty text_a with lexical gives score 1.0."""
        result = assert_with_judge(
            assertion_name="empty_a",
            text_a="",
            text_b="some reference text",
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_empty_text_b_lexical(self) -> None:
        """Empty text_b with lexical gives score 0.0."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="empty_b",
                text_a="some text",
                text_b="",
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["score"] == 0.0

    def test_both_empty_lexical(self) -> None:
        """Both texts empty with lexical gives score 1.0."""
        result = assert_with_judge(
            assertion_name="both_empty",
            text_a="",
            text_b="",
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["score"] == 1.0


# ===================================================================
# Hardening edge-case tests (appended)
# ===================================================================


class TestConfigureDefaultJudgeEdgeCases:
    """Edge cases for configure/get default judge."""

    def test_set_none_clears_default(self) -> None:
        """configure_default_judge(None) clears judge."""
        configure_default_judge(_echo_judge)
        assert get_default_judge() is _echo_judge
        configure_default_judge(None)
        assert get_default_judge() is None

    def test_second_call_replaces_first(self) -> None:
        """Second configure replaces the first judge."""
        configure_default_judge(_echo_judge)
        configure_default_judge(_score_high)
        assert get_default_judge() is _score_high

    def test_get_default_returns_none_initially(self) -> None:
        """No default judge is set at startup."""
        assert get_default_judge() is None


class TestResolveJudgeEdgeCases:
    """Edge cases for resolve_judge."""

    def test_all_three_levels_explicit_wins(self) -> None:
        """Explicit judge wins over default and fallback."""
        configure_default_judge(_score_high)
        judge, method = resolve_judge(
            explicit_judge=_alternative_judge,
            method="auto",
            fallback_method="embedding",
        )
        assert judge is _alternative_judge
        assert method == "llm"

    def test_only_fallback_set(self) -> None:
        """No explicit or default, fallback method used."""
        judge, method = resolve_judge(
            method="auto",
            fallback_method="embedding",
        )
        assert judge is None
        assert method == "embedding"


class TestAssertWithJudgeEdgeCases:
    """Edge cases for assert_with_judge."""

    def test_judge_raises_exception(self) -> None:
        """Judge that raises produces failing result."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_with_judge(
                assertion_name="crash_test",
                text_a="input",
                text_b="reference",
                judge_fn=_two_arg_error,
                min_score=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert "RuntimeError" in result.message


class TestThreadSafetyEdgeCases:
    """Concurrent configure + resolve edge case."""

    def test_concurrent_configure_and_resolve(self) -> None:
        """Concurrent configure + resolve is safe."""
        errors: list[Exception] = []
        results: list[tuple] = []

        def configurer() -> None:
            try:
                for _ in range(50):
                    configure_default_judge(_echo_judge)
                    configure_default_judge(_score_high)
            except Exception as exc:
                errors.append(exc)

        def resolver() -> None:
            try:
                for _ in range(50):
                    r = resolve_judge(method="auto")
                    results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=configurer),
            threading.Thread(target=resolver),
            threading.Thread(target=resolver),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        for _judge, method in results:
            assert method in ("llm", "lexical")
