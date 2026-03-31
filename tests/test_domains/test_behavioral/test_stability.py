"""Tests for output stability assertion.

Covers ``assert_output_stability`` from the behavioral consistency
module.  All model calls use deterministic mocks; no external
dependencies required.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.behavioral import (
    assert_output_stability,
)


# -- Shared helpers --------------------------------------------------

SEED = 42


def _result_shape_ok(result: TestResult) -> None:
    """Assert TestResult has the mandatory fields."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# -- Mock model functions --------------------------------------------

_RNG_COUNTER = 0


def deterministic_model(text: str) -> str:
    """Always returns the same output for the same input."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"answer-{h}"


def random_model(text: str) -> str:
    """Returns a different output on every call."""
    global _RNG_COUNTER  # noqa: PLW0603
    _RNG_COUNTER += 1
    return f"answer-{_RNG_COUNTER}"


def mostly_stable_model(text: str) -> str:
    """Stable 80% of the time, random 20%."""
    global _RNG_COUNTER  # noqa: PLW0603
    _RNG_COUNTER += 1
    if _RNG_COUNTER % 5 == 0:
        return f"rare-variant-{_RNG_COUNTER}"
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"answer-{h}"


def classifier_stable(text: str) -> str:
    """Deterministic classifier label."""
    return "positive"


_ERR_COUNT = 0


def error_model(text: str) -> str:
    """Raises on alternating calls."""
    global _ERR_COUNT  # noqa: PLW0603
    _ERR_COUNT += 1
    if _ERR_COUNT % 2 == 0:
        raise RuntimeError("model crashed")
    return "stable answer"


# -- Inputs -----------------------------------------------------------

SINGLE_INPUT = ["What was the major 20th century war?"]
MULTI_INPUTS = [
    "What was the major 20th century war?",
    "Explain gravity briefly.",
    "Name the largest ocean.",
]


# ===================================================================
# TestOutputStability
# ===================================================================


class TestOutputStability:
    """Tests for ``assert_output_stability``."""

    def test_deterministic_model_passes(self) -> None:
        """Same output every run gives stability=1.0."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=5,
            equivalence_method="token_f1",
            min_stability=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert (
            result.name
            == "llm.behavioral.output_stability"
        )

    def test_random_model_fails(self) -> None:
        """Different output each run gives low stability."""
        global _RNG_COUNTER  # noqa: PLW0603
        _RNG_COUNTER = 0
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_stability(
                model_fn=random_model,
                inputs=SINGLE_INPUT,
                n_runs=5,
                equivalence_method="token_f1",
                min_stability=0.8,
            )
        r = exc.value.result
        assert r.passed is False
        _result_shape_ok(r)

    def test_min_stability_threshold_low(self) -> None:
        """Low threshold allows some instability."""
        global _RNG_COUNTER  # noqa: PLW0603
        _RNG_COUNTER = 0
        result = assert_output_stability(
            model_fn=mostly_stable_model,
            inputs=SINGLE_INPUT,
            n_runs=5,
            equivalence_method="token_f1",
            min_stability=0.3,
        )
        assert result.passed is True

    def test_min_stability_threshold_high(self) -> None:
        """High threshold catches partial instability."""
        global _RNG_COUNTER  # noqa: PLW0603
        _RNG_COUNTER = 0
        with pytest.raises(MltkAssertionError):
            assert_output_stability(
                model_fn=mostly_stable_model,
                inputs=SINGLE_INPUT,
                n_runs=10,
                equivalence_method="token_f1",
                min_stability=1.0,
            )

    def test_n_runs_parameter(self) -> None:
        """Model is called exactly n_runs times per input."""
        call_count = 0

        def counting_model(text: str) -> str:
            nonlocal call_count
            call_count += 1
            return "fixed"

        n = 7
        assert_output_stability(
            model_fn=counting_model,
            inputs=SINGLE_INPUT,
            n_runs=n,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        assert call_count == n * len(SINGLE_INPUT)

    def test_n_runs_less_than_2_fails(self) -> None:
        """n_runs < 2 is invalid; cannot measure stability."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_output_stability(
                model_fn=deterministic_model,
                inputs=SINGLE_INPUT,
                n_runs=1,
                equivalence_method="token_f1",
                min_stability=0.8,
            )

    def test_single_input(self) -> None:
        """Single input with deterministic model passes."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.8,
        )
        assert result.passed is True

    def test_multiple_inputs(self) -> None:
        """Multiple inputs all evaluated for stability."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=MULTI_INPUTS,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_per_input_stability_in_details(self) -> None:
        """Details include per-input stability breakdown."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=MULTI_INPUTS,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        details = result.details
        assert (
            "per_input" in details
            or "per_input_stability" in details
            or "input_scores" in details
        )

    def test_worst_input_in_details(self) -> None:
        """Details include worst-input info."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=MULTI_INPUTS,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        details = result.details
        assert (
            "worst_input" in details
            or "worst_score" in details
            or "min_score" in details
        )

    def test_method_token_f1(self) -> None:
        """token_f1 method works for stability checks."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.8,
        )
        assert result.details["method"] == "token_f1"

    def test_method_label_match(self) -> None:
        """label_match method for classifier stability."""
        result = assert_output_stability(
            model_fn=classifier_stable,
            inputs=SINGLE_INPUT,
            n_runs=5,
            equivalence_method="label_match",
            min_stability=1.0,
        )
        assert result.passed is True
        assert result.details["method"] == "label_match"

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_pairs"
    )
    def test_method_embedding_mock(
        self, mock_embed: MagicMock,
    ) -> None:
        """embedding method delegates to backend."""
        mock_embed.return_value = [0.99, 0.98, 0.97]
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="embedding",
            min_stability=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "embedding"
        mock_embed.assert_called()

    def test_model_exception_counted_as_instability(
        self,
    ) -> None:
        """Model exceptions count as different outputs."""
        global _ERR_COUNT  # noqa: PLW0603
        _ERR_COUNT = 0
        with pytest.raises(MltkAssertionError):
            assert_output_stability(
                model_fn=error_model,
                inputs=SINGLE_INPUT,
                n_runs=6,
                equivalence_method="token_f1",
                min_stability=1.0,
            )

    def test_empty_inputs_fails(self) -> None:
        """Empty inputs list produces a failing result."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_output_stability(
                model_fn=deterministic_model,
                inputs=[],
                n_runs=3,
                equivalence_method="token_f1",
                min_stability=0.8,
            )

    def test_n_unique_outputs_in_details(self) -> None:
        """Details include count of unique outputs per input."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=5,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        details = result.details
        assert "per_input_stability" in details
        per_input = details["per_input_stability"]
        assert len(per_input) > 0
        assert "n_unique_outputs" in per_input[0]

    def test_method_in_details(self) -> None:
        """Details include the method used."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        assert "method" in result.details

    def test_severity_on_failure(self) -> None:
        """Failed stability has CRITICAL severity."""
        global _RNG_COUNTER  # noqa: PLW0603
        _RNG_COUNTER = 0
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_stability(
                model_fn=random_model,
                inputs=SINGLE_INPUT,
                n_runs=5,
                equivalence_method="token_f1",
                min_stability=0.9,
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    def test_unknown_method_fails(self) -> None:
        """Unknown method produces a failing result."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_stability(
                model_fn=deterministic_model,
                inputs=SINGLE_INPUT,
                n_runs=3,
                equivalence_method="fake_method",
                min_stability=0.5,
            )
        msg = exc.value.result.message
        assert "fake_method" in msg

    def test_stability_score_in_details(self) -> None:
        """Details include the computed stability score."""
        result = assert_output_stability(
            model_fn=deterministic_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.5,
        )
        details = result.details
        assert "avg_stability" in details

    # -- New edge-case and parametrized tests --------------------

    def test_stability_single_run(self) -> None:
        """N=1 is invalid; must raise or fail gracefully."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_output_stability(
                model_fn=deterministic_model,
                inputs=SINGLE_INPUT,
                n_runs=1,
                equivalence_method="token_f1",
                min_stability=0.8,
            )

    def test_stability_all_identical(self) -> None:
        """All N runs produce same output => pass."""
        result = assert_output_stability(
            model_fn=lambda t: "fixed answer",
            inputs=["prompt A", "prompt B"],
            n_runs=5,
            equivalence_method="label_match",
            min_stability=1.0,
        )
        assert result.passed is True
        assert result.details["avg_stability"] == 1.0

    def test_stability_all_different(self) -> None:
        """All N runs produce unique output => fail."""
        counter = {"v": 0}

        def unique_model(text: str) -> str:
            counter["v"] += 1
            return f"variant-{counter['v']}"

        with pytest.raises(MltkAssertionError) as exc:
            assert_output_stability(
                model_fn=unique_model,
                inputs=SINGLE_INPUT,
                n_runs=5,
                equivalence_method="label_match",
                min_stability=0.5,
            )
        assert exc.value.result.passed is False

    def test_stability_unicode_responses(self) -> None:
        """Japanese/emoji responses do not crash."""
        responses = [
            "\u3053\u3093\u306b\u3061\u306f",
            "\U0001f600\U0001f389",
            "\u4f60\u597d\u4e16\u754c",
        ]
        idx = {"i": 0}

        def unicode_model(text: str) -> str:
            idx["i"] = (idx["i"] + 1) % len(responses)
            return responses[idx["i"]]

        try:
            assert_output_stability(
                model_fn=unicode_model,
                inputs=["hello"],
                n_runs=3,
                equivalence_method="token_f1",
                min_stability=0.1,
            )
        except MltkAssertionError:
            pass  # may fail on stability, must not crash

    def test_stability_empty_responses(self) -> None:
        """Model returns empty string every time."""
        result = assert_output_stability(
            model_fn=lambda t: "",
            inputs=SINGLE_INPUT,
            n_runs=4,
            equivalence_method="label_match",
            min_stability=0.5,
        )
        # All outputs identical ("") => stability 1.0
        assert result.passed is True

    def test_stability_threshold_boundary(self) -> None:
        """Exactly at threshold boundary => passes."""
        # label_match with 2 of 3 pairs matching
        # yields stability = 2/3 ~ 0.6667
        call_counter = {"n": 0}

        def two_thirds_model(text: str) -> str:
            call_counter["n"] += 1
            if call_counter["n"] % 3 == 0:
                return "variant"
            return "stable"

        # 3 runs => pairs: (1,2) match, (1,3) no,
        # (2,3) no => 1/3 = 0.3333
        # Use min_stability just at or below actual
        result = assert_output_stability(
            model_fn=two_thirds_model,
            inputs=SINGLE_INPUT,
            n_runs=3,
            equivalence_method="label_match",
            min_stability=0.33,
        )
        assert result.passed is True

    def test_stability_large_n(self) -> None:
        """N=20 runs with deterministic model passes."""
        call_count = {"n": 0}

        def counting_model(text: str) -> str:
            call_count["n"] += 1
            return "consistent output"

        result = assert_output_stability(
            model_fn=counting_model,
            inputs=SINGLE_INPUT,
            n_runs=20,
            equivalence_method="label_match",
            min_stability=1.0,
        )
        assert result.passed is True
        assert call_count["n"] == 20

    def test_stability_custom_metric(self) -> None:
        """token_f1 metric with custom threshold."""
        # Two tokens overlap partially
        cycle = {"i": 0}

        def partial_model(text: str) -> str:
            cycle["i"] += 1
            if cycle["i"] % 2 == 0:
                return "the quick brown fox"
            return "the quick red fox"

        # token_f1 between those two strings > 0.5
        result = assert_output_stability(
            model_fn=partial_model,
            inputs=SINGLE_INPUT,
            n_runs=4,
            equivalence_method="token_f1",
            min_stability=0.3,
            similarity_threshold=0.5,
        )
        assert result.passed is True
