"""Tests for long-context LLM evaluation assertions."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.long_context import (
    assert_context_utilization,
    assert_needle_in_haystack,
    assert_no_lost_in_middle,
)


# ---------------------------------------------------------------------------
# Helper model functions
# ---------------------------------------------------------------------------


def _always_finds_needle(prompt: str) -> str:
    """Model that always returns the needle text."""
    return "The secret code is 7492"


def _never_finds_needle(prompt: str) -> str:
    """Model that returns irrelevant text."""
    return "I have no idea what you are asking about."


def _finds_edges_only(prompt: str) -> str:
    """Model that finds needle at positions 0.0 and 1.0 but not middle."""
    if "0.0" in prompt or "start" in prompt.lower():
        return "The secret code is 7492"
    if "1.0" in prompt or "end" in prompt.lower():
        return "The secret code is 7492"
    # For needle-in-haystack, the needle is embedded in the doc.
    # We detect position by checking where the needle appears:
    doc_start = prompt.find("Document:\n")
    if doc_start == -1:
        return "Unknown"
    doc = prompt[doc_start:]
    needle_pos = doc.find("The secret code is 7492")
    if needle_pos == -1:
        return "Unknown"
    # Needle near start or end of document?
    doc_len = len(doc)
    relative = needle_pos / doc_len if doc_len > 0 else 0.5
    if relative < 0.15 or relative > 0.85:
        return "The secret code is 7492"
    return "I cannot find that information in the document."


def _uses_all_facts(prompt: str) -> str:
    """Model that echoes back all facts."""
    return (
        "The capital of France is Paris. "
        "The Eiffel Tower is 330 meters tall. "
        "France has a population of 67 million. "
        "The official language is French. "
        "France uses the Euro currency."
    )


def _uses_no_facts(prompt: str) -> str:
    """Model that ignores all provided facts."""
    return "I cannot provide any information on this topic."


def _uses_one_fact(prompt: str) -> str:
    """Model that only uses the first fact."""
    return "The capital of France is Paris."


def _extract_question(prompt: str) -> str:
    """Extract the question portion from a structured prompt."""
    marker = "Question:"
    idx = prompt.find(marker)
    if idx == -1:
        return prompt.lower()
    return prompt[idx:].lower()


def _uniform_accuracy_model(prompt: str) -> str:
    """Model that answers correctly regardless of fact position."""
    q = _extract_question(prompt)
    if "speed of light" in q:
        return "The speed of light is 299792458 m/s."
    if "water boil" in q:
        return "Water boils at 100 degrees Celsius."
    if "orbit" in q:
        return "Earth orbits the Sun in 365.25 days."
    return "I do not know."


def _middle_blind_model(prompt: str) -> str:
    """Model that answers start/end but not middle questions."""
    q = _extract_question(prompt)
    if "speed of light" in q:
        return "The speed of light is 299792458 m/s."
    if "orbit" in q:
        return "Earth orbits the Sun in 365.25 days."
    # Misses middle question entirely
    return "I am not sure about that."


# ---------------------------------------------------------------------------
# Needle-in-a-Haystack tests
# ---------------------------------------------------------------------------


class TestNeedleInHaystack:
    """Tests for assert_needle_in_haystack."""

    def test_model_finds_all_positions_passes(self) -> None:
        """PASS: Model retrieves needle at every position."""
        haystack = "Lorem ipsum dolor sit amet. " * 200
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack=haystack,
            min_recall=0.8,
        )
        assert result.passed is True
        assert result.details["recall"] == 1.0
        assert result.details["n_positions"] == 5
        assert all(result.details["per_position"].values())

    def test_model_misses_all_positions_fails(self) -> None:
        """FAIL: Model never retrieves the needle."""
        haystack = "Lorem ipsum dolor sit amet. " * 200
        with pytest.raises(MltkAssertionError) as exc:
            assert_needle_in_haystack(
                model_fn=_never_finds_needle,
                needle="The secret code is 7492",
                haystack=haystack,
                min_recall=0.8,
            )
        assert exc.value.result.details["recall"] == 0.0

    def test_model_misses_middle_fails(self) -> None:
        """FAIL: Model retrieves edges but misses middle positions."""
        haystack = "Lorem ipsum dolor sit amet. " * 500
        with pytest.raises(MltkAssertionError):
            assert_needle_in_haystack(
                model_fn=_finds_edges_only,
                needle="The secret code is 7492",
                haystack=haystack,
                min_recall=1.0,
            )

    def test_custom_positions(self) -> None:
        """PASS: Custom positions list is respected."""
        haystack = "Background text. " * 100
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack=haystack,
            positions=[0.0, 0.5, 1.0],
            min_recall=0.8,
        )
        assert result.passed is True
        assert result.details["n_positions"] == 3

    def test_single_position(self) -> None:
        """PASS: Single position works correctly."""
        haystack = "Background text. " * 50
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack=haystack,
            positions=[0.5],
            min_recall=1.0,
        )
        assert result.passed is True
        assert result.details["n_positions"] == 1

    def test_empty_haystack(self) -> None:
        """EDGE: Empty haystack still inserts needle and calls model."""
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack="",
            min_recall=0.8,
        )
        assert result.passed is True
        assert result.details["haystack_length"] == 0

    def test_model_error_handled(self) -> None:
        """EDGE: Model exception counts as not-found, does not crash."""
        def error_model(prompt: str) -> str:
            raise RuntimeError("GPU out of memory")

        with pytest.raises(MltkAssertionError):
            assert_needle_in_haystack(
                model_fn=error_model,
                needle="The secret code is 7492",
                haystack="Some text " * 100,
                min_recall=0.8,
            )

    def test_details_structure(self) -> None:
        """Verify TestResult details contain all expected keys."""
        haystack = "Text " * 50
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack=haystack,
        )
        assert "recall" in result.details
        assert "min_recall" in result.details
        assert "per_position" in result.details
        assert "n_positions" in result.details
        assert "needle_length" in result.details
        assert "haystack_length" in result.details

    def test_assertion_name(self) -> None:
        """Verify assertion uses the correct name."""
        result = assert_needle_in_haystack(
            model_fn=_always_finds_needle,
            needle="The secret code is 7492",
            haystack="Text " * 50,
        )
        assert result.name == "llm.long_context.needle_in_haystack"


# ---------------------------------------------------------------------------
# Context Utilization tests
# ---------------------------------------------------------------------------


class TestContextUtilization:
    """Tests for assert_context_utilization."""

    def test_model_uses_all_facts_passes(self) -> None:
        """PASS: Model references enough facts in its response."""
        facts = [
            "The capital of France is Paris.",
            "The Eiffel Tower is 330 meters tall.",
            "France has a population of 67 million.",
            "The official language is French.",
            "France uses the Euro currency.",
        ]
        result = assert_context_utilization(
            model_fn=_uses_all_facts,
            facts=facts,
            question="Summarize what you know about France.",
            min_facts_used=3,
        )
        assert result.passed is True
        assert result.details["facts_used"] >= 3
        assert result.details["total_facts"] == 5

    def test_model_ignores_facts_fails(self) -> None:
        """FAIL: Model ignores all provided facts."""
        facts = [
            "The capital of France is Paris.",
            "The Eiffel Tower is 330 meters tall.",
            "France has a population of 67 million.",
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_context_utilization(
                model_fn=_uses_no_facts,
                facts=facts,
                question="Summarize France.",
                min_facts_used=2,
            )
        assert exc.value.result.details["facts_used"] == 0

    def test_model_uses_too_few_facts_fails(self) -> None:
        """FAIL: Model uses only 1 fact when 3 are required."""
        facts = [
            "The capital of France is Paris.",
            "The Eiffel Tower is 330 meters tall.",
            "France has a population of 67 million.",
        ]
        with pytest.raises(MltkAssertionError):
            assert_context_utilization(
                model_fn=_uses_one_fact,
                facts=facts,
                question="Tell me about France.",
                min_facts_used=3,
            )

    def test_single_fact(self) -> None:
        """EDGE: Single fact with min_facts_used=1."""
        result = assert_context_utilization(
            model_fn=_uses_one_fact,
            facts=["The capital of France is Paris."],
            question="What is the capital?",
            min_facts_used=1,
        )
        assert result.passed is True
        assert result.details["facts_used"] == 1
        assert result.details["per_fact_found"] == [True]

    def test_model_error_handled(self) -> None:
        """EDGE: Model exception produces a failing result."""
        def error_model(prompt: str) -> str:
            raise RuntimeError("Connection timeout")

        with pytest.raises(MltkAssertionError) as exc:
            assert_context_utilization(
                model_fn=error_model,
                facts=["Fact one.", "Fact two."],
                question="What?",
                min_facts_used=1,
            )
        assert exc.value.result.details["facts_used"] == 0

    def test_model_returns_empty(self) -> None:
        """EDGE: Model returns empty string -- no facts used."""
        def empty_model(prompt: str) -> str:
            return ""

        with pytest.raises(MltkAssertionError):
            assert_context_utilization(
                model_fn=empty_model,
                facts=["Alpha beta gamma.", "Delta epsilon zeta."],
                question="Summarize.",
                min_facts_used=1,
            )

    def test_assertion_name(self) -> None:
        """Verify assertion uses the correct name."""
        result = assert_context_utilization(
            model_fn=_uses_all_facts,
            facts=["The capital of France is Paris."],
            question="What is the capital?",
            min_facts_used=1,
        )
        assert result.name == "llm.long_context.utilization"

    def test_per_fact_found_length(self) -> None:
        """per_fact_found list has same length as facts input."""
        facts = ["Alpha.", "Beta.", "Gamma."]
        result = assert_context_utilization(
            model_fn=_uses_all_facts,
            facts=facts,
            question="Tell me the letters.",
            min_facts_used=0,
        )
        assert len(result.details["per_fact_found"]) == len(facts)


# ---------------------------------------------------------------------------
# No Lost-in-Middle tests
# ---------------------------------------------------------------------------


class TestNoLostInMiddle:
    """Tests for assert_no_lost_in_middle."""

    def test_uniform_accuracy_passes(self) -> None:
        """PASS: Model answers all positions correctly."""
        facts = [
            "The speed of light is 299792458 m/s.",
            "Water boils at 100 degrees Celsius.",
            "Earth orbits the Sun in 365.25 days.",
        ]
        questions = [
            "What is the speed of light?",
            "At what temperature does water boil?",
            "How long does Earth take to orbit the Sun?",
        ]
        result = assert_no_lost_in_middle(
            model_fn=_uniform_accuracy_model,
            facts=facts,
            questions=questions,
            min_accuracy=0.7,
        )
        assert result.passed is True
        assert result.details["accuracy"] == pytest.approx(1.0)
        assert result.details["n_questions"] == 3

    def test_middle_gap_fails(self) -> None:
        """FAIL: Model misses middle question -- lost-in-middle effect."""
        facts = [
            "The speed of light is 299792458 m/s.",
            "Water boils at 100 degrees Celsius.",
            "Earth orbits the Sun in 365.25 days.",
        ]
        questions = [
            "What is the speed of light?",
            "At what temperature does water boil?",
            "How long does Earth take to orbit the Sun?",
        ]
        # min_accuracy=1.0 means all must be correct
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_lost_in_middle(
                model_fn=_middle_blind_model,
                facts=facts,
                questions=questions,
                min_accuracy=1.0,
            )
        details = exc.value.result.details
        assert details["per_position_correct"]["middle"] is False
        assert details["per_position_correct"]["beginning"] is True
        assert details["per_position_correct"]["end"] is True

    def test_per_position_accuracy_reported(self) -> None:
        """Verify per-position accuracy dict is populated."""
        facts = [
            "The speed of light is 299792458 m/s.",
            "Water boils at 100 degrees Celsius.",
            "Earth orbits the Sun in 365.25 days.",
        ]
        questions = [
            "What is the speed of light?",
            "At what temperature does water boil?",
            "How long does Earth take to orbit the Sun?",
        ]
        result = assert_no_lost_in_middle(
            model_fn=_uniform_accuracy_model,
            facts=facts,
            questions=questions,
            min_accuracy=0.5,
        )
        ppa = result.details["per_position_accuracy"]
        assert "beginning" in ppa
        assert "middle" in ppa
        assert "end" in ppa

    def test_mismatched_lengths_raises(self) -> None:
        """EDGE: Different-length facts/questions raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            assert_no_lost_in_middle(
                model_fn=_uniform_accuracy_model,
                facts=["Fact 1.", "Fact 2."],
                questions=["Q1?"],
                min_accuracy=0.5,
            )

    def test_model_returns_empty(self) -> None:
        """EDGE: Empty model response -- nothing matches."""
        def empty_model(prompt: str) -> str:
            return ""

        with pytest.raises(MltkAssertionError):
            assert_no_lost_in_middle(
                model_fn=empty_model,
                facts=["Alpha beta.", "Gamma delta."],
                questions=["What alpha?", "What gamma?"],
                min_accuracy=0.5,
            )

    def test_model_error_handled(self) -> None:
        """EDGE: Model exception does not crash the assertion."""
        def error_model(prompt: str) -> str:
            raise RuntimeError("Segfault")

        with pytest.raises(MltkAssertionError):
            assert_no_lost_in_middle(
                model_fn=error_model,
                facts=["Fact A.", "Fact B.", "Fact C."],
                questions=["Q A?", "Q B?", "Q C?"],
                min_accuracy=0.5,
            )

    def test_assertion_name(self) -> None:
        """Verify assertion uses the correct name."""
        facts = [
            "The speed of light is 299792458 m/s.",
            "Water boils at 100 degrees Celsius.",
            "Earth orbits the Sun in 365.25 days.",
        ]
        questions = [
            "What is the speed of light?",
            "At what temperature does water boil?",
            "How long does Earth take to orbit the Sun?",
        ]
        result = assert_no_lost_in_middle(
            model_fn=_uniform_accuracy_model,
            facts=facts,
            questions=questions,
            min_accuracy=0.5,
        )
        assert result.name == "llm.long_context.no_lost_in_middle"

    def test_single_fact_pair(self) -> None:
        """EDGE: Single fact/question pair still works."""
        result = assert_no_lost_in_middle(
            model_fn=_uniform_accuracy_model,
            facts=["The speed of light is 299792458 m/s."],
            questions=["What is the speed of light?"],
            min_accuracy=0.5,
        )
        assert result.passed is True
        assert result.details["n_questions"] == 1
