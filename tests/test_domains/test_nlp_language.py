"""Tests for mltk.domains.nlp.language -- langdetect-backed language ID."""

from __future__ import annotations

import pytest

pytest.importorskip("langdetect", reason="langdetect required for language-ID tests")

from mltk.core.assertion import MltkAssertionError  # noqa: E402
from mltk.domains.nlp.language import assert_language  # noqa: E402


class TestAssertLanguage:
    def test_english_text_passes(self) -> None:
        result = assert_language(
            "The cat sat on the mat and watched the birds outside.",
            expected="en",
        )
        assert result.passed is True
        assert result.details["detected"] == "en"

    def test_wrong_language_fails(self) -> None:
        with pytest.raises(MltkAssertionError):
            assert_language(
                "The cat sat on the mat and watched the birds outside.",
                expected="fr",
            )

    def test_probability_gate(self) -> None:
        result = assert_language(
            "This is a long enough English sentence for a confident detection.",
            expected="en",
            min_prob=0.5,
        )
        assert result.passed is True
        # min_prob gates on the EXPECTED language's confidence.
        assert 0.0 <= result.details["expected_probability"] <= 1.0
        assert result.details["expected_probability"] >= 0.5
        assert result.details["detected_probability"] >= 0.5

    def test_empty_text_fails(self) -> None:
        with pytest.raises(MltkAssertionError, match="empty"):
            assert_language("", expected="en")

    def test_whitespace_only_fails(self) -> None:
        with pytest.raises(MltkAssertionError, match="empty"):
            assert_language("   \n", expected="en")

    def test_missing_langdetect_message_names_mlspec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "langdetect", None)
        monkeypatch.setitem(sys.modules, "langdetect.lang_detect_exception", None)
        with pytest.raises(ImportError, match=r"mlspec\[langdetect\]"):
            assert_language("hello world this is english text", expected="en")


def test_probability_fields_have_fixed_meaning() -> None:
    """The two confidences do not swap referents based on min_prob.

    `probability` used to carry the EXPECTED language's confidence when
    min_prob was set and the DETECTED language's when it was not. Same
    input and same verdict then reported 0.9999967 or 0.0 under one key,
    and the 0.9999967 sat beside expected="en" while meaning confidence
    in "fr" -- the opposite of what happened.
    """
    pytest.importorskip("langdetect")
    from mltk.core.assertion import MltkAssertionError
    from mltk.domains.nlp.language import assert_language

    french = "Bonjour, je suis tres heureux de vous rencontrer aujourd hui."

    details = []
    for kwargs in ({}, {"min_prob": 0.5}):
        try:
            details.append(assert_language(french, "en", **kwargs).details)
        except MltkAssertionError as exc:
            details.append(exc.result.details)

    without, with_min = details
    # Detected confidence is the same run either way.
    assert without["detected"] == with_min["detected"] == "fr"
    assert without["detected_probability"] == with_min["detected_probability"]
    assert without["detected_probability"] > 0.9
    # Expected never appeared among the candidates, in both calls.
    assert without["expected_probability"] == with_min["expected_probability"] == 0.0


def test_empty_text_reports_both_confidences_as_none() -> None:
    """Detection never ran, so neither confidence is a number."""
    from mltk.core.assertion import MltkAssertionError
    from mltk.domains.nlp.language import assert_language

    with pytest.raises(MltkAssertionError) as exc:
        assert_language("   ", "en")
    details = exc.value.result.details
    assert details["detected"] is None
    assert details["detected_probability"] is None
    assert details["expected_probability"] is None
