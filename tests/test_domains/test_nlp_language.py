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
        assert 0.0 <= result.details["probability"] <= 1.0
        assert result.details["probability"] >= 0.5

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
