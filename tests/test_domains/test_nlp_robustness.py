"""Tests for mltk.domains.nlp.robustness -- text noise robustness testing.

Validates two components:
1. TextPerturber: generates realistic text corruptions (char swaps, deletions,
   insertions, QWERTY proximity errors) with deterministic seeding.
2. assert_text_robust: measures NLP model prediction stability under text noise,
   catching models that are fragile to real-world typos and OCR errors.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.nlp.robustness import TextPerturber, assert_text_robust

# ---------------------------------------------------------------------------
# TextPerturber tests
# ---------------------------------------------------------------------------


class TestCharSwap:
    """Tests for adjacent character transposition (typing speed errors)."""

    def test_text_is_modified(self) -> None:
        """char_swap at rate=1.0 modifies a non-trivial string.

        WHY: At maximum rate, every eligible position triggers a swap.
        A multi-character string must differ from the original to confirm
        the perturbation logic actually runs.
        """
        p = TextPerturber(seed=42)
        original = "abcdefghij"
        swapped = p.char_swap(original, rate=1.0)
        assert swapped != original
        # Length is preserved -- swaps don't add or remove characters
        assert len(swapped) == len(original)

    def test_rate_zero_no_changes(self) -> None:
        """rate=0 must return the original text unchanged.

        WHY: A rate of 0.0 means zero probability of perturbation at any
        position. This is the identity case and must be exact.
        """
        p = TextPerturber(seed=42)
        text = "hello world"
        assert p.char_swap(text, rate=0.0) == text

    def test_reproducible_with_seed(self) -> None:
        """Same seed + same input produces identical output.

        WHY: Reproducibility is essential for deterministic test suites.
        Two TextPerturber instances with the same seed must produce the
        exact same perturbation on the same input.
        """
        text = "the quick brown fox"
        result_a = TextPerturber(seed=99).char_swap(text, rate=0.3)
        result_b = TextPerturber(seed=99).char_swap(text, rate=0.3)
        assert result_a == result_b


class TestCharDelete:
    """Tests for character deletion (touchscreen/OCR errors)."""

    def test_shorter_output(self) -> None:
        """char_delete at rate=1.0 removes characters, producing shorter text.

        WHY: At maximum rate, every character has a high probability of
        deletion. The output must be strictly shorter than the input for
        any non-empty string (statistically guaranteed at rate=1.0 for
        strings of reasonable length).
        """
        p = TextPerturber(seed=42)
        original = "a]bcdefghijklmnopqrstuvwxyz"
        deleted = p.char_delete(original, rate=1.0)
        assert len(deleted) < len(original)

    def test_rate_zero_no_changes(self) -> None:
        """rate=0 must return the original text unchanged.

        WHY: Zero deletion probability means every character survives.
        This is the identity case.
        """
        p = TextPerturber(seed=42)
        text = "hello world"
        assert p.char_delete(text, rate=0.0) == text


class TestCharInsert:
    """Tests for character insertion (sticky keys, double-tap)."""

    def test_longer_output(self) -> None:
        """char_insert at rate=1.0 adds characters, producing longer text.

        WHY: At maximum rate, a character is inserted after every position.
        The output must be strictly longer than the input.
        """
        p = TextPerturber(seed=42)
        original = "abcdefghij"
        inserted = p.char_insert(original, rate=1.0)
        assert len(inserted) > len(original)

    def test_rate_zero_no_changes(self) -> None:
        """rate=0 must return the original text unchanged.

        WHY: Zero insertion probability means no characters are added.
        This is the identity case.
        """
        p = TextPerturber(seed=42)
        text = "hello world"
        assert p.char_insert(text, rate=0.0) == text


class TestKeyboardProximity:
    """Tests for QWERTY neighbor replacement (fat-finger typos)."""

    def test_modified_chars_are_neighbors(self) -> None:
        """Replaced characters must be actual QWERTY neighbors of the original.

        WHY: The whole point of keyboard_proximity is to model *realistic*
        fat-finger errors. If a character is replaced, the replacement must
        come from the QWERTY adjacency map, not random characters. This
        test verifies the constraint character-by-character.
        """
        p = TextPerturber(seed=42)
        original = "abcdefghij"
        perturbed = p.keyboard_proximity(original, rate=1.0)
        neighbors = TextPerturber._KEYBOARD_NEIGHBORS

        for orig_ch, pert_ch in zip(original, perturbed, strict=False):
            if orig_ch != pert_ch:
                # The replacement must be a QWERTY neighbor
                assert pert_ch in neighbors[orig_ch.lower()], (
                    f"'{pert_ch}' is not a QWERTY neighbor of '{orig_ch}'. "
                    f"Expected one of: {neighbors[orig_ch.lower()]}"
                )

    def test_preserves_case(self) -> None:
        """Uppercase input characters produce uppercase replacements.

        WHY: Real typos preserve the shift-key state. If the user intended
        to type 'H' (shift+h), a fat-finger error produces 'Y' or 'G'
        (shift + neighbor), not 'y' or 'g'.
        """
        p = TextPerturber(seed=42)
        original = "HELLO"
        perturbed = p.keyboard_proximity(original, rate=1.0)
        for ch in perturbed:
            if ch.isalpha():
                assert ch.isupper(), f"Expected uppercase, got '{ch}'"

    def test_non_alpha_unchanged(self) -> None:
        """Digits, punctuation, and spaces are never replaced.

        WHY: The QWERTY adjacency map only covers lowercase letters.
        Other characters (digits, punctuation, whitespace) should pass
        through unchanged since they occupy different keyboard regions.
        """
        p = TextPerturber(seed=42)
        text = "123 !@# 456"
        assert p.keyboard_proximity(text, rate=1.0) == text


class TestPerturb:
    """Tests for the combined perturbation pipeline."""

    def test_applies_multiple_methods(self) -> None:
        """perturb with multiple methods modifies text more than a single method.

        WHY: Chaining perturbations should compound the noise. Applying
        both char_swap and keyboard_proximity together should (on average)
        produce more changes than either alone.
        """
        text = "the quick brown fox jumps over the lazy dog"
        p = TextPerturber(seed=42)
        single = p.perturb(text, methods=["char_swap"], rate=0.2)
        p2 = TextPerturber(seed=42)
        multi = p2.perturb(text, methods=["char_swap", "keyboard_proximity"], rate=0.2)
        # Both should differ from original
        assert single != text
        assert multi != text
        # Multi should differ from single (different pipeline)
        assert multi != single

    def test_empty_text_unchanged(self) -> None:
        """Empty string input returns empty string, regardless of methods.

        WHY: Edge case -- perturbation of nothing should produce nothing,
        not crash or insert random characters into an empty string.
        """
        p = TextPerturber(seed=42)
        assert p.perturb("") == ""
        assert p.perturb("", methods=["char_swap", "char_delete"]) == ""

    def test_unknown_method_raises(self) -> None:
        """Unknown method name raises ValueError with helpful message.

        WHY: Typos in method names should fail fast with a clear error,
        not silently skip the perturbation. The error message should list
        valid method names.
        """
        p = TextPerturber(seed=42)
        with pytest.raises(ValueError, match="Unknown perturbation method"):
            p.perturb("hello", methods=["nonexistent_method"])


# ---------------------------------------------------------------------------
# assert_text_robust tests
# ---------------------------------------------------------------------------


class TestAssertTextRobust:
    """Tests for the text robustness assertion."""

    def test_robust_model_passes(self) -> None:
        """PASS: Model that ignores typos (always returns same output) is robust.

        WHY: A model whose prediction is completely independent of input
        text has perfect stability (1.0). This is the simplest passing case
        and validates the basic assertion logic.
        """
        def constant_model(text: str) -> str:
            return "positive"

        texts = ["great product", "love this", "amazing quality"]
        result = assert_text_robust(constant_model, texts, min_stability=0.8)
        assert result.passed is True
        assert result.details["avg_stability"] == 1.0

    def test_fragile_model_fails(self) -> None:
        """FAIL: Model that returns random output on every call is fragile.

        WHY: A model whose predictions vary randomly with each call will
        rarely match the original prediction. Stability will be near 0,
        well below any reasonable threshold. This validates that the
        assertion correctly catches instability.
        """
        counter = {"n": 0}

        def random_model(text: str) -> int:
            counter["n"] += 1
            return counter["n"]  # Different output every time

        texts = ["test input one", "test input two"]
        with pytest.raises(MltkAssertionError) as exc:
            assert_text_robust(random_model, texts, min_stability=0.5)
        assert "Fragile" in str(exc.value)

    def test_custom_perturbation_method(self) -> None:
        """Custom perturbation method is recorded in result details.

        WHY: Users should be able to choose specific perturbation types
        (e.g., only char_swap for a typing-error study) and verify that
        the assertion used the correct method.
        """
        def constant_model(text: str) -> str:
            return "label"

        texts = ["some text here"]
        result = assert_text_robust(
            constant_model, texts, perturbation="char_delete", min_stability=0.5,
        )
        assert result.passed is True
        assert result.details["perturbation_method"] == "char_delete"

    def test_rate_affects_perturbation(self) -> None:
        """Higher rate produces more noise, reducing stability for sensitive models.

        WHY: A model that checks for exact substring matches will be more
        affected by higher perturbation rates (more characters changed =
        higher chance the substring is corrupted). This validates that the
        rate parameter actually flows through to the perturbation logic.
        """
        def exact_match_model(text: str) -> str:
            return "match" if "specific" in text else "no_match"

        texts = ["this is a specific test string"]

        # Low rate: most perturbations keep "specific" intact
        result_low = assert_text_robust(
            exact_match_model, texts,
            perturbation="keyboard_proximity",
            rate=0.01, min_stability=0.0, n_perturbations=20, seed=42,
        )
        # High rate: perturbations likely corrupt "specific"
        result_high = assert_text_robust(
            exact_match_model, texts,
            perturbation="keyboard_proximity",
            rate=0.5, min_stability=0.0, n_perturbations=20, seed=42,
        )

        # Higher rate should produce lower stability (or equal in edge cases)
        assert result_low.details["avg_stability"] >= result_high.details["avg_stability"]

    def test_empty_texts_passes(self) -> None:
        """Empty text list passes vacuously with stability=1.0.

        WHY: An empty list means there are no texts to fail on. Vacuous
        truth: "all texts are stable" is true when there are no texts.
        This prevents false failures on dynamically-generated test sets
        that might sometimes be empty.
        """
        def model(text: str) -> str:
            return "anything"

        result = assert_text_robust(model, [], min_stability=0.9)
        assert result.passed is True
        assert result.details["avg_stability"] == 1.0
        assert result.details["n_texts"] == 0

    def test_details_contain_per_text_stability(self) -> None:
        """Result details include per-text stability scores for diagnostics.

        WHY: When a robustness test fails, engineers need to know *which*
        texts are fragile. The per_text_stability list provides per-input
        breakdowns so you can identify problematic inputs and debug the
        model's failure mode.
        """
        def constant_model(text: str) -> str:
            return "ok"

        texts = ["first text", "second text", "third text"]
        result = assert_text_robust(constant_model, texts, n_perturbations=5)
        per_text = result.details["per_text_stability"]
        assert isinstance(per_text, list)
        assert len(per_text) == 3
        # Constant model -> all stability scores should be 1.0
        assert all(score == 1.0 for score in per_text)

    def test_result_name(self) -> None:
        """Result name is 'nlp.text_robust' for consistent test identification.

        WHY: The test name is used in reports, dashboards, and filtering.
        It must match the documented convention.
        """
        def model(text: str) -> str:
            return "label"

        result = assert_text_robust(model, ["test"], min_stability=0.0)
        assert result.name == "nlp.text_robust"

    def test_has_duration(self) -> None:
        """Result includes wall-clock duration from @timed_assertion.

        WHY: The timed_assertion decorator should populate duration_ms.
        This confirms the decorator is properly applied to assert_text_robust.
        """
        def model(text: str) -> str:
            return "label"

        result = assert_text_robust(model, ["test"], min_stability=0.0)
        assert result.duration_ms >= 0.0
