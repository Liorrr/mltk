"""NLP text robustness testing -- verify model stability under realistic text noise.

Real-world text inputs are noisy. Users make typos, OCR pipelines garble
characters, touchscreen keyboards miss taps, and copy-paste introduces encoding
artifacts. A production-grade NLP model must handle these gracefully.

This module provides two tools:

1. **TextPerturber** -- a configurable noise generator that creates realistic
   text corruptions modeled after actual error sources (typing speed, QWERTY
   layout, touchscreen behavior, sticky keys).

2. **assert_text_robust** -- an assertion that measures prediction stability
   across perturbed inputs. If a sentiment classifier flips from "positive" to
   "negative" just because a user typed "machnie" instead of "machine", the
   model is fragile and this assertion will catch it.

Pure Python, zero external dependencies. Uses ``random.Random`` instances
(not the global ``random.seed``) to stay thread-safe and reproducible.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


class TextPerturber:
    """Generate realistic text perturbations for robustness testing.

    Real-world text inputs are noisy: typos from fat fingers, OCR errors from
    scanned documents, encoding artifacts from copy-paste, and autocorrect
    mistakes. A robust NLP model should produce consistent predictions despite
    these common corruptions.

    TextPerturber generates controlled noise to test model robustness. Each
    method targets a specific real-world noise source:

    - **char_swap**: Adjacent character transposition (typing speed errors).
      When you type quickly, your fingers sometimes hit keys in the wrong order,
      turning "the" into "teh" or "machine" into "macihne".

    - **char_delete**: Missing characters (touchscreen keyboards, OCR).
      Touchscreen taps that don't register, or OCR engines that fail to
      recognize a character, produce shortened words like "learnig" for
      "learning".

    - **char_insert**: Extra characters (sticky keys, double-tap).
      Mechanical keyboards with sticky switches or touchscreen double-taps
      produce "learrning" for "learning".

    - **keyboard_proximity**: Wrong character from neighboring key (QWERTY
      layout). Fat-finger errors hit an adjacent key instead of the intended
      one, producing "mschine" (s neighbors a on QWERTY) for "machine".

    The ``rate`` parameter controls corruption intensity: 0.0 = no changes,
    0.1 = ~10% of characters affected, 1.0 = every character affected.

    Parameters:
        seed: Random seed for reproducible perturbations. When ``None``,
            results vary between runs. When set, the same seed always
            produces the same perturbation for the same input.

    Example:
        >>> p = TextPerturber(seed=42)
        >>> original = "machine learning is great"
        >>> noisy = p.char_swap(original, rate=0.1)
        >>> # noisy might be "machien learning is great"
        >>> noisy2 = p.keyboard_proximity(original, rate=0.1)
        >>> # noisy2 might be "mschine learning is great"
    """

    # QWERTY keyboard adjacency map (each key -> string of nearby keys).
    # This models the physical layout of a standard US QWERTY keyboard.
    # Keys that are physically adjacent are more likely to be hit by
    # a fat-finger error than distant keys.
    _KEYBOARD_NEIGHBORS: dict[str, str] = {
        "q": "wa",
        "w": "qeas",
        "e": "wrds",
        "r": "etfs",
        "t": "rygf",
        "y": "tuhg",
        "u": "yijh",
        "i": "uokj",
        "o": "iplk",
        "p": "ol",
        "a": "qwsz",
        "s": "wedxza",
        "d": "erfcxs",
        "f": "rtgvcd",
        "g": "tyhbvf",
        "h": "yujnbg",
        "j": "uikmnh",
        "k": "iolmj",
        "l": "opk",
        "z": "asx",
        "x": "zsdc",
        "c": "xdfv",
        "v": "cfgb",
        "b": "vghn",
        "n": "bhjm",
        "m": "njk",
    }

    def __init__(self, seed: int | None = None) -> None:
        """Initialize with optional random seed for reproducibility.

        Args:
            seed: If provided, perturbations are deterministic -- the same
                seed + same input always produces the same output. This is
                essential for reproducible test suites. Uses a local
                ``random.Random`` instance rather than ``random.seed()``
                to avoid interfering with other random state (thread-safe).
        """
        self._rng = random.Random(seed)

    def char_swap(self, text: str, rate: float = 0.1) -> str:
        """Swap adjacent characters to simulate typing speed errors.

        When typing quickly, fingers sometimes strike keys in the wrong order.
        This is especially common with frequent bigrams: "th" -> "ht",
        "er" -> "re", "in" -> "ni". The effect is transposition of two
        adjacent characters at random positions throughout the text.

        Args:
            text: Input text to perturb.
            rate: Probability that each character position triggers a swap
                with its right neighbor. 0.0 = no swaps, 1.0 = swap at every
                eligible position. A realistic typo rate is 0.05-0.15.

        Returns:
            Text with some adjacent characters transposed.

        Example:
            >>> p = TextPerturber(seed=0)
            >>> p.char_swap("hello world", rate=0.2)  # might produce "hlelo world"
        """
        if not text or rate <= 0.0:
            return text

        chars = list(text)
        i = 0
        while i < len(chars) - 1:
            if self._rng.random() < rate:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                # Skip the next position to avoid double-swapping the same char
                i += 2
            else:
                i += 1
        return "".join(chars)

    def char_delete(self, text: str, rate: float = 0.05) -> str:
        """Delete random characters to simulate touchscreen/OCR errors.

        Touchscreen keyboards sometimes fail to register a tap, and OCR
        engines occasionally miss characters in scanned documents. Both
        produce shortened words: "learning" -> "learnig", "the" -> "th".

        A lower default rate (0.05) reflects that character deletion is
        less common than transposition in real-world data, but more
        destructive -- a missing character can completely change a word's
        meaning or make it unrecognizable to a tokenizer.

        Args:
            text: Input text to perturb.
            rate: Probability that each character is deleted. 0.0 = no
                deletions, 1.0 = delete every character. Realistic: 0.02-0.08.

        Returns:
            Text with some characters removed (shorter or equal length).

        Example:
            >>> p = TextPerturber(seed=0)
            >>> p.char_delete("hello world", rate=0.2)  # might produce "hllo wrld"
        """
        if not text or rate <= 0.0:
            return text

        return "".join(ch for ch in text if self._rng.random() >= rate)

    def char_insert(self, text: str, rate: float = 0.05) -> str:
        """Insert random characters to simulate sticky keys or double-taps.

        Mechanical keyboards with worn switches sometimes register a keypress
        twice ("sticky keys"), and touchscreen keyboards can register an
        unintended tap on a neighboring area. This produces extra characters:
        "learning" -> "learrning" or "leaxrning".

        Inserted characters are drawn from lowercase ASCII letters to keep
        the noise realistic -- real sticky-key errors produce valid characters,
        not random unicode.

        Args:
            text: Input text to perturb.
            rate: Probability that a random character is inserted after each
                position. 0.0 = no insertions, 1.0 = insert after every char.
                Realistic: 0.02-0.08.

        Returns:
            Text with some extra characters inserted (longer or equal length).

        Example:
            >>> p = TextPerturber(seed=0)
            >>> p.char_insert("hello", rate=0.3)  # might produce "hezllo"
        """
        if not text or rate <= 0.0:
            return text

        result: list[str] = []
        for ch in text:
            result.append(ch)
            if self._rng.random() < rate:
                result.append(self._rng.choice(string.ascii_lowercase))
        return "".join(result)

    def keyboard_proximity(self, text: str, rate: float = 0.1) -> str:
        """Replace characters with QWERTY neighbors to simulate fat-finger typos.

        On a physical keyboard, fat-finger errors hit an adjacent key instead
        of the intended one. The error distribution is NOT uniform -- it
        depends on the physical keyboard layout. For example, pressing "a"
        might accidentally hit "q", "w", "s", or "z" (its QWERTY neighbors),
        but never "p" or "m" (far away on the keyboard).

        This method uses a QWERTY adjacency map to select realistic
        replacement characters. Characters not in the map (digits,
        punctuation, uppercase, spaces) are left unchanged, since they
        occupy different keyboard regions with different error patterns.

        Args:
            text: Input text to perturb.
            rate: Probability that each character is replaced by a neighbor.
                0.0 = no replacements, 1.0 = replace every eligible char.
                Realistic: 0.05-0.15.

        Returns:
            Text with some characters replaced by their QWERTY neighbors.

        Example:
            >>> p = TextPerturber(seed=0)
            >>> p.keyboard_proximity("hello", rate=0.5)  # might produce "gello"
        """
        if not text or rate <= 0.0:
            return text

        result: list[str] = []
        for ch in text:
            lower = ch.lower()
            if self._rng.random() < rate and lower in self._KEYBOARD_NEIGHBORS:
                neighbors = self._KEYBOARD_NEIGHBORS[lower]
                replacement = self._rng.choice(neighbors)
                # Preserve original case
                result.append(replacement.upper() if ch.isupper() else replacement)
            else:
                result.append(ch)
        return "".join(result)

    def perturb(
        self,
        text: str,
        methods: list[str] | None = None,
        rate: float = 0.1,
    ) -> str:
        """Apply one or more perturbation methods sequentially.

        This is a convenience method that chains multiple perturbation types
        into a single call. Methods are applied in the order given, so the
        output of one becomes the input of the next. This models real-world
        scenarios where multiple noise sources compound: a user might both
        fat-finger a key AND accidentally double-tap, producing overlapping
        error types.

        When no methods are specified, all four perturbation types are applied
        in sequence: char_swap -> char_delete -> char_insert ->
        keyboard_proximity. This produces the most thorough (and aggressive)
        noise for stress-testing.

        Args:
            text: Input text to perturb.
            methods: List of method names to apply in order. Valid names:
                ``"char_swap"``, ``"char_delete"``, ``"char_insert"``,
                ``"keyboard_proximity"``. If ``None``, all methods are applied.
            rate: Perturbation rate passed to each method.

        Returns:
            Text after all specified perturbations have been applied.

        Raises:
            ValueError: If an unknown method name is provided.

        Example:
            >>> p = TextPerturber(seed=42)
            >>> p.perturb("hello world", methods=["char_swap", "char_delete"])
        """
        if methods is None:
            methods = ["char_swap", "char_delete", "char_insert", "keyboard_proximity"]

        method_map: dict[str, Callable[[str, float], str]] = {
            "char_swap": self.char_swap,
            "char_delete": self.char_delete,
            "char_insert": self.char_insert,
            "keyboard_proximity": self.keyboard_proximity,
        }

        for method_name in methods:
            if method_name not in method_map:
                raise ValueError(
                    f"Unknown perturbation method: '{method_name}'. "
                    f"Supported: {sorted(method_map.keys())}"
                )
            text = method_map[method_name](text, rate)
        return text


@timed_assertion
def assert_text_robust(
    model_fn: Callable[[str], Any],
    texts: list[str],
    perturbation: str = "keyboard_proximity",
    n_perturbations: int = 5,
    min_stability: float = 0.8,
    rate: float = 0.1,
    seed: int | None = 42,
) -> TestResult:
    """Assert that an NLP model produces stable predictions under text noise.

    A robust NLP model should produce the same prediction whether the input
    says "machine learning" or "macihne learnign" (a common typo). This
    assertion generates N perturbed variants of each input text, runs the
    model on all variants, and measures prediction stability -- the fraction
    of perturbed inputs that produce the same output as the clean original.

    **How it works:**

    1. For each text in ``texts``, compute the original prediction via
       ``model_fn(text)``.
    2. Generate ``n_perturbations`` noisy variants using ``TextPerturber``
       with the specified perturbation method and rate.
    3. Run ``model_fn`` on each noisy variant.
    4. Stability for one text = (number of variants matching the original
       prediction) / n_perturbations.
    5. Average stability across all texts.
    6. **PASS** if average stability >= ``min_stability``.

    This catches models that are over-fit to clean, well-formatted text and
    break on the messy inputs that real users produce.

    Args:
        model_fn: Function that takes a single string and returns a prediction.
            The prediction can be any type (string label, int class, float
            score) -- equality comparison is used to check stability.
        texts: List of input texts to test. Each text is perturbed independently.
            An empty list results in a passing test with stability 1.0 (vacuously
            true -- there are no texts to fail on).
        perturbation: Perturbation method name. One of ``"char_swap"``,
            ``"char_delete"``, ``"char_insert"``, ``"keyboard_proximity"``.
        n_perturbations: Number of perturbed variants to generate per text.
            Higher values give more statistical confidence but increase runtime.
            5-10 is typical for unit tests; 50-100 for thorough CI pipelines.
        min_stability: Minimum average stability score to pass (0.0-1.0).
            0.8 means at least 80% of perturbed inputs must produce the same
            prediction as the original. Lower thresholds for exploratory
            testing, higher for production gates.
        rate: Perturbation intensity passed to TextPerturber. Higher rates
            produce noisier text. 0.1 is a moderate default; use 0.05 for
            light noise or 0.2 for aggressive stress testing.
        seed: Random seed for reproducible perturbations. Default 42 ensures
            deterministic test results. Set to ``None`` for randomized testing.

    Returns:
        TestResult with details:
            - ``avg_stability``: Mean stability across all texts (0.0-1.0).
            - ``min_stability``: The threshold that was required.
            - ``n_texts``: Number of input texts tested.
            - ``n_perturbations``: Number of variants generated per text.
            - ``perturbation_method``: Which perturbation method was used.
            - ``per_text_stability``: List of per-text stability scores.

    Example:
        >>> def sentiment(text):
        ...     return "positive" if "great" in text.lower() else "negative"
        >>> texts = ["This product is great", "Terrible experience"]
        >>> assert_text_robust(sentiment, texts, min_stability=0.6)
    """
    if not texts:
        return assert_true(
            True,
            name="nlp.text_robust",
            message="No texts to test (vacuously stable)",
            severity=Severity.CRITICAL,
            avg_stability=1.0,
            min_stability=min_stability,
            n_texts=0,
            n_perturbations=n_perturbations,
            perturbation_method=perturbation,
            per_text_stability=[],
        )

    perturber = TextPerturber(seed=seed)

    # Validate the perturbation method by attempting a dummy perturb
    valid_methods = {"char_swap", "char_delete", "char_insert", "keyboard_proximity"}
    if perturbation not in valid_methods:
        return assert_true(
            False,
            name="nlp.text_robust",
            message=(
                f"Unknown perturbation: '{perturbation}'. "
                f"Supported: {sorted(valid_methods)}"
            ),
            severity=Severity.CRITICAL,
        )

    per_text_stability: list[float] = []

    for text in texts:
        original_pred = model_fn(text)
        n_matching = 0

        for _ in range(n_perturbations):
            noisy_text = perturber.perturb(text, methods=[perturbation], rate=rate)
            noisy_pred = model_fn(noisy_text)
            if noisy_pred == original_pred:
                n_matching += 1

        text_stability = n_matching / n_perturbations if n_perturbations > 0 else 1.0
        per_text_stability.append(text_stability)

    avg_stability = sum(per_text_stability) / len(per_text_stability)
    passed = avg_stability >= min_stability

    message = (
        f"Text robustness: stability={avg_stability:.4f} >= {min_stability} "
        f"({len(texts)} texts, {n_perturbations} perturbations each, "
        f"method={perturbation})"
        if passed
        else f"Fragile: stability={avg_stability:.4f} < {min_stability} "
        f"({len(texts)} texts, {n_perturbations} perturbations each, "
        f"method={perturbation})"
    )

    return assert_true(
        passed,
        name="nlp.text_robust",
        message=message,
        severity=Severity.CRITICAL,
        avg_stability=avg_stability,
        min_stability=min_stability,
        n_texts=len(texts),
        n_perturbations=n_perturbations,
        perturbation_method=perturbation,
        per_text_stability=per_text_stability,
    )
