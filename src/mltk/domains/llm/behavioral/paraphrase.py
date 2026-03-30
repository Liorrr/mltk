"""Deterministic and LLM-backed paraphrase generation.

Provides ``ParaphraseGenerator`` — a utility for producing
semantically equivalent rephrasings of a text.  The template
method is zero-dependency and deterministic (CI-friendly);
the LLM method delegates to a user-supplied callable for
higher quality diversity.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# ---------------------------------------------------------------
# Contraction maps (bidirectional)
# ---------------------------------------------------------------

_EXPAND: list[tuple[str, str]] = [
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("didn't", "did not"),
    ("can't", "cannot"),
    ("couldn't", "could not"),
    ("won't", "will not"),
    ("wouldn't", "would not"),
    ("shouldn't", "should not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("weren't", "were not"),
    ("hasn't", "has not"),
    ("haven't", "have not"),
    ("hadn't", "had not"),
    ("it's", "it is"),
    ("that's", "that is"),
    ("there's", "there is"),
    ("what's", "what is"),
    ("who's", "who is"),
    ("let's", "let us"),
    ("i'm", "i am"),
    ("you're", "you are"),
    ("we're", "we are"),
    ("they're", "they are"),
    ("i've", "i have"),
    ("you've", "you have"),
    ("we've", "we have"),
    ("they've", "they have"),
    ("i'll", "i will"),
    ("you'll", "you will"),
    ("we'll", "we will"),
    ("they'll", "they will"),
    ("i'd", "i would"),
    ("you'd", "you would"),
    ("we'd", "we would"),
    ("they'd", "they would"),
]

# Build reverse map: expanded -> contracted
_CONTRACT: list[tuple[str, str]] = [
    (expanded, contracted)
    for contracted, expanded in _EXPAND
]

_FILLERS = ("So, ", "Well, ", "Basically, ", "")

_QUESTION_STARTERS = (
    "What is",
    "What are",
    "Who is",
    "Who are",
    "Where is",
    "Where are",
    "When is",
    "When did",
    "How do",
    "How does",
    "How is",
    "Why is",
    "Why do",
    "Why does",
)

_REFORMULATIONS = (
    "Explain",
    "Describe",
    "Tell me about",
    "Can you explain",
)

# Subordinating conjunctions for clause reordering
_CLAUSE_CONJUNCTIONS = (
    " because ",
    " since ",
    " when ",
)


class ParaphraseGenerator:
    """Generate paraphrases using deterministic templates.

    Template-based generation is fast and reproducible but
    produces surface-level variations. For deeper semantic
    diversity, use ``generate_llm()`` with a user-provided
    LLM function, or write paraphrases manually.

    Research note: CheckList (Ribeiro et al., ACL 2020)
    found that human-curated test cases catch more bugs
    than auto-generated ones. Use templates as a starting
    point, then add domain-specific manual paraphrases.
    """

    # -----------------------------------------------------
    # Template helpers
    # -----------------------------------------------------

    @staticmethod
    def _apply_contractions(
        text: str,
    ) -> str | None:
        """Toggle contractions. Returns None if no change."""
        lower = text.lower()
        # Try expanding contractions first
        for contracted, expanded in _EXPAND:
            if contracted in lower:
                result = re.sub(
                    re.escape(contracted),
                    expanded,
                    text,
                    flags=re.IGNORECASE,
                )
                if result != text:
                    return result
        # Try contracting expanded forms
        for expanded, contracted in _CONTRACT:
            if expanded in lower:
                result = re.sub(
                    re.escape(expanded),
                    contracted,
                    text,
                    flags=re.IGNORECASE,
                )
                if result != text:
                    return result
        return None

    @staticmethod
    def _reorder_clauses(
        text: str,
    ) -> str | None:
        """Swap clauses around because/since/when."""
        for conj in _CLAUSE_CONJUNCTIONS:
            if conj in text.lower():
                idx = text.lower().index(conj)
                before = text[:idx].strip()
                after = text[idx + len(conj):].strip()
                if not before or not after:
                    continue
                # Strip trailing punctuation from after
                punct = ""
                if after and after[-1] in ".?!":
                    punct = after[-1]
                    after = after[:-1].strip()
                conj_word = conj.strip()
                reordered = (
                    f"{conj_word.capitalize()} "
                    f"{after}, {before.lower()}"
                    f"{punct}"
                )
                return reordered
        return None

    @staticmethod
    def _reformulate_question(
        text: str,
    ) -> list[str]:
        """Turn 'What is X?' into 'Explain X', etc."""
        results: list[str] = []
        stripped = text.rstrip("?").strip()
        lower = stripped.lower()
        for starter in _QUESTION_STARTERS:
            if lower.startswith(starter.lower()):
                topic = stripped[len(starter):].strip()
                if not topic:
                    continue
                for reform in _REFORMULATIONS:
                    results.append(
                        f"{reform} {topic}"
                    )
                break
        return results

    @staticmethod
    def _add_fillers(text: str) -> list[str]:
        """Prepend filler words."""
        results: list[str] = []
        for filler in _FILLERS:
            candidate = f"{filler}{text}" if filler else text
            if candidate != text:
                results.append(candidate)
        return results

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def generate_template(
        self,
        text: str,
        n: int = 5,
    ) -> list[str]:
        """Template-based paraphrases (zero-dep, deterministic).

        Applies question reformulation, filler insertion,
        clause reordering, and contraction toggling to
        produce up to *n* unique paraphrases.

        Args:
            text: Input text to paraphrase.
            n: Maximum number of paraphrases to return.

        Returns:
            List of paraphrased strings (may be fewer than
            *n* if templates do not produce enough unique
            variants).
        """
        candidates: list[str] = []

        # 1. Question reformulation
        candidates.extend(
            self._reformulate_question(text)
        )

        # 2. Filler insertion
        candidates.extend(self._add_fillers(text))

        # 3. Clause reordering
        reordered = self._reorder_clauses(text)
        if reordered is not None:
            candidates.append(reordered)

        # 4. Contraction toggle
        toggled = self._apply_contractions(text)
        if toggled is not None:
            candidates.append(toggled)
            # Also combine with fillers
            for filler in _FILLERS:
                if filler:
                    candidates.append(
                        f"{filler}{toggled}"
                    )

        # Deduplicate preserving order, exclude original
        seen: set[str] = {text}
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique[:n]

    def generate_llm(
        self,
        text: str,
        llm_fn: Callable[[str], str],
        n: int = 5,
    ) -> list[str]:
        """LLM-based paraphrases (highest quality, needs API).

        Sends a rephrasing prompt to *llm_fn* and parses the
        response into individual paraphrases (one per line).

        Args:
            text: Input text to paraphrase.
            llm_fn: ``str -> str`` callable that accepts a
                prompt and returns the LLM response.
            n: Number of paraphrases to request.

        Returns:
            List of paraphrased strings parsed from the LLM
            response.  May be fewer than *n* if the LLM
            returns fewer lines.
        """
        prompt = (
            f"Rephrase the following text {n} "
            f"different ways, preserving the exact "
            f"meaning. Return one paraphrase per "
            f"line.\n\nText: {text}"
        )
        raw = llm_fn(prompt)
        lines = [
            line.strip() for line in raw.splitlines()
            if line.strip()
        ]
        # Strip leading numbering like "1. " or "1) "
        cleaned: list[str] = []
        for line in lines:
            stripped = re.sub(
                r"^\d+[\.\)]\s*", "", line,
            )
            if stripped:
                cleaned.append(stripped)
        return cleaned[:n]

    def generate(
        self,
        text: str,
        n: int = 5,
        method: str = "template",
        llm_fn: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Unified interface -- dispatches to template or LLM.

        Args:
            text: Input text to paraphrase.
            n: Maximum number of paraphrases to return.
            method: ``"template"`` (zero-dep) or ``"llm"``
                (requires *llm_fn*).
            llm_fn: Required when ``method="llm"``.

        Returns:
            List of paraphrased strings.

        Raises:
            ValueError: If *method* is unknown or ``"llm"``
                is selected without providing *llm_fn*.
        """
        if method == "template":
            return self.generate_template(text, n=n)

        if method == "llm":
            if llm_fn is None:
                raise ValueError(
                    "llm_fn is required for "
                    "method='llm'"
                )
            return self.generate_llm(
                text, llm_fn, n=n,
            )

        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Supported: 'template', 'llm'"
        )
