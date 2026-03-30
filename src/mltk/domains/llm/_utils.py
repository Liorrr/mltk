"""Private shared utilities for LLM domain modules.

Keep this module minimal — only truly shared, stateless helpers live here.
No heavy dependencies. For model-backed helpers, see ``_backends.py``.
"""

from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    """NFKC normalize + strip zero-width / invisible characters.

    Defends against homoglyph, zero-width, and bidirectional override
    attacks that bypass token-overlap checks (SEC-5, 44-100% bypass rate).
    """
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cf", "Cc", "Co")
        or ch in ("\n", "\r", "\t")
    )


def _tokenize(text: str) -> set[str]:
    """Normalize, lowercase, strip punctuation, split on whitespace.

    Used by rag, ragas, agentic, coherence, and conversation modules to
    normalise text before token-overlap computations.  Applies NFKC
    normalization and zero-width character stripping before tokenizing.
    """
    text = _normalize(text)
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())
