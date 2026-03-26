"""Private shared utilities for LLM domain modules.

Keep this module minimal — only truly shared, stateless helpers live here.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Used by rag, ragas, agentic, coherence, and conversation modules to
    normalise text before token-overlap computations.
    """
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())
