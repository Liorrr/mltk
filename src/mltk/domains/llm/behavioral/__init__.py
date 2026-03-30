"""Behavioral consistency testing for LLMs.

Test whether models produce consistent outputs across:
- Paraphrased inputs (same intent, different wording)
- Formatting changes (case, spacing, punctuation)
- Repeated runs (non-determinism detection)
- Retrieval queries (RAG document consistency) -- S70
- Semantic equivalence (NLI bidirectional) -- S70
- Directional expectations (CheckList DIR) -- S70

First-mover: no competitor offers these as pytest assertions.
"""

from __future__ import annotations

from mltk.domains.llm.behavioral.invariance import (
    assert_format_invariance,
    assert_paraphrase_invariance,
)
from mltk.domains.llm.behavioral.paraphrase import (
    ParaphraseGenerator,
)
from mltk.domains.llm.behavioral.retrieval import (
    assert_retrieval_consistency,
)
from mltk.domains.llm.behavioral.semantic import (
    assert_directional_expectation,
    assert_semantic_equivalence,
)
from mltk.domains.llm.behavioral.stability import (
    assert_output_stability,
)

__all__ = [
    "assert_paraphrase_invariance",
    "assert_format_invariance",
    "assert_output_stability",
    "assert_semantic_equivalence",
    "assert_directional_expectation",
    "assert_retrieval_consistency",
    "ParaphraseGenerator",
]
