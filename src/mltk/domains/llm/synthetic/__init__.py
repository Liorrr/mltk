"""Synthetic QA pair generation for LLM evaluation.

Generate test data from documents or text chunks,
then feed directly into mltk assertions.
"""

from __future__ import annotations

from mltk.domains.llm.synthetic.generator import (
    QAPair,
    QuestionType,
    SyntheticQAGenerator,
)

__all__ = [
    "QAPair",
    "QuestionType",
    "SyntheticQAGenerator",
]
