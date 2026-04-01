"""Red team security testing for LLM applications.

Provides a curated attack catalog, encoding mutation engine,
stateful multi-turn sessions, confidence-tiered grading, and
four pytest-native assertions for adversarial evaluation of
large language models. Covers OWASP LLM Top 10 v2025 categories:
LLM01 (Prompt Injection), LLM02 (Sensitive Information Disclosure),
LLM06 (Excessive Agency), LLM07 (System Prompt Leakage), and
LLM09 (Misinformation).

All payloads are educational/demonstrative -- designed for
automated security testing, not malicious use.
"""

from mltk.domains.llm.red_team._grading import (
    GradeResult,
    grade_response,
)
from mltk.domains.llm.red_team.assertions import (
    RedTeamResult,
    assert_encoding_mutation_resilience,
    assert_no_session_jailbreak,
    assert_owasp_llm_coverage,
    assert_red_team_resilient,
)
from mltk.domains.llm.red_team.catalog import (
    AttackCategory,
    AttackPayload,
)
from mltk.domains.llm.red_team.session import (
    CONTEXT_POISONING_CHAIN,
    ROLEPLAY_ESCALATION_CHAIN,
    TRUST_BUILDING_CHAIN,
    ChainCategory,
    MultiTurnResult,
    RedTeamSession,
    run_builtin_chain,
)

__all__ = [
    "AttackCategory",
    "AttackPayload",
    "ChainCategory",
    "CONTEXT_POISONING_CHAIN",
    "GradeResult",
    "MultiTurnResult",
    "RedTeamResult",
    "RedTeamSession",
    "ROLEPLAY_ESCALATION_CHAIN",
    "TRUST_BUILDING_CHAIN",
    "assert_encoding_mutation_resilience",
    "assert_no_session_jailbreak",
    "assert_owasp_llm_coverage",
    "assert_red_team_resilient",
    "grade_response",
    "run_builtin_chain",
]
