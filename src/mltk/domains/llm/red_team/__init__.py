"""Red team security testing for LLM applications.

Provides a curated attack catalog, encoding mutation engine,
and four pytest-native assertions for adversarial evaluation
of large language models. Covers OWASP LLM Top 10 categories
LLM01 (Prompt Injection), LLM02 (Data Extraction), LLM06
(Excessive Agency), LLM07 (System Prompt Theft), and LLM09
(Harmful Content).

All payloads are educational/demonstrative -- designed for
automated security testing, not malicious use.
"""

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

__all__ = [
    "AttackCategory",
    "AttackPayload",
    "RedTeamResult",
    "assert_red_team_resilient",
    "assert_no_session_jailbreak",
    "assert_owasp_llm_coverage",
    "assert_encoding_mutation_resilience",
]
