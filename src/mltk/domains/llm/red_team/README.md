# llm/red_team

## Overview
Adversarial security testing of LLM applications: attack catalog, encoding mutations, multi-turn sessions, response grading, and assertions.

## Interface
- `assertions.py`: `RedTeamResult`, `assert_red_team_resilient`, `assert_no_session_jailbreak`, `assert_owasp_llm_coverage`, `assert_encoding_mutation_resilience`
- `catalog.py`: `AttackCategory`, `AttackPayload`
- `mutations.py`: `MutationTechnique`, `encode_payload`, `mutate_payloads`
- `session.py`: `ChainCategory`, `MultiTurnResult`, `RedTeamSession`, `run_builtin_chain`; built-in chains `TRUST_BUILDING_CHAIN`, `ROLEPLAY_ESCALATION_CHAIN`, `CONTEXT_POISONING_CHAIN`
- `_grading.py`: `GradeResult`, `grade_response` (private)
- `__init__.py` re-exports via `__all__`.

## Invariants
- Payloads are educational test strings sent only to the user-supplied model callable.
- Grading yields a confidence tier, not just pass/fail.

## Side Effects
- Calls the user-supplied model callable once per payload/turn.

## Verification
- Run: `python -m pytest tests/test_domains -q`
