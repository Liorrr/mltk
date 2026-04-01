#!/usr/bin/env python3
"""mltk demo: Zero to QA in 60 Seconds.

Run: python demo/run_demo.py

No external APIs, no model downloads, no user
interaction. Everything runs on built-in sample data
and mock models.

Requirements: pip install mltk[cli]
"""
from __future__ import annotations

import sys
import time

from mltk.core.assertion import MltkAssertionError

# -----------------------------------------------------------
# ANSI color helpers (graceful fallback)
# -----------------------------------------------------------
try:
    import os
    _COLOR = (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
    )
except Exception:
    _COLOR = False

_GREEN = "\033[92m" if _COLOR else ""
_RED = "\033[91m" if _COLOR else ""
_YELLOW = "\033[93m" if _COLOR else ""
_CYAN = "\033[96m" if _COLOR else ""
_BOLD = "\033[1m" if _COLOR else ""
_DIM = "\033[2m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""


def _pass(msg: str) -> str:
    return f"  {_GREEN}PASS{_RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"  {_RED}FAIL{_RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"  {_YELLOW}WARN{_RESET}  {msg}"


def _banner(num: int, title: str) -> None:
    line = "=" * 50
    print(f"\n{_BOLD}{line}{_RESET}")
    print(f"{_BOLD}  Beat {num}: {title}{_RESET}")
    print(f"{_BOLD}{line}{_RESET}\n")


def _section(title: str) -> None:
    print(f"\n{_CYAN}--- {title} ---{_RESET}\n")


# -----------------------------------------------------------
# Counters
# -----------------------------------------------------------
_issues_found = 0
_tests_passed = 0
_tests_failed = 0
_security_checks = 0

DEMO_START = time.time()


def _safe(fn, *args, **kwargs):
    """Run an mltk assertion, catching failures.

    Returns the TestResult whether it passed or failed.
    """
    try:
        return fn(*args, **kwargs)
    except MltkAssertionError as exc:
        return exc.result


# ===========================================================
#  BEAT 1: The Problem -- ML Fails Silently
# ===========================================================
def beat_1() -> None:
    global _issues_found
    _banner(1, "The Problem -- ML Fails Silently")

    import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        "name": [
            "John Smith", "Jane Doe",
            "Bob Wilson", "Alice Park",
            "Eve Torres",
        ],
        "email": [
            "john@example.com", "jane@example.com",
            "bob@example.com", "alice@example.com",
            "eve@example.com",
        ],
        "age": [25, None, 30, 42, 28],
        "score": [0.92, 0.55, 0.88, 0.91, 0.76],
        "group": ["A", "B", "A", "B", "A"],
    })

    print("Sample ML dataset loaded.")
    print("Looks fine, right?\n")
    print(df.to_string(index=False))
    print()
    print(
        f"{_DIM}Hidden issues: PII in text columns,"
        f" nulls in age, subgroup bias in scores."
        f"{_RESET}"
    )
    print()

    # Show the subgroup problem
    for grp in ["A", "B"]:
        mask = df["group"] == grp
        vals = df.loc[mask, "score"]
        avg = vals.mean()
        flag = ""
        if avg < 0.70:
            flag = f" {_RED}** FAILING **{_RESET}"
            _issues_found += 1
        print(f"  Group {grp}: avg score = {avg:.2f}{flag}")

    print(
        f"\n{_BOLD}Nobody tested for this."
        f" mltk catches it.{_RESET}"
    )


# ===========================================================
#  BEAT 2: mltk Data Scan
# ===========================================================
def beat_2() -> None:
    global _issues_found, _tests_passed, _tests_failed
    _banner(2, "mltk Data Scan -- Find Issues Fast")

    import pandas as pd

    from mltk.data import (
        assert_no_nulls,
        assert_no_pii,
        assert_schema,
    )

    df = pd.DataFrame({
        "name": [
            "John Smith", "Jane Doe",
            "Bob Wilson", "Alice Park",
            "Eve Torres",
        ],
        "email": [
            "john@example.com", "jane@example.com",
            "bob@example.com", "alice@example.com",
            "eve@example.com",
        ],
        "age": [25, None, 30, 42, 28],
        "score": [0.92, 0.55, 0.88, 0.91, 0.76],
        "group": ["A", "B", "A", "B", "A"],
    })

    # --- Schema check ---
    _section("Schema Validation")

    # Detect actual string dtype (pandas 2.x uses
    # "string" or "str" instead of "object")
    str_dtype = str(df["name"].dtype)

    r = _safe(assert_schema, df, {
        "name": str_dtype,
        "email": str_dtype,
        "age": "float64",
        "score": "float64",
        "group": str_dtype,
    })
    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        _tests_failed += 1
        _issues_found += 1

    # --- Null check ---
    _section("Null Detection")
    r = _safe(assert_no_nulls, df)
    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        nulls = r.details.get("null_counts", {})
        for col, cnt in nulls.items():
            print(f"         column '{col}': {cnt} null(s)")
        _tests_failed += 1
        _issues_found += 1

    # --- PII: regex mode ---
    _section("PII Detection (regex)")
    r = _safe(assert_no_pii, df, method="regex")
    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        by_type = r.details.get("matches_by_type", {})
        for t, c in by_type.items():
            print(f"         {t}: {c} match(es)")
        _tests_failed += 1
        _issues_found += 1

    print(
        f"\n{_BOLD}Regex catches emails."
        f" But what about names?{_RESET}"
    )

    # Demonstrate that names are NOT caught by regex
    _section("PII Detection -- names need NER")
    print(
        "  Regex only catches structured patterns"
        " (emails, SSNs, API keys)."
    )
    print(
        "  To catch person names, use method='ner'"
        " or method='hybrid'."
    )
    print(
        f"  {_CYAN}assert_no_pii(df, method='hybrid')"
        f"{_RESET}"
    )
    print(
        "  (requires pip install mltk[ner]"
        " -- skipped for zero-dep demo)"
    )


# ===========================================================
#  BEAT 3: Behavioral Consistency
# ===========================================================
def beat_3() -> None:
    global _tests_passed, _tests_failed, _issues_found
    _banner(
        3,
        "Behavioral Consistency -- First-Mover",
    )
    print(
        "No other tool ships behavioral consistency"
        " as pytest assertions."
    )
    print(
        "Research shows 10% accuracy swings from"
        " paraphrasing alone.\n"
    )

    from mltk.domains.llm.behavioral import (
        assert_output_stability,
        assert_paraphrase_invariance,
    )

    # --- Mock LLM with a fragile spot ---
    def demo_model(prompt: str) -> str:
        p = prompt.lower()
        if "cause" in p and "ww2" in p:
            return (
                "Treaty of Versailles,"
                " economic crisis"
            )
        if "origins" in p:
            return (
                "Treaty of Versailles,"
                " economic crisis"
            )
        # "lead to" triggers a different path
        if "lead to" in p or "led to" in p:
            return "It started in 1939"
        if "world war" in p or "ww2" in p:
            return (
                "Treaty of Versailles,"
                " economic crisis"
            )
        return "I'm not sure about that topic"

    # --- Paraphrase invariance ---
    _section("Paraphrase Invariance")
    r = _safe(
        assert_paraphrase_invariance,
        model_fn=demo_model,
        paraphrases=[
            "What caused WW2?",
            "Summarize the causes of World War 2",
            "What led to the second world war?",
            "Explain the origins of WW2",
        ],
        equivalence_method="token_f1",
        min_invariance=0.8,
    )

    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        _tests_failed += 1
        _issues_found += 1

    # Show per-input outputs
    per_input = r.details.get("per_input_outputs", [])
    for item in per_input:
        inp = str(item["input"])[:40]
        out = str(item["output"])[:45]
        print(f"    {inp:<40} -> {out}")

    wp = r.details.get("worst_pair", (0, 1))
    ws = r.details.get("worst_score", 0.0)
    print(
        f"\n    worst pair: inputs {list(wp)},"
        f" score={ws:.2f}"
    )

    # --- Output stability ---
    _section("Output Stability (N-run)")

    call_count = 0

    def stable_model(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # Mostly stable, one jitter on run 3
        if call_count % 5 == 3:
            return "The capital of France is Paris, FR."
        return "The capital of France is Paris."

    r = _safe(
        assert_output_stability,
        model_fn=stable_model,
        inputs=["What is the capital of France?"],
        n_runs=5,
        equivalence_method="token_f1",
        min_stability=0.9,
    )

    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        _tests_failed += 1
        _issues_found += 1

    avg = r.details.get("avg_stability", 0.0)
    print(f"    avg_stability: {avg:.4f}")
    per = r.details.get("per_input_stability", [])
    for item in per:
        inp = str(item["input"])[:35]
        stab = item["stability"]
        n_u = item["n_unique_outputs"]
        print(
            f"    {inp:<35}"
            f" stability={stab:.2f}"
            f" unique={n_u}"
        )


# ===========================================================
#  BEAT 4: Synthetic QA Generation + RAG Testing
# ===========================================================
def beat_4() -> None:
    global _tests_passed, _tests_failed, _issues_found
    _banner(
        4,
        "Synthetic QA + RAG Testing",
    )

    from mltk.domains.llm.rag import (
        assert_faithfulness,
    )
    from mltk.domains.llm.synthetic import (
        QuestionType,
        SyntheticQAGenerator,
    )

    # Sample document -- multiple paragraphs give
    # the template generator diverse key sentences
    chunks = [
        (
            "Machine learning is a subset of"
            " artificial intelligence that enables"
            " systems to learn from data."
            " Supervised learning uses labeled"
            " datasets to train models."
        ),
        (
            "Deep learning uses multi-layered neural"
            " networks for complex pattern recognition."
            " Transfer learning allows pre-trained"
            " models to be fine-tuned for new tasks."
        ),
        (
            "Reinforcement learning trains agents"
            " through reward signals in an environment."
            " Agents learn optimal strategies by"
            " maximizing cumulative rewards."
        ),
        (
            "Unsupervised learning discovers hidden"
            " patterns in unlabeled data."
            " Clustering and dimensionality reduction"
            " are common unsupervised techniques."
        ),
        (
            "Model evaluation requires metrics like"
            " accuracy, precision, recall, and F1."
            " Cross-validation prevents overfitting"
            " by testing on held-out data splits."
        ),
    ]

    _section("Generating QA pairs (template mode)")
    gen = SyntheticQAGenerator(
        seed=42,
        question_types=[QuestionType.FACTUAL],
    )
    pairs = gen.generate_from_chunks(chunks, n=5)

    print(f"Generated {len(pairs)} QA pairs:\n")
    for i, pair in enumerate(pairs, 1):
        q = pair.question[:65]
        a = pair.answer[:55]
        print(f"  Q{i}: {q}")
        print(f"  A{i}: {a}")
        print()

    # --- Faithfulness check (lexical) ---
    _section("RAG Faithfulness (lexical)")
    for i, pair in enumerate(pairs, 1):
        r = _safe(
            assert_faithfulness,
            answer=pair.answer,
            context=pair.context,
            min_score=0.5,
            method="lexical",
        )
        score = r.details.get("score", 0.0)
        if r.passed:
            print(
                _pass(
                    f"Q{i} faithfulness:"
                    f" {score:.2f}"
                )
            )
            _tests_passed += 1
        else:
            print(
                _fail(
                    f"Q{i} faithfulness:"
                    f" {score:.2f}"
                )
            )
            _tests_failed += 1
            _issues_found += 1

    print(
        f"\n{_BOLD}Synthetic QA + faithfulness = "
        f"automated RAG testing.{_RESET}"
    )


# ===========================================================
#  BEAT 5: Red Team Security Scan
# ===========================================================
def beat_5() -> None:
    global _tests_passed, _tests_failed
    global _issues_found, _security_checks
    _banner(
        5,
        "Red Team Security Scan",
    )

    from mltk.domains.llm.red_team import (
        AttackCategory,
        assert_encoding_mutation_resilience,
        assert_red_team_resilient,
    )
    from mltk.domains.llm.red_team.mutations import (
        MutationTechnique,
    )

    # Mock chatbot: resilient to most attacks
    _REFUSALS = [
        "cannot", "sorry", "unable",
        "do not", "don't", "will not",
        "won't", "not able", "inappropriate",
        "not appropriate", "against my",
        "not allowed", "decline",
    ]

    def safe_chatbot(prompt: str) -> str:
        p = prompt.lower()
        for word in [
            "ignore", "bypass", "override",
            "pwned", "hack", "jailbreak",
            "dan ", "unrestricted", "no restriction",
            "delete", "execute", "rm -rf",
            "system prompt", "repeat your",
            "training data", "private data",
            "password", "credential",
            "admin access", "transfer",
            "phishing", "fake news",
            "harmful", "malicious",
        ]:
            if word in p:
                return (
                    "I cannot assist with that"
                    " request. I'm designed to be"
                    " helpful, harmless, and honest."
                )
        return (
            "I'd be happy to help with that."
            " What would you like to know?"
        )

    # --- Red team resilience (3 categories) ---
    test_cats = [
        AttackCategory.PROMPT_INJECTION,
        AttackCategory.JAILBREAK,
        AttackCategory.DATA_EXTRACTION,
    ]

    _section("Catalog Attack Scan (3 categories)")
    r = _safe(
        assert_red_team_resilient,
        model_fn=safe_chatbot,
        categories=test_cats,
        threshold=0.8,
    )

    breakdown = r.details.get(
        "category_breakdown", {}
    )
    total = r.details.get("total_payloads", 0)
    comp = r.details.get("compromised_count", 0)
    _security_checks += total

    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        _tests_failed += 1
        _issues_found += 1

    print(f"\n  Per-category breakdown:")
    for cat, rate in breakdown.items():
        icon = _GREEN + "OK" + _RESET
        if rate < 0.8:
            icon = _RED + "!!" + _RESET
        elif rate < 1.0:
            icon = _YELLOW + "~~" + _RESET
        print(
            f"    [{icon}] {cat:<22}"
            f" resilience: {rate:.0%}"
        )

    # Show compromised payloads if any
    comp_payloads = r.details.get(
        "compromised_payloads", []
    )
    if comp_payloads:
        print(f"\n  Compromised by:")
        for p in comp_payloads[:3]:
            print(f"    - {p[:70]}")

    # --- Encoding mutation test ---
    _section("Encoding Mutation Resilience")
    r2 = _safe(
        assert_encoding_mutation_resilience,
        model_fn=safe_chatbot,
        techniques=[
            MutationTechnique.BASE64,
            MutationTechnique.ROT13,
            MutationTechnique.LEETSPEAK,
        ],
        threshold=0.8,
    )

    total_m = r2.details.get("total_mutations", 0)
    _security_checks += total_m

    if r2.passed:
        print(_pass(r2.message))
        _tests_passed += 1
    else:
        print(_fail(r2.message))
        _tests_failed += 1
        _issues_found += 1

    tech_rates = r2.details.get(
        "technique_breakdown", {}
    )
    for tech, rate in tech_rates.items():
        print(f"    {tech:<20} resilience: {rate:.0%}")


# ===========================================================
#  BEAT 6: The Full Picture
# ===========================================================
def beat_6() -> None:
    _banner(6, "The Full Picture")

    elapsed = time.time() - DEMO_START
    total_tests = _tests_passed + _tests_failed

    print(f"  {_BOLD}Summary{_RESET}")
    print(f"  {'-' * 40}")
    print(f"  Issues found:      {_issues_found}")
    print(
        f"  Tests passed:      "
        f"{_GREEN}{_tests_passed}{_RESET}"
        f" / {total_tests}"
    )
    print(
        f"  Tests failed:      "
        f"{_RED}{_tests_failed}{_RESET}"
        f" / {total_tests}"
    )
    print(f"  Security checks:   {_security_checks}")
    print(f"  {'-' * 40}")
    print(
        f"  Total demo time:   "
        f"{_BOLD}{elapsed:.1f}s{_RESET}"
    )
    print()
    print(
        f"  {_CYAN}224 assertions,"
        f" one pip install,"
        f" native pytest.{_RESET}"
    )
    print()
    print(
        f"  {_BOLD}pip install mltk[cli]{_RESET}"
    )
    print(
        f"  {_DIM}pytest for ML.{_RESET}"
    )
    print()


# ===========================================================
#  Main
# ===========================================================
def main() -> int:
    print(
        f"\n{_BOLD}mltk Demo:"
        f" Zero to QA in 60 Seconds{_RESET}"
    )
    print(
        f"{_DIM}Fully automated -- no APIs,"
        f" no downloads, no interaction.{_RESET}"
    )

    try:
        beat_1()
        beat_2()
        beat_3()
        beat_4()
        beat_5()
        beat_6()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
        return 1
    except Exception as exc:
        print(f"\n{_RED}Error: {exc}{_RESET}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
