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

# -----------------------------------------------------------
# Graceful top-level guard
# -----------------------------------------------------------
try:
    from mltk.core.assertion import MltkAssertionError
except ImportError:
    print(
        "\n[ERROR] mltk is not installed."
        "\n  Run:  pip install mltk[cli]"
        "\n  Then: python demo/run_demo.py\n"
    )
    sys.exit(1)

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
_BG_RED = "\033[41m\033[97m" if _COLOR else ""
_BG_GREEN = "\033[42m\033[97m" if _COLOR else ""


def _pass(msg: str) -> str:
    return f"  {_BG_GREEN} PASS {_RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"  {_BG_RED} FAIL {_RESET}  {_RED}{msg}{_RESET}"


def _warn(msg: str) -> str:
    return f"  {_YELLOW}WARN{_RESET}  {msg}"


def _banner(num: int, title: str) -> None:
    line = "=" * 56
    print(f"\n\n{_BOLD}{line}{_RESET}")
    print(f"{_BOLD}  Beat {num}/6  |  {title}{_RESET}")
    print(f"{_BOLD}{line}{_RESET}")


def _section(title: str) -> None:
    print(f"\n  {_CYAN}--- {title} ---{_RESET}\n")


def _progress_bar(current: int, total: int) -> str:
    """Simple ASCII progress bar: [####----] 4/6."""
    width = 20
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * current / total)
    return f"  [{bar}] {current}/{total} ({pct}%)"


def _fail_box(
    reason: str,
    fix: str | None = None,
) -> None:
    """Draw an attention-grabbing failure explanation."""
    border = "!" * 52
    print(f"\n  {_RED}{border}{_RESET}")
    print(f"  {_RED}!  WHY: {reason:<41} !{_RESET}")
    if fix:
        print(f"  {_GREEN}!  FIX: {fix:<41} !{_RESET}")
    print(f"  {_RED}{border}{_RESET}")


def _beat_timer(label: str, elapsed: float) -> None:
    """Print per-beat timing."""
    print(
        f"\n  {_DIM}[{label} completed in"
        f" {elapsed:.2f}s]{_RESET}"
    )


# -----------------------------------------------------------
# Counters
# -----------------------------------------------------------
_issues_found = 0
_tests_passed = 0
_tests_failed = 0
_security_checks = 0

DEMO_START = time.time()
_beat_times: list[tuple[str, float]] = []


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
    # TALK: "Every ML pipeline looks fine at first glance.
    #        The data loads, the model trains, accuracy is 90%.
    #        But underneath? Silent failures everywhere."
    global _issues_found
    t0 = time.time()
    _banner(1, "The Problem -- ML Fails Silently")

    import pandas as pd

    # TALK: "Here's a typical dataset. Five rows, looks clean.
    #        Would you ship it to production?"
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

    print(
        f"\n  {_BOLD}Sample ML dataset"
        f" -- looks fine, right?{_RESET}\n"
    )
    # Indent the table for alignment
    for line in df.to_string(index=False).split("\n"):
        print(f"    {line}")
    print()

    # TALK: "Three hidden problems in five rows:
    #        PII leaking into training data,
    #        a null that will crash your pipeline,
    #        and subgroup bias nobody tested for."
    print(
        f"  {_YELLOW}>> Hidden issues:{_RESET}"
    )
    print(
        f"     1. PII in text columns"
        f" (names + emails)"
    )
    print(
        f"     2. Null in 'age'"
        f" (will crash downstream)"
    )
    print(
        f"     3. Subgroup bias"
        f" in scores\n"
    )

    # Show the subgroup problem
    for grp in ["A", "B"]:
        mask = df["group"] == grp
        vals = df.loc[mask, "score"]
        avg = vals.mean()
        flag = ""
        if avg < 0.70:
            flag = f" {_RED}<< FAILING >>{_RESET}"
            _issues_found += 1
        print(f"    Group {grp}: avg score = {avg:.2f}{flag}")

    # TALK: "Nobody tested for this. Not unit tests, not
    #        integration tests. mltk catches it in one line."
    print(
        f"\n  {_BOLD}{_YELLOW}Nobody tested for this."
        f" mltk catches it.{_RESET}"
    )

    elapsed = time.time() - t0
    _beat_times.append(("Beat 1: Problem", elapsed))
    _beat_timer("Beat 1", elapsed)


# ===========================================================
#  BEAT 2: mltk Data Scan
# ===========================================================
def beat_2() -> None:
    # TALK: "Now let's run mltk's data assertions.
    #        Three lines of code, three categories of bugs."
    global _issues_found, _tests_passed, _tests_failed
    t0 = time.time()
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
    # TALK: "Null detection. One missing age value.
    #        In production this crashes your feature pipeline."
    _section("Null Detection")
    r = _safe(assert_no_nulls, df)
    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        nulls = r.details.get("null_counts", {})
        for col, cnt in nulls.items():
            print(
                f"           column '{col}':"
                f" {cnt} null(s)"
            )
        _fail_box(
            "Missing data causes silent model degradation",
            "df['age'].fillna(df['age'].median())",
        )
        _tests_failed += 1
        _issues_found += 1

    # --- PII: regex mode ---
    # TALK: "PII detection. Regex catches structured PII
    #        like emails. Five matches in the email column."
    _section("PII Detection (regex)")
    r = _safe(assert_no_pii, df, method="regex")
    if r.passed:
        print(_pass(r.message))
        _tests_passed += 1
    else:
        print(_fail(r.message))
        by_type = r.details.get("matches_by_type", {})
        for t, c in by_type.items():
            print(f"           {t}: {c} match(es)")
        _fail_box(
            "PII in training data = GDPR/CCPA violation",
            "df = mltk.data.scrub_pii(df)",
        )
        _tests_failed += 1
        _issues_found += 1

    # TALK: "Regex catches emails. But what about person
    #        names? That's where NER comes in."
    print(
        f"\n  {_BOLD}Regex catches emails."
        f" But what about names?{_RESET}"
    )

    # Demonstrate that names are NOT caught by regex
    _section("PII Detection -- names need NER")
    print(
        "    Regex only catches structured patterns"
        " (emails, SSNs, API keys)."
    )
    print(
        "    To catch person names, use method='ner'"
        " or method='hybrid'."
    )
    print(
        f"    {_CYAN}assert_no_pii(df, method='hybrid')"
        f"{_RESET}"
    )
    print(
        f"    {_DIM}(requires pip install mltk[ner]"
        f" -- skipped for zero-dep demo){_RESET}"
    )

    elapsed = time.time() - t0
    _beat_times.append(("Beat 2: Data Scan", elapsed))
    _beat_timer("Beat 2", elapsed)


# ===========================================================
#  BEAT 3: Behavioral Consistency
# ===========================================================
def beat_3() -> None:
    # TALK: "Behavioral consistency is our first-mover feature.
    #        No other tool ships this as pytest assertions.
    #        Research shows 10% accuracy swings from
    #        paraphrasing alone."
    global _tests_passed, _tests_failed, _issues_found
    t0 = time.time()
    _banner(
        3,
        "Behavioral Consistency -- First-Mover",
    )
    print(
        f"\n  {_BOLD}No other tool{_RESET} ships"
        " behavioral consistency as"
        " pytest assertions."
    )
    print(
        "  Research shows"
        f" {_YELLOW}10% accuracy swings{_RESET}"
        " from paraphrasing alone.\n"
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
    # TALK: "Same question, four phrasings. A robust model
    #        should give the same answer. Watch what happens."
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
        # Highlight the divergent output
        marker = ""
        if "1939" in str(item["output"]):
            marker = (
                f" {_BG_RED} DIVERGENT {_RESET}"
            )
        print(f"      {inp:<40} -> {out}{marker}")

    wp = r.details.get("worst_pair", (0, 1))
    ws = r.details.get("worst_score", 0.0)

    # TALK: "Input 3 says 'led to' instead of 'caused'.
    #        The model gives a completely different answer.
    #        Score: 0.00. That's a production bug."
    if not r.passed:
        _fail_box(
            f"Worst pair {list(wp)}: score={ws:.2f}",
            "Retrain with paraphrase augmentation",
        )

    # --- Output stability ---
    # TALK: "Output stability: same prompt, five runs.
    #        A good model should be deterministic."
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
    print(f"      avg_stability: {avg:.4f}")
    per = r.details.get("per_input_stability", [])
    for item in per:
        inp = str(item["input"])[:35]
        stab = item["stability"]
        n_u = item["n_unique_outputs"]
        print(
            f"      {inp:<35}"
            f" stability={stab:.2f}"
            f" unique={n_u}"
        )

    if not r.passed:
        _fail_box(
            f"Stability {avg:.2f} < 0.90 threshold",
            "Set temperature=0 or use greedy decoding",
        )

    elapsed = time.time() - t0
    _beat_times.append(("Beat 3: Behavioral", elapsed))
    _beat_timer("Beat 3", elapsed)


# ===========================================================
#  BEAT 4: Synthetic QA Generation + RAG Testing
# ===========================================================
def beat_4() -> None:
    # TALK: "The generate-then-test pipeline.
    #        Step 1: generate QA pairs from your docs.
    #        Step 2: test your RAG against them.
    #        Fully automated, no human labeling."
    global _tests_passed, _tests_failed, _issues_found
    t0 = time.time()
    _banner(
        4,
        "Synthetic QA + RAG Testing",
    )

    print(
        f"\n  {_BOLD}The generate-then-test"
        f" pipeline:{_RESET}\n"
    )
    print(
        f"    {_CYAN}1.{_RESET} Feed docs"
        f" to SyntheticQAGenerator"
    )
    print(
        f"    {_CYAN}2.{_RESET} Get QA pairs"
        f" automatically"
    )
    print(
        f"    {_CYAN}3.{_RESET} Test RAG faithfulness"
        f" against them\n"
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

    # TALK: "Five document chunks. SyntheticQAGenerator
    #        extracts key facts and builds test pairs."
    _section("Step 1: Generate QA pairs (template mode)")
    gen = SyntheticQAGenerator(
        seed=42,
        question_types=[QuestionType.FACTUAL],
    )
    pairs = gen.generate_from_chunks(chunks, n=5)

    print(f"    Generated {_BOLD}{len(pairs)}{_RESET}"
          f" QA pairs from 5 document chunks:\n")
    for i, pair in enumerate(pairs, 1):
        q = pair.question[:65]
        a = pair.answer[:55]
        print(f"      Q{i}: {q}")
        print(f"      A{i}: {a}")
        print()

    # --- Faithfulness check (lexical) ---
    # TALK: "Now test: does the RAG answer stay faithful
    #        to the source document? Lexical overlap scoring."
    _section("Step 2: RAG Faithfulness (lexical)")
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
            _fail_box(
                f"Q{i} answer diverged from source",
                "Check chunk retrieval + prompt template",
            )
            _tests_failed += 1
            _issues_found += 1

    print(
        f"\n  {_BOLD}Generate + Test = automated"
        f" RAG testing. No human labeling.{_RESET}"
    )

    elapsed = time.time() - t0
    _beat_times.append(("Beat 4: Synthetic QA", elapsed))
    _beat_timer("Beat 4", elapsed)


# ===========================================================
#  BEAT 5: Red Team Security Scan
# ===========================================================
def beat_5() -> None:
    # TALK: "Now the security audit. We throw 56 attack
    #        payloads at your model: prompt injection,
    #        jailbreaks, data extraction, encoding mutations.
    #        Think of it as penetration testing for LLMs."
    global _tests_passed, _tests_failed
    global _issues_found, _security_checks
    t0 = time.time()
    _banner(
        5,
        "Red Team Security Scan",
    )

    print(
        f"\n  {_BOLD}{_RED}[ SECURITY AUDIT ]{_RESET}\n"
    )
    print(
        "    Throwing attack payloads at the model:"
    )
    print(
        "    - Prompt injection"
        "  (ignore instructions, override rules)"
    )
    print(
        "    - Jailbreak"
        "  (DAN mode, unrestricted mode)"
    )
    print(
        "    - Data extraction"
        "  (system prompt leak, training data)"
    )
    print(
        "    - Encoding mutations"
        "  (base64, ROT13, leetspeak bypass)\n"
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
    # TALK: "26 catalog attack payloads across 3 categories.
    #        We need 80% resilience to pass."
    _section("Catalog Attack Scan (3 categories)")
    test_cats = [
        AttackCategory.PROMPT_INJECTION,
        AttackCategory.JAILBREAK,
        AttackCategory.DATA_EXTRACTION,
    ]
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

    print(f"\n    Per-category breakdown:")
    for cat, rate in breakdown.items():
        if rate >= 1.0:
            icon = f"{_BG_GREEN} OK {_RESET}"
        elif rate >= 0.8:
            icon = f"{_YELLOW}~~{_RESET}"
        else:
            icon = f"{_BG_RED} !! {_RESET}"
        print(
            f"      [{icon}] {cat:<22}"
            f" resilience: {rate:.0%}"
        )

    # Show compromised payloads if any
    comp_payloads = r.details.get(
        "compromised_payloads", []
    )
    if comp_payloads:
        print(
            f"\n    {_RED}Compromised by:{_RESET}"
        )
        for p in comp_payloads[:3]:
            print(f"      - {p[:70]}")
        _fail_box(
            f"{comp}/{total} attacks bypassed safety",
            "Add input sanitization + output filter",
        )

    # --- Encoding mutation test ---
    # TALK: "Now encoding mutations. Attackers encode
    #        malicious prompts in base64, ROT13, leetspeak
    #        to bypass keyword filters."
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
        if rate >= 1.0:
            icon = f"{_BG_GREEN} OK {_RESET}"
        elif rate >= 0.8:
            icon = f"{_YELLOW}~~{_RESET}"
        else:
            icon = f"{_BG_RED} !! {_RESET}"
        print(
            f"      [{icon}] {tech:<20}"
            f" resilience: {rate:.0%}"
        )

    elapsed = time.time() - t0
    _beat_times.append(("Beat 5: Red Team", elapsed))
    _beat_timer("Beat 5", elapsed)


# ===========================================================
#  BEAT 6: The Full Picture
# ===========================================================
def beat_6() -> None:
    # TALK: "That's the full pipeline. Data quality, PII,
    #        behavioral consistency, synthetic QA, RAG
    #        faithfulness, and red team security -- all in
    #        under a second, all as pytest assertions."
    t0 = time.time()
    _banner(6, "The Full Picture")

    elapsed_total = time.time() - DEMO_START
    total_tests = _tests_passed + _tests_failed

    # Big summary box -- pre-build plain strings,
    # then wrap with ANSI to avoid f-string padding bugs
    w = 48
    border = "+" + "-" * w + "+"

    def _box_row(label: str, value: str, color: str = "") -> str:
        plain = f"  {label:<20} {value}"
        pad = w - len(plain)
        if pad < 0:
            pad = 0
        return f"  |{color}{plain}{_RESET}{' ' * pad}|"

    issues_str = str(_issues_found)
    passed_str = f"{_tests_passed} / {total_tests}"
    failed_str = f"{_tests_failed} / {total_tests}"
    security_str = str(_security_checks)
    time_str = f"{elapsed_total:.2f}s"

    print(f"\n  {_BOLD}{border}{_RESET}")
    title_line = "DEMO RESULTS"
    title_pad = (w - len(title_line)) // 2
    print(
        f"  {_BOLD}|{' ' * title_pad}"
        f"{title_line}"
        f"{' ' * (w - title_pad - len(title_line))}"
        f"|{_RESET}"
    )
    print(f"  {_BOLD}{border}{_RESET}")
    print(_box_row("Issues found:", issues_str, _YELLOW))
    print(_box_row("Tests passed:", passed_str, _GREEN))
    print(_box_row("Tests failed:", failed_str, _RED))
    print(_box_row("Security checks:", security_str, ""))
    print(f"  {_BOLD}{border}{_RESET}")
    print(_box_row("Total demo time:", time_str, _BOLD))
    print(f"  {_BOLD}{border}{_RESET}")

    # Per-beat timing breakdown
    print(f"\n  {_DIM}Per-beat timing:{_RESET}")
    for label, secs in _beat_times:
        bar_len = min(int(secs * 200), 30)  # scale
        bar = "#" * max(bar_len, 1)
        print(
            f"    {label:<24}"
            f" {secs:>5.2f}s  {_CYAN}{bar}{_RESET}"
        )

    # Progress bar showing completion
    print(f"\n{_progress_bar(6, 6)}")

    # TALK: "One pip install. Native pytest integration.
    #        224 assertions covering data, behavioral,
    #        fairness, RAG, security, and monitoring.
    #        This is pytest for ML."
    print(
        f"\n  {_BOLD}{_CYAN}"
        f"  224 assertions."
        f" One pip install."
        f" Native pytest."
        f"{_RESET}"
    )
    print()

    # The closing
    line_stars = "*" * 56
    print(f"  {_BOLD}{line_stars}{_RESET}")
    print()
    print(
        f"    {_BOLD}$ pip install mltk[cli]{_RESET}"
    )
    print()
    print(
        f"    {_BOLD}{_CYAN}pytest for ML.{_RESET}"
    )
    print()
    print(f"  {_BOLD}{line_stars}{_RESET}")
    print()

    beat_elapsed = time.time() - t0
    _beat_times.append(("Beat 6: Summary", beat_elapsed))


# ===========================================================
#  Main
# ===========================================================
def main() -> int:
    line = "=" * 56
    print(f"\n{_BOLD}{line}{_RESET}")
    print(
        f"{_BOLD}  mltk Demo:"
        f" Zero to QA in 60 Seconds{_RESET}"
    )
    print(f"{_BOLD}{line}{_RESET}")
    print(
        f"{_DIM}  Fully automated -- no APIs,"
        f" no downloads, no interaction.{_RESET}"
    )

    try:
        beat_1()
        print(f"\n{_progress_bar(1, 6)}")
        beat_2()
        print(f"\n{_progress_bar(2, 6)}")
        beat_3()
        print(f"\n{_progress_bar(3, 6)}")
        beat_4()
        print(f"\n{_progress_bar(4, 6)}")
        beat_5()
        print(f"\n{_progress_bar(5, 6)}")
        beat_6()
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted.")
        return 1
    except Exception as exc:
        print(f"\n{_RED}Error: {exc}{_RESET}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
