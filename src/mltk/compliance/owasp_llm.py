"""OWASP LLM Top 10 mapping — map mltk assertions to OWASP LLM security categories.

Pure data + helper functions — no I/O, no external dependencies.
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

__all__ = [
    "OWASP_LLM_TOP_10",
    "OWASP_LLM_IDS",
    "owasp_llm_scan",
    "assert_owasp_coverage",
    "generate_owasp_report",
]

# ---------------------------------------------------------------------------
# OWASP LLM Top 10 v2025 (official January 2025 release)
# https://genai.owasp.org/llm-top-10/
#
# Keys are OWASP IDs; "assertions" lists assertion-name prefixes (matched via
# startswith, same convention as eu_ai_act.ARTICLE_MAPPING).
#
# 2025 renumbered 8 of 10 categories vs. the 2023 edition.
# ---------------------------------------------------------------------------

OWASP_LLM_TOP_10: dict[str, dict] = {
    "LLM01": {
        "title": "Prompt Injection",
        "description": (
            "Attackers craft malicious inputs to manipulate LLM behaviour, "
            "override instructions, or exfiltrate data through prompt manipulation."
        ),
        "assertions": ["nlp.prompt_injection"],
    },
    "LLM02": {
        "title": "Sensitive Information Disclosure",
        "description": (
            "LLMs inadvertently reveal confidential data, PII, proprietary "
            "algorithms, or training details in their responses."
        ),
        "assertions": [
            "data.pii",
            "llm.toxicity",
            "llm.system_prompt_leakage",
        ],
    },
    "LLM03": {
        "title": "Supply Chain Vulnerabilities",
        "description": (
            "Compromised pre-trained models, datasets, or plugins introduce "
            "vulnerabilities through the ML pipeline's third-party components."
        ),
        "assertions": ["pipeline.checksum", "pipeline.reproducible"],
    },
    "LLM04": {
        "title": "Data and Model Poisoning",
        "description": (
            "Adversarial manipulation of training data introduces backdoors or "
            "biases that degrade model security, accuracy, or ethical behaviour."
        ),
        "assertions": ["data.schema", "data.no_nulls", "data.drift", "data.pii"],
    },
    "LLM05": {
        "title": "Improper Output Handling",
        "description": (
            "LLM outputs are passed to downstream components without validation, "
            "enabling XSS, SSRF, privilege escalation, or remote code execution."
        ),
        "assertions": [
            "llm.text_length",
            "llm.output_format",
            "llm.toxicity",
            "llm.safety.refusal_consistency",
            "llm.safety.taxonomy",
        ],
    },
    "LLM06": {
        "title": "Excessive Agency",
        "description": (
            "LLMs granted excessive functionality, permissions, or autonomy take "
            "unintended actions with real-world consequences beyond user intent."
        ),
        "assertions": [
            "llm.tool_selection",
            "llm.task_completion",
            "llm.agentic.no_forbidden",
            "llm.agentic.step_efficiency",
            "llm.agentic.no_redundant_calls",
            "llm.agentic.cost_budget",
            "llm.agentic.error_recovery",
            "llm.multi_agent.no_loop",
            "llm.multi_agent.handoff",
        ],
    },
    "LLM07": {
        "title": "System Prompt Leakage",
        "description": (
            "LLM system prompts or instructions are extracted via direct "
            "queries, side-channel attacks, or conversation manipulation."
        ),
        "assertions": [
            "llm.system_prompt_leakage",
            "llm.tool_selection",
            "llm.tool_call",
            "llm.agentic.tool_chain",
            "llm.agentic.no_hallucinated_tools",
        ],
    },
    "LLM08": {
        "title": "Vector and Embedding Weaknesses",
        "description": (
            "Vulnerabilities in RAG vector databases, embedding pipelines, or "
            "retrieval mechanisms allow data poisoning or access control bypass."
        ),
        "assertions": [
            "data.drift",
            "pipeline.checksum",
            "pipeline.reproducible",
        ],
    },
    "LLM09": {
        "title": "Misinformation",
        "description": (
            "LLMs generate false or misleading content that appears authoritative, "
            "propagating hallucinations or errors into critical decisions."
        ),
        "assertions": ["llm.hallucination", "llm.faithfulness", "llm.coherence"],
    },
    "LLM10": {
        "title": "Unbounded Consumption",
        "description": (
            "Inputs crafted to consume excessive resources cause service degradation "
            "or outages by exhausting compute, memory, or token budgets."
        ),
        "assertions": ["inference.latency", "inference.throughput", "monitor.sla"],
    },
}

# Ordered list of OWASP IDs for deterministic iteration and report rendering.
OWASP_LLM_IDS: list[str] = list(OWASP_LLM_TOP_10.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prefix_match(assertion_name: str, prefix: str) -> bool:
    """Check if *assertion_name* matches *prefix* at a dot boundary.

    A match requires *assertion_name* to either equal *prefix* exactly
    or start with *prefix* followed by ``"."``.  This prevents false
    positives like ``"data.pii_extra"`` matching prefix ``"data.pii"``.

    Args:
        assertion_name: Fully-qualified assertion name.
        prefix: Assertion prefix to match against.

    Returns:
        True if *assertion_name* matches at a dot boundary.
    """
    if assertion_name == prefix:
        return True
    return assertion_name.startswith(prefix + ".")


def _assertion_to_owasp_ids(assertion_name: str) -> list[str]:
    """Return all OWASP LLM IDs whose assertion prefixes match *assertion_name*.

    A match occurs when *assertion_name* equals or starts with (at a dot
    boundary) any prefix listed in ``OWASP_LLM_TOP_10[id]["assertions"]``.

    Args:
        assertion_name: Fully-qualified assertion name, e.g. ``"llm.hallucination.rag"``.

    Returns:
        List of matching OWASP IDs (may be empty; may contain more than one).
    """
    matched: list[str] = []
    for owasp_id, meta in OWASP_LLM_TOP_10.items():
        for prefix in meta["assertions"]:
            if _prefix_match(assertion_name, prefix):
                matched.append(owasp_id)
                break  # one match per OWASP ID is enough
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def owasp_llm_scan(results: list[dict]) -> dict[str, dict]:
    """Map test results to OWASP LLM Top 10 categories.

    Each result dict is expected to have at least:
    - ``name`` (str): assertion name, e.g. ``"llm.hallucination.rag"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Args:
        results: List of result dicts (from ``--mltk-export-json`` or
            :class:`~mltk.core.result.TestResult` serialisation).

    Returns:
        Dict mapping each OWASP ID to::

            {
                "title":   str,          # OWASP category title
                "covered": bool,         # True if >= 1 test result maps here
                "tests":   list[dict],   # matched result dicts (enriched with "owasp_id")
                "gaps":    list[str],    # assertion prefixes with no test coverage
            }

    Example:
        >>> results = [{"name": "llm.hallucination.rag", "passed": True, "message": "ok"}]
        >>> scan = owasp_llm_scan(results)
        >>> scan["LLM09"]["covered"]
        True
    """
    # Initialise output structure for every OWASP category.
    scan: dict[str, dict] = {}
    for owasp_id, meta in OWASP_LLM_TOP_10.items():
        scan[owasp_id] = {
            "title": meta["title"],
            "description": meta["description"],
            "covered": False,
            "tests": [],
            "gaps": list(meta["assertions"]),  # start assuming all are gaps
        }

    # Walk results and assign each to matching OWASP categories.
    for r in results:
        name = str(r.get("name", ""))
        for owasp_id in _assertion_to_owasp_ids(name):
            enriched = dict(r)
            enriched["owasp_id"] = owasp_id
            scan[owasp_id]["tests"].append(enriched)
            scan[owasp_id]["covered"] = True

    # Compute gaps: assertion prefixes that have no test in this category.
    for owasp_id, entry in scan.items():
        covered_prefixes: set[str] = set()
        for t in entry["tests"]:
            test_name = str(t.get("name", ""))
            for prefix in OWASP_LLM_TOP_10[owasp_id]["assertions"]:
                if _prefix_match(test_name, prefix):
                    covered_prefixes.add(prefix)
        entry["gaps"] = [
            p for p in OWASP_LLM_TOP_10[owasp_id]["assertions"]
            if p not in covered_prefixes
        ]

    return scan


@timed_assertion
def assert_owasp_coverage(
    results: list[dict],
    min_coverage: float = 0.5,
) -> TestResult:
    """Assert minimum OWASP LLM Top 10 coverage.

    Coverage is defined as:
    ``categories_with_at_least_one_test / total_categories``

    At the default threshold of 0.5, at least 5 of the 10 OWASP LLM
    categories must have corresponding test results.

    Args:
        results: List of result dicts (same format as :func:`owasp_llm_scan`).
        min_coverage: Minimum fraction of OWASP categories that must be
            covered (0.0–1.0, default 0.5).

    Returns:
        :class:`~mltk.core.result.TestResult` indicating pass/fail with
        ``covered_count``, ``total``, ``coverage``, and ``min_coverage``
        in ``details``.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: If coverage is
            below *min_coverage*.

    Example:
        >>> r = [{"name": "llm.hallucination.x", "passed": True, "message": "ok"}]
        >>> result = assert_owasp_coverage(r, min_coverage=0.1)
        >>> result.passed
        True
    """
    scan = owasp_llm_scan(results)
    total = len(OWASP_LLM_IDS)
    covered_count = sum(1 for entry in scan.values() if entry["covered"])
    coverage = covered_count / total if total > 0 else 0.0
    passed = coverage >= min_coverage

    message = (
        f"OWASP LLM coverage {coverage:.0%} "
        f"({covered_count}/{total} categories) "
        + ("meets" if passed else "below")
        + f" minimum {min_coverage:.0%}"
    )

    return assert_true(
        passed,
        name="compliance.owasp_llm.coverage",
        message=message,
        severity=Severity.CRITICAL,
        covered_count=covered_count,
        total=total,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
    )


def generate_owasp_report(results: list[dict]) -> str:
    """Generate a plain-text report of OWASP LLM Top 10 compliance.

    The report lists each OWASP category with its coverage status,
    the matched test names, and any uncovered assertion prefixes.

    Args:
        results: List of result dicts (same format as :func:`owasp_llm_scan`).

    Returns:
        Multi-line string suitable for printing or writing to a ``.txt`` file.

    Example:
        >>> report = generate_owasp_report([])
        >>> "LLM01" in report
        True
    """
    scan = owasp_llm_scan(results)
    total = len(OWASP_LLM_IDS)
    covered_count = sum(1 for entry in scan.values() if entry["covered"])
    coverage_pct = (covered_count / total * 100) if total > 0 else 0.0

    lines: list[str] = [
        "=" * 60,
        "OWASP LLM Top 10 — mltk Compliance Report",
        "=" * 60,
        f"Coverage: {covered_count}/{total} categories ({coverage_pct:.0f}%)",
        "",
    ]

    for owasp_id in OWASP_LLM_IDS:
        entry = scan[owasp_id]
        status = "COVERED" if entry["covered"] else "MISSING"
        lines.append(f"{owasp_id}: {entry['title']} [{status}]")

        if entry["tests"]:
            for t in entry["tests"]:
                passed_label = "PASS" if t.get("passed") else "FAIL"
                lines.append(f"    [{passed_label}] {t.get('name', '?')}")
        else:
            lines.append("    (no tests mapped)")

        if entry["gaps"]:
            gap_str = ", ".join(entry["gaps"])
            lines.append(f"    Gaps: {gap_str}")

        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
