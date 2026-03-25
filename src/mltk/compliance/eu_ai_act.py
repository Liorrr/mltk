"""EU AI Act article mappings and risk classification helpers.

Pure data + helper functions — no I/O, no external dependencies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Risk level definitions
# ---------------------------------------------------------------------------

RISK_LEVELS = ["unacceptable", "high", "limited", "minimal"]

RISK_CLASSIFICATION: dict[str, dict] = {
    "unacceptable": {
        "label": "Unacceptable Risk",
        "description": (
            "AI systems posing an unacceptable risk to safety or fundamental rights. "
            "Prohibited under EU AI Act Title II."
        ),
        "color": "#f85149",
        "badge_class": "fail",
        "articles_required": [
            "Art. 10",
            "Art. 10(2f)",
            "Art. 14",
            "Art. 15",
            "Art. 72",
        ],
    },
    "high": {
        "label": "High Risk",
        "description": (
            "AI systems with significant impact on health, safety, or fundamental rights. "
            "Subject to mandatory requirements under EU AI Act Title III."
        ),
        "color": "#d29922",
        "badge_class": "warn",
        "articles_required": [
            "Art. 10",
            "Art. 10(2f)",
            "Art. 14",
            "Art. 15",
            "Art. 72",
        ],
    },
    "limited": {
        "label": "Limited Risk",
        "description": (
            "AI systems with specific transparency obligations. "
            "Governed by EU AI Act Title IV."
        ),
        "color": "#58a6ff",
        "badge_class": "info",
        "articles_required": ["Art. 10", "Art. 72"],
    },
    "minimal": {
        "label": "Minimal Risk",
        "description": (
            "AI systems with minimal impact. Voluntary codes of conduct apply. "
            "No mandatory compliance testing required."
        ),
        "color": "#3fb950",
        "badge_class": "pass",
        "articles_required": [],
    },
}

# ---------------------------------------------------------------------------
# Assertion-prefix → EU AI Act article mapping
# ---------------------------------------------------------------------------
# Keys are assertion name prefixes (matched via startswith).
# Order matters: more specific prefixes should come first.

ARTICLE_MAPPING: dict[str, dict] = {
    "data.no_pii": {
        "article": "Art. 10",
        "title": "Data Governance",
        "description": (
            "Training, validation, and testing data must be subject to appropriate "
            "data-governance practices including privacy and PII handling (Art. 10(5))."
        ),
    },
    "data.no_nulls": {
        "article": "Art. 10",
        "title": "Data Governance",
        "description": (
            "Datasets must be relevant, representative, and free of errors. "
            "Missing-value checks satisfy Art. 10(3) data quality requirements."
        ),
    },
    "data.schema": {
        "article": "Art. 10",
        "title": "Data Governance",
        "description": (
            "Datasets must have appropriate structure. Schema validation satisfies "
            "Art. 10(3) requirements for data completeness and correctness."
        ),
    },
    "data.drift": {
        "article": "Art. 72",
        "title": "Post-market Monitoring",
        "description": (
            "Input distribution drift must be monitored post-deployment. "
            "Satisfies Art. 72 continuous post-market monitoring obligations."
        ),
    },
    "model.bias": {
        "article": "Art. 10(2f)",
        "title": "Bias Detection",
        "description": (
            "High-risk AI systems must examine possible biases that could affect "
            "health, safety, or fundamental rights (Art. 10(2)(f))."
        ),
    },
    "model.slice": {
        "article": "Art. 14",
        "title": "Human Oversight",
        "description": (
            "Slice-level performance breakdown enables human oversight of subgroup "
            "disparities, supporting Art. 14 human-oversight requirements."
        ),
    },
    "model.calibration": {
        "article": "Art. 14",
        "title": "Human Oversight",
        "description": (
            "Calibration checks ensure confidence scores are interpretable, "
            "supporting human operators in overriding AI decisions (Art. 14(4))."
        ),
    },
    "model.metric": {
        "article": "Art. 15",
        "title": "Accuracy & Robustness",
        "description": (
            "High-risk AI systems must achieve appropriate levels of accuracy and "
            "be robust to errors and inconsistencies (Art. 15(1–2))."
        ),
    },
    "model.regression": {
        "article": "Art. 15",
        "title": "Accuracy & Robustness",
        "description": (
            "Regression tests detect performance degradation, satisfying the "
            "robustness and consistency requirements of Art. 15."
        ),
    },
    "model.adversarial": {
        "article": "Art. 15",
        "title": "Accuracy & Robustness",
        "description": (
            "Adversarial robustness tests address Art. 15(3) requirements that "
            "high-risk AI must be resilient to attempts to alter its use or performance."
        ),
    },
    "monitor.degradation": {
        "article": "Art. 72",
        "title": "Post-market Monitoring",
        "description": (
            "Performance-degradation monitoring satisfies the proactive post-market "
            "monitoring obligation introduced by Art. 72 of the EU AI Act."
        ),
    },
    "monitor.sla": {
        "article": "Art. 72",
        "title": "Post-market Monitoring",
        "description": (
            "SLA / latency monitoring ensures the deployed system continues to meet "
            "operational requirements, supporting Art. 72 monitoring plans."
        ),
    },
}

# Canonical article metadata — used to render per-article sections in order.
ARTICLE_META: list[dict] = [
    {
        "article": "Art. 10",
        "title": "Data Governance",
        "description": (
            "Training, validation and testing datasets shall be subject to "
            "appropriate data-governance and management practices."
        ),
    },
    {
        "article": "Art. 10(2f)",
        "title": "Bias Detection",
        "description": (
            "Data shall be examined for possible biases that could affect health, "
            "safety or fundamental rights of persons."
        ),
    },
    {
        "article": "Art. 14",
        "title": "Human Oversight",
        "description": (
            "High-risk AI systems shall be designed and developed to allow effective "
            "oversight by natural persons during deployment."
        ),
    },
    {
        "article": "Art. 15",
        "title": "Accuracy & Robustness",
        "description": (
            "High-risk AI systems shall be designed with appropriate levels of "
            "accuracy, robustness and cybersecurity."
        ),
    },
    {
        "article": "Art. 72",
        "title": "Post-market Monitoring",
        "description": (
            "Providers shall establish and implement a post-market monitoring system "
            "collecting and reviewing data on performance throughout the lifecycle."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def classify_risk(risk_level: str) -> dict:
    """Return risk classification details for the given level.

    Args:
        risk_level: One of ``RISK_LEVELS`` (case-insensitive).

    Returns:
        Dict with keys: label, description, color, badge_class, articles_required.

    Raises:
        ValueError: If risk_level is not a known level.

    Example:
        >>> info = classify_risk("high")
        >>> info["label"]
        'High Risk'
    """
    key = risk_level.lower().strip()
    if key not in RISK_CLASSIFICATION:
        raise ValueError(
            f"Unknown risk level {risk_level!r}. "
            f"Valid levels: {RISK_LEVELS}"
        )
    return RISK_CLASSIFICATION[key]


def _prefix_to_article(assertion_name: str) -> str | None:
    """Return the EU AI Act article for an assertion name, or None."""
    for prefix, meta in ARTICLE_MAPPING.items():
        if assertion_name.startswith(prefix):
            return meta["article"]
    return None


def map_results_to_articles(results: list[dict]) -> dict[str, list]:
    """Group test results by EU AI Act article.

    Each result dict is expected to have at least:
    - ``name`` (str): assertion name, e.g. ``"model.bias.demographic_parity"``
    - ``passed`` (bool): whether the assertion passed
    - ``message`` (str, optional): human-readable message

    Results whose names do not match any known article prefix are placed in
    the special ``"uncategorised"`` bucket.

    Args:
        results: List of result dicts (from JSON or TestResult serialisation).

    Returns:
        Dict mapping article strings (e.g. ``"Art. 10"``) to lists of result dicts.
        Each result dict is enriched with an ``"article"`` key.

    Example:
        >>> r = [{"name": "model.bias.x", "passed": True, "message": "ok"}]
        >>> grouped = map_results_to_articles(r)
        >>> "Art. 10(2f)" in grouped
        True
    """
    grouped: dict[str, list] = {}
    for r in results:
        name = str(r.get("name", ""))
        article = _prefix_to_article(name) or "uncategorised"
        enriched = dict(r)
        enriched["article"] = article
        grouped.setdefault(article, []).append(enriched)
    return grouped


def find_gaps(results: list[dict], risk_level: str) -> list[str]:
    """Find EU AI Act articles required for *risk_level* but not covered by results.

    A requirement is considered *covered* if at least one result exists whose
    assertion name maps to that article.

    Args:
        results: List of result dicts (same format as :func:`map_results_to_articles`).
        risk_level: One of ``RISK_LEVELS`` (case-insensitive).

    Returns:
        Sorted list of article strings that have no matching test results.

    Example:
        >>> gaps = find_gaps([], "high")
        >>> "Art. 10" in gaps
        True
    """
    classification = classify_risk(risk_level)
    required = set(classification["articles_required"])
    covered = set()
    for r in results:
        name = str(r.get("name", ""))
        article = _prefix_to_article(name)
        if article:
            covered.add(article)
    return sorted(required - covered)
