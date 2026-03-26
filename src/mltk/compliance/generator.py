"""EU AI Act compliance report generator.

Produces a self-contained HTML report from a mltk test-results JSON file,
mapping each assertion to the relevant EU AI Act article and highlighting gaps.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mltk.compliance.eu_ai_act import (
    ARTICLE_META,
    classify_risk,
    find_gaps,
    map_results_to_articles,
)
from mltk.report._helpers import _load_results


def generate_compliance_report(
    results_path: str | Path,
    risk_level: str = "high",
    system_name: str = "AI System",
    output_dir: str | Path = "./mltk-reports",
) -> Path:
    """Generate an EU AI Act compliance HTML report from a test results JSON file.

    Args:
        results_path: Path to a JSON file containing mltk test results.
            Each result must have at minimum a ``"name"`` (str) and
            ``"passed"`` (bool) key; an optional ``"message"`` (str) is shown
            in the report.
        risk_level: EU AI Act risk classification. One of ``"unacceptable"``,
            ``"high"``, ``"limited"``, ``"minimal"``. Defaults to ``"high"``.
        system_name: Human-readable name of the AI system under evaluation.
        output_dir: Directory where the HTML report file will be written.

    Returns:
        :class:`pathlib.Path` pointing to the generated HTML file.

    Raises:
        ImportError: If ``jinja2`` is not installed.
        ValueError: If *risk_level* is not a recognised value.
        FileNotFoundError: If *results_path* does not exist.

    Example:
        >>> from mltk.compliance import generate_compliance_report
        >>> path = generate_compliance_report(
        ...     "results.json",
        ...     risk_level="high",
        ...     system_name="My Classifier",
        ... )
        >>> path.suffix
        '.html'
    """
    try:
        from jinja2 import Template
    except ImportError as err:
        raise ImportError(
            "jinja2 is required for compliance report generation. "
            "Install with: pip install mltk[report]"
        ) from err

    # Load template
    template_path = Path(__file__).parent / "templates" / "eu_ai_act.html.j2"
    template = Template(template_path.read_text(encoding="utf-8"))

    # Load + classify results
    results = _load_results(results_path)
    classification = classify_risk(risk_level)
    grouped = map_results_to_articles(results)
    gaps = find_gaps(results, risk_level)

    # Summary stats
    total = len(results)
    passed_count = sum(1 for r in results if r.get("passed", False))
    failed_count = total - passed_count
    compliance_score = round((passed_count / total * 100) if total > 0 else 0.0, 1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build per-article sections (only those relevant to the chosen risk level)
    required_articles = set(classification["articles_required"])
    article_sections = []
    for meta in ARTICLE_META:
        art = meta["article"]
        if art not in required_articles:
            continue
        art_results = grouped.get(art, [])
        art_passed = sum(1 for r in art_results if r.get("passed", False))
        art_failed = len(art_results) - art_passed
        article_sections.append(
            {
                "article": art,
                "title": meta["title"],
                "description": meta["description"],
                "results": art_results,
                "passed": art_passed,
                "failed": art_failed,
                "total": len(art_results),
                "covered": len(art_results) > 0,
                "status": (
                    "gap"
                    if len(art_results) == 0
                    else ("pass" if art_failed == 0 else "fail")
                ),
            }
        )

    # Render HTML
    html = template.render(
        system_name=system_name,
        risk_level=risk_level,
        risk_label=classification["label"],
        risk_color=classification["color"],
        risk_badge_class=classification["badge_class"],
        risk_description=classification["description"],
        timestamp=timestamp,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        compliance_score=compliance_score,
        article_sections=article_sections,
        gaps=gaps,
    )

    # Write output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"eu-ai-act-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")

    return out_path
