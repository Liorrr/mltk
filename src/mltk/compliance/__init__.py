"""Compliance reporting — EU AI Act, OWASP LLM Top 10, FDA audit trails, PDF."""

from mltk.compliance.eu_ai_act import classify_risk, find_gaps, map_results_to_articles
from mltk.compliance.fda import generate_fda_audit_trail
from mltk.compliance.generator import generate_compliance_report
from mltk.compliance.owasp_llm import (
    assert_owasp_coverage,
    generate_owasp_report,
    owasp_llm_scan,
)
from mltk.compliance.pdf_export import export_compliance_pdf

__all__ = [
    "generate_compliance_report",
    "owasp_llm_scan",
    "assert_owasp_coverage",
    "generate_owasp_report",
    "generate_fda_audit_trail",
    "export_compliance_pdf",
    "classify_risk",
    "map_results_to_articles",
    "find_gaps",
]
