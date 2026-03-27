"""Compliance — EU AI Act, NIST AI RMF, ISO 42001, OWASP, FDA, HIPAA, Custom, PDF."""

from mltk.compliance.eu_ai_act import classify_risk, find_gaps, map_results_to_articles
from mltk.compliance.fda import generate_fda_audit_trail
from mltk.compliance.generator import generate_compliance_report
from mltk.compliance.hipaa import assert_hipaa_coverage
from mltk.compliance.hipaa import find_gaps as find_hipaa_gaps
from mltk.compliance.iso_42001 import (
    assert_iso_42001_coverage,
    map_results_to_clauses,
)
from mltk.compliance.iso_42001 import find_gaps as find_iso_42001_gaps
from mltk.compliance.nist_ai_rmf import (
    assert_nist_rmf_coverage,
    map_results_to_measures,
)
from mltk.compliance.nist_ai_rmf import find_gaps as find_nist_rmf_gaps
from mltk.compliance.owasp_llm import (
    assert_owasp_coverage,
    generate_owasp_report,
    owasp_llm_scan,
)
from mltk.compliance.pdf_export import export_compliance_pdf

__all__ = [
    "generate_compliance_report",
    # OWASP
    "owasp_llm_scan",
    "assert_owasp_coverage",
    "generate_owasp_report",
    # FDA
    "generate_fda_audit_trail",
    # PDF
    "export_compliance_pdf",
    # EU AI Act
    "classify_risk",
    "map_results_to_articles",
    "find_gaps",
    # NIST AI RMF
    "assert_nist_rmf_coverage",
    "find_nist_rmf_gaps",
    "map_results_to_measures",
    # ISO 42001
    "assert_iso_42001_coverage",
    "find_iso_42001_gaps",
    "map_results_to_clauses",
    # HIPAA
    "assert_hipaa_coverage",
    "find_hipaa_gaps",
]
