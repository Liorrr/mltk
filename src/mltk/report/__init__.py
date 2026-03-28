"""Report generation — Plotly charts + Jinja2 templates to interactive HTML."""

from mltk.report.bias_report import generate_bias_report
from mltk.report.generator import generate_report
from mltk.report.junit import export_junit_xml
from mltk.report.model_card import generate_model_card
from mltk.report.score import compute_ml_test_score
from mltk.report.summarizer import summarize_test_history
from mltk.report.visual_diff import generate_diff_report

__all__ = [
    "generate_bias_report",
    "generate_diff_report",
    "generate_report",
    "generate_model_card",
    "compute_ml_test_score",
    "summarize_test_history",
    "export_junit_xml",
]
