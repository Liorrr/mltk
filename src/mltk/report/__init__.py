"""Report generation — Plotly charts + Jinja2 templates to interactive HTML."""

from mltk.report.bias_report import generate_bias_report
from mltk.report.generator import generate_report
from mltk.report.model_card import generate_model_card
from mltk.report.score import compute_ml_test_score

__all__ = [
    "generate_bias_report",
    "generate_report",
    "generate_model_card",
    "compute_ml_test_score",
]
