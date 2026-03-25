"""Report generation — Plotly charts + Jinja2 templates to interactive HTML."""

from mltk.report.generator import generate_report
from mltk.report.score import compute_ml_test_score

__all__ = ["generate_report", "compute_ml_test_score"]
