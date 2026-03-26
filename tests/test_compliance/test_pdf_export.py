"""Tests for compliance PDF/print-ready HTML export."""

from __future__ import annotations

from mltk.compliance.pdf_export import _inject_print_css, export_compliance_pdf

SAMPLE_HTML = """<html><head><title>Report</title></head>
<body><h1>EU AI Act Compliance</h1><p>Results here.</p></body></html>"""


def test_inject_print_css_has_media_print(tmp_path):
    result = _inject_print_css(SAMPLE_HTML)
    assert "@media print" in result


def test_inject_print_css_before_head_close(tmp_path):
    result = _inject_print_css(SAMPLE_HTML)
    idx_print = result.index("@media print")
    idx_head = result.index("</head>")
    assert idx_print < idx_head


def test_export_creates_print_html(tmp_path):
    src = tmp_path / "report.html"
    src.write_text(SAMPLE_HTML, encoding="utf-8")
    out = export_compliance_pdf(str(src))
    assert out.exists()
    assert "-print.html" in out.name
    assert "@media print" in out.read_text()


def test_export_custom_output(tmp_path):
    src = tmp_path / "report.html"
    src.write_text(SAMPLE_HTML, encoding="utf-8")
    custom = tmp_path / "custom-output.html"
    out = export_compliance_pdf(str(src), output_path=str(custom))
    assert out == custom
    assert out.exists()
