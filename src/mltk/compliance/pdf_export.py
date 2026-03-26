"""Convert HTML compliance reports to print-ready PDF or HTML."""

from __future__ import annotations

from pathlib import Path

_PRINT_CSS = """
<style>
@media print {
    body { background: #fff !important; color: #000 !important; font-family: serif; }
    div { background: #fff !important; color: #000 !important; border-color: #ccc !important; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #666; padding: 6px 10px; text-align: left; }
    th { background: #eee !important; font-weight: bold; }
    span { color: #000 !important; }
    h1, h2, h3 { page-break-after: avoid; color: #000 !important; }
    .page-break { page-break-before: always; }
    @page { margin: 2cm; }
}
</style>
"""


def _inject_print_css(html_content: str) -> str:
    """Inject print-optimized CSS into an HTML document."""
    if "</head>" in html_content:
        return html_content.replace("</head>", _PRINT_CSS + "</head>")
    if "<body" in html_content:
        return html_content.replace("<body", _PRINT_CSS + "<body")
    return _PRINT_CSS + html_content


def export_compliance_pdf(
    html_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an HTML compliance report to PDF or print-ready HTML.

    Strategy:
    1. Try weasyprint (if installed via ``mltk[pdf]``)
    2. Fallback: inject ``@media print`` CSS for browser Print-to-PDF

    Args:
        html_path: Path to the HTML compliance report.
        output_path: Destination path. If None, replaces ``.html`` with
            ``.pdf`` (weasyprint) or ``-print.html`` (fallback).

    Returns:
        Path to the generated output file.
    """
    src = Path(html_path)
    html_content = src.read_text(encoding="utf-8")

    # Try weasyprint
    try:
        import weasyprint  # type: ignore[import-untyped]

        pdf_path = Path(output_path) if output_path else src.with_suffix(".pdf")
        weasyprint.HTML(string=html_content).write_pdf(str(pdf_path))
        return pdf_path
    except ImportError:
        pass

    # Fallback: print-ready HTML
    print_path = (
        Path(output_path)
        if output_path
        else src.with_name(src.stem + "-print.html")
    )
    print_html = _inject_print_css(html_content)
    print_path.write_text(print_html, encoding="utf-8")
    return print_path
