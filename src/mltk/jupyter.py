"""Jupyter notebook helpers — rich display for mltk results."""

from mltk.core.result import TestSuite


def display_report(suite: TestSuite) -> None:
    """Display a rich inline report in a Jupyter notebook cell."""
    try:
        from IPython.display import HTML, display

        display(HTML(suite._repr_html_()))
    except ImportError:
        # Not in Jupyter — print plain text
        print(f"mltk: {suite.passed_count}/{suite.total} passed ({suite.score:.1f}%)")
        for r in suite.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")
