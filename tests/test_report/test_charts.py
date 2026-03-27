"""Tests for visualization charts in the HTML report."""

from pathlib import Path


class TestReportCharts:
    """Verify charts are rendered in the HTML report."""

    def test_report_contains_donut_chart(self, tmp_path: Path) -> None:
        """Report HTML includes SVG donut chart for pass/fail."""
        from mltk.report.generator import generate_report

        results = [
            {"nodeid": "test_a", "outcome": "passed", "duration": 0.1},
            {"nodeid": "test_b", "outcome": "passed", "duration": 0.2},
            {"nodeid": "test_c", "outcome": "failed", "duration": 0.3},
        ]
        report_path = generate_report(results, output_dir=tmp_path)
        html = report_path.read_text(encoding="utf-8")
        # SVG circle element for the donut
        assert "<circle" in html or "donut" in html.lower()

    def test_report_contains_module_bars(self, tmp_path: Path) -> None:
        """Report HTML includes module breakdown bars."""
        from mltk.report.generator import generate_report

        results = [
            {"nodeid": "tests/test_core/test_a.py::test_1", "outcome": "passed", "duration": 0.1},
            {"nodeid": "tests/test_core/test_a.py::test_2", "outcome": "failed", "duration": 0.2},
            {"nodeid": "tests/test_model/test_b.py::test_3", "outcome": "passed", "duration": 0.1},
        ]
        report_path = generate_report(results, output_dir=tmp_path)
        html = report_path.read_text(encoding="utf-8")
        # Module names should appear in the bar chart section
        assert "test_core" in html or "module" in html.lower()

    def test_report_no_external_deps(self, tmp_path: Path) -> None:
        """Report HTML does not reference external CDN scripts."""
        from mltk.report.generator import generate_report

        results = [
            {"nodeid": "test_x", "outcome": "passed", "duration": 0.05},
        ]
        report_path = generate_report(results, output_dir=tmp_path)
        html = report_path.read_text(encoding="utf-8")
        assert "cdn." not in html.lower()
        assert "plotly" not in html.lower()
        assert "chart.js" not in html.lower()

    def test_empty_results_no_crash(self, tmp_path: Path) -> None:
        """Empty results list doesn't crash chart generation."""
        from mltk.report.generator import generate_report

        report_path = generate_report([], output_dir=tmp_path)
        html = report_path.read_text(encoding="utf-8")
        assert "MLTK" in html or "mltk" in html.lower()

    def test_all_passed_donut(self, tmp_path: Path) -> None:
        """100% pass rate renders correctly."""
        from mltk.report.generator import generate_report

        results = [
            {"nodeid": f"test_{i}", "outcome": "passed", "duration": 0.01}
            for i in range(10)
        ]
        report_path = generate_report(results, output_dir=tmp_path)
        html = report_path.read_text(encoding="utf-8")
        assert "100" in html  # 100% pass rate somewhere in the report
