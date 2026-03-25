"""Tests for mltk.report.generator -- HTML report generation."""

from pathlib import Path

from mltk.report.generator import generate_report
from mltk.report.score import compute_ml_test_score


class TestGenerateReport:
    """Tests for HTML report generation."""

    def test_report_generates_file(self, tmp_path: Path) -> None:
        """Report creates an HTML file in output directory."""
        results = [
            {"nodeid": "tests/test_data.py::test_a", "outcome": "passed", "duration": 0.1},
            {"nodeid": "tests/test_data.py::test_b", "outcome": "failed", "duration": 0.2},
            {"nodeid": "tests/test_model.py::test_c", "outcome": "passed", "duration": 0.3},
        ]
        path = generate_report(results, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_report_contains_sections(self, tmp_path: Path) -> None:
        """HTML report contains expected content sections."""
        results = [
            {"nodeid": "tests/test_data.py::test_schema", "outcome": "passed", "duration": 0.05},
            {"nodeid": "tests/test_model.py::test_metric", "outcome": "failed", "duration": 0.1},
        ]
        path = generate_report(results, output_dir=tmp_path, title="Test Run")
        html = path.read_text(encoding="utf-8")
        assert "MLTK" in html
        assert "Test Run" in html
        assert "test_schema" in html
        assert "test_metric" in html

    def test_report_dark_theme(self, tmp_path: Path) -> None:
        """Report uses dark theme CSS variables."""
        results = [{"nodeid": "test_a", "outcome": "passed", "duration": 0.01}]
        path = generate_report(results, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "#0d1117" in html  # Dark background color

    def test_empty_results(self, tmp_path: Path) -> None:
        """Report handles empty results gracefully."""
        path = generate_report([], output_dir=tmp_path)
        assert path.exists()


class TestMLTestScore:
    """Tests for ML Test Score calculator."""

    def test_score_with_results(self) -> None:
        """Score computes from test results."""
        results = [
            {"nodeid": "tests/test_data/test_schema.py::test_valid", "outcome": "passed"},
            {"nodeid": "tests/test_data/test_drift.py::test_no_drift", "outcome": "passed"},
            {"nodeid": "tests/test_model/test_metrics.py::test_accuracy", "outcome": "passed"},
            {"nodeid": "tests/test_model/test_bias.py::test_fair", "outcome": "failed"},
        ]
        score = compute_ml_test_score(results)
        assert "total" in score
        assert "categories" in score
        assert score["max"] == 28

    def test_empty_results(self) -> None:
        """Score handles empty results."""
        score = compute_ml_test_score([])
        assert score["total"] == 0
        assert score["percentage"] == 0
