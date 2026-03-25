"""Tests for mltk.report.generator -- HTML report generation.

Report generation is the user-facing output of mltk. These tests verify that:
- HTML files are created with correct extensions
- Reports contain all expected sections (title, test names, outcomes)
- Dark theme CSS is applied for readability
- Empty test runs produce valid reports (not crashes)
- ML Test Score rubric correctly computes from results
"""

from pathlib import Path

from mltk.report.generator import generate_report
from mltk.report.score import compute_ml_test_score


class TestGenerateReport:
    """Tests for HTML report generation.

    Validates that generate_report produces valid HTML files with correct
    structure, styling, and content for both normal and edge-case inputs.
    """

    def test_report_generates_file(self, tmp_path: Path) -> None:
        """PASS: Report creates an .html file in the output directory.

        WHY: The report file is the primary artifact teams review after a
        test run. If the file is not created or has the wrong extension,
        CI artifact collection and browser rendering will fail.
        Expected: File exists and has .html suffix.
        """
        results = [
            {"nodeid": "tests/test_data.py::test_a", "outcome": "passed", "duration": 0.1},
            {"nodeid": "tests/test_data.py::test_b", "outcome": "failed", "duration": 0.2},
            {"nodeid": "tests/test_model.py::test_c", "outcome": "passed", "duration": 0.3},
        ]
        path = generate_report(results, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_report_contains_sections(self, tmp_path: Path) -> None:
        """PASS: HTML report contains branding, title, and all test names.

        WHY: Stakeholders scan reports for specific test names and outcomes.
        If the report omits test_schema or test_metric, teams cannot tell
        whether those checks ran. The title and "MLTK" branding verify the
        correct template was used.
        Expected: HTML contains "MLTK", custom title, and both test names.
        """
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
        """PASS: Report HTML contains dark theme background color.

        WHY: mltk uses a dark theme (#0d1117) for visual consistency with
        developer tooling (VS Code, GitHub Dark). This ensures the CSS
        template is correctly embedded, not a plain unstyled page.
        Expected: #0d1117 present in HTML output.
        """
        results = [{"nodeid": "test_a", "outcome": "passed", "duration": 0.01}]
        path = generate_report(results, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "#0d1117" in html  # Dark background color

    def test_empty_results(self, tmp_path: Path) -> None:
        """PASS: Empty results produce a valid report file (not a crash).

        WHY: A test run with zero collected tests (e.g., marker filtering
        excluded everything) should still produce a report showing "0 tests"
        rather than crashing the report generator.
        Expected: File exists.
        """
        path = generate_report([], output_dir=tmp_path)
        assert path.exists()


class TestMLTestScore:
    """Tests for the ML Test Score calculator.

    ML Test Score is a rubric (max 28 points) that rates how comprehensive
    your ML testing is across categories (data, model, infra, monitoring).
    These tests verify the scoring logic.
    """

    def test_score_with_results(self) -> None:
        """PASS: Score computed from mixed pass/fail results across categories.

        WHY: The ML Test Score maps test node IDs to categories (data, model,
        etc.) and awards points. This verifies the mapping works and the
        max score is correct (28 = 7 categories x 4 levels).
        Expected: total, categories, and max=28 present in score dict.
        """
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
        """Edge case: Empty results produce score 0 with 0% percentage.

        WHY: No tests run means no coverage of any ML testing category.
        The scorer must return a valid dict (total=0, percentage=0) rather
        than raising a ZeroDivisionError.
        Expected: score["total"]==0, score["percentage"]==0.
        """
        score = compute_ml_test_score([])
        assert score["total"] == 0
        assert score["percentage"] == 0
