"""Tests for mltk.report.generator -- HTML report generation.

Report generation is the user-facing output of mltk. These tests verify that:
- HTML files are created with correct extensions
- Reports contain all expected sections (title, test names, outcomes)
- Dark theme CSS is applied for readability
- Empty test runs produce valid reports (not crashes)
- ML Test Score rubric correctly computes from results
"""

from pathlib import Path

from mltk.core.result import Severity, TestResult
from mltk.report.generator import generate_report
from mltk.report.score import compute_ml_test_score


def _result(
    name: str = "assertion.example",
    *,
    details: dict | None = None,
    passed: bool = True,
) -> TestResult:
    """Build a deterministic assertion result for report rendering tests."""
    return TestResult(
        name=name,
        passed=passed,
        severity=Severity.INFO,
        message="recorded message",
        details=details or {},
    )


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

    def test_report_contains_charts(self, tmp_path: Path) -> None:
        """PASS: Report embeds self-contained SVG charts.

        WHY: Charts provide visual pass/fail and duration overview. They are
        rendered as inline SVG with no external JS dependency (the generator
        never uses plotly — see test_report_no_external_deps in
        test_charts.py), so they must be present regardless of which optional
        packages are installed. The previous version of this test asserted a
        'plotly' reference and had only ever skipped: no environment that ran
        it had plotly installed until CI did — and then it failed.
        Expected: HTML contains the inline SVG donut chart markup.
        """
        results = [
            {"nodeid": "test_a", "outcome": "passed", "duration": 0.05},
            {"nodeid": "test_b", "outcome": "failed", "duration": 0.12},
            {"nodeid": "test_c", "outcome": "passed", "duration": 0.30},
        ]
        path = generate_report(results, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "<svg" in html.lower()
        assert "<circle" in html or "donut" in html.lower()

    def test_report_no_crash_minimal_results(self, tmp_path: Path) -> None:
        """PASS: Report generates a valid file from a minimal results list.

        WHY: Users who install only the report extra's jinja2 must get a
        valid, self-contained HTML report — the generator has no optional
        chart dependency to degrade around (charts are inline SVG).
        Expected: File exists and contains the test details section.
        """
        results = [{"nodeid": "test_x", "outcome": "passed", "duration": 0.01}]
        path = generate_report(results, output_dir=tmp_path)
        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "test_x" in html


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


class TestAssertionDetails:
    """Verify assertion details are rendered safely and with fixed limits."""

    def test_details_render_and_escape_values_and_nodeids(self, tmp_path: Path) -> None:
        """DETAILS: Recorded key/value details and untrusted text are rendered safely.

        WHY: Reports must show the evidence behind an assertion without allowing model
        output or pytest node IDs to become HTML markup.
        """
        result = _result(details={"score": 0.95, "output": "<script>alert(1)</script>"})
        path = generate_report(
            [{
                "nodeid": "test_<script>",
                "outcome": "passed",
                "duration": 0,
                "ml_results": [result],
            }],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert "score = 0.95" in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "<script>alert" not in html
        assert "test_&lt;script&gt;" in html

    def test_detail_value_is_truncated_to_120_characters(self, tmp_path: Path) -> None:
        """LIMIT: A long recorded value is truncated before template rendering.

        WHY: Unbounded model output can make a report unreadable; the exact limit
        keeps output predictable for both humans and downstream checks.
        """
        path = generate_report(
            [{
                "nodeid": "test_long",
                "outcome": "passed",
                "ml_results": [_result(details={"value": "x" * 300})],
            }],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")
        rendered = f'{"x" * 119}…'

        assert rendered in html
        assert "x" * 120 not in html

    def test_detail_and_assertion_limits_show_overflow_lines(self, tmp_path: Path) -> None:
        """LIMITS: Detail keys and assertions show bounded overflow indicators.

        WHY: Reports need to preserve the presence of omitted evidence while avoiding
        huge tables when a test records many assertions or fields.
        """
        details = {f"key{i}": f"value{i}" for i in range(25)}
        assertions = [_result(name=f"assertion-{i}") for i in range(55)]
        assertions[0] = _result(name="assertion-0", details=details)
        path = generate_report(
            [{"nodeid": "test_limits", "outcome": "passed", "ml_results": assertions}],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert "key19 = value19" in html
        assert "key20 = value20" not in html
        assert "… and 5 more" in html
        assert "assertion-49" in html
        assert "assertion-50" not in html
        assert "… and 5 more assertions" in html

    def test_empty_details_have_no_detail_rows(self, tmp_path: Path) -> None:
        """EMPTY: An assertion with no details has no key/value row markup.

        WHY: Missing evidence should remain empty rather than displaying placeholder
        values or an empty-looking detail row.
        """
        path = generate_report(
            [{
                "nodeid": "test_empty",
                "outcome": "passed",
                "ml_results": [_result(name="empty.assertion")],
            }],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert '<span class="assertion-name">empty.assertion</span>' in html
        assert '<div class="assertion-detail">' not in html
        assert '<div class="assertion-details">' not in html

    def test_legacy_result_fallback_and_explicit_empty_results(self, tmp_path: Path) -> None:
        """COMPAT: Legacy records fall back to ml_result, but an explicit empty list wins.

        WHY: Suite and scan exporters still provide only ml_result, while the pytest
        collector deliberately uses an empty ml_results list when nothing was recorded.
        """
        legacy = _result(name="legacy.assertion", details={"legacy": "value"})
        suppressed = _result(name="suppressed.assertion", details={"hidden": "value"})
        path = generate_report(
            [
                {"nodeid": "legacy", "outcome": "passed", "ml_result": legacy},
                {
                    "nodeid": "explicit_empty",
                    "outcome": "passed",
                    "ml_result": suppressed,
                    "ml_results": [],
                },
            ],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert "legacy.assertion" in html
        assert "legacy = value" in html
        assert "suppressed.assertion" not in html
        assert "hidden = value" not in html

    def test_unencodable_detail_values_fall_back_to_str(self, tmp_path: Path) -> None:
        """ROBUST: A detail json.dumps cannot encode renders via str() instead.

        WHY: Tuple or numpy-int dict keys and circular structures raise in json.dumps;
        one such custom detail must not abort the HTML report for the whole session.
        """
        circular: list[object] = []
        circular.append(circular)
        path = generate_report(
            [{
                "nodeid": "test_odd",
                "outcome": "passed",
                "ml_results": [_result(details={"per_pair": {(0, 1): 0.9}, "loop": circular})],
            }],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert "per_pair = {(0, 1): 0.9}" in html
        assert "loop = [[...]]" in html

    def test_dict_and_list_details_render_as_json(self, tmp_path: Path) -> None:
        """FORMAT: Dict and list details use JSON serialization before escaping.

        WHY: Structured evidence should remain recognizable in the compact report
        instead of using Python repr syntax or fabricated summaries.
        """
        path = generate_report(
            [{
                "nodeid": "test_json",
                "outcome": "passed",
                "ml_results": [_result(details={"mapping": {"a": "b"}, "items": [1, 2]})],
            }],
            output_dir=tmp_path,
        )
        html = path.read_text(encoding="utf-8")

        assert "mapping = {&#34;a&#34;: &#34;b&#34;}" in html
        assert "items = [1, 2]" in html
