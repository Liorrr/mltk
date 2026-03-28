"""Tests for mltk.report.junit -- JUnit XML export.

JUnit XML is the lingua franca of CI/CD test reporting.  These tests verify
that export_junit_xml produces well-formed, standards-compliant XML that
Jenkins, GitLab CI, Azure DevOps, and CircleCI can ingest without errors.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from mltk.report.junit import export_junit_xml, format_result_to_junit


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _all_passed() -> list[dict]:
    """Three passing results."""
    return [
        {
            "name": "data.schema.check",
            "passed": True,
            "duration_ms": 50.0,
            "message": "schema ok",
        },
        {
            "name": "data.drift.psi",
            "passed": True,
            "duration_ms": 120.0,
            "message": "PSI=0.03",
        },
        {
            "name": "model.metric.accuracy",
            "passed": True,
            "duration_ms": 80.0,
            "message": "accuracy 0.92 >= 0.80",
        },
    ]


def _mixed_results() -> list[dict]:
    """Two passes and two failures."""
    return [
        {
            "name": "data.schema.check",
            "passed": True,
            "duration_ms": 50.0,
            "message": "schema ok",
        },
        {
            "name": "model.metric.accuracy",
            "passed": False,
            "duration_ms": 120.0,
            "message": "accuracy 0.75 < 0.80",
        },
        {
            "name": "data.drift.psi",
            "passed": True,
            "duration_ms": 30.0,
            "message": "PSI=0.01",
        },
        {
            "name": "model.bias.dpd",
            "passed": False,
            "duration_ms": 200.0,
            "message": "DPD 0.15 > 0.10",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportJunitXml:
    """Tests for the export_junit_xml function.

    Validates XML structure, attribute counts, file creation, and
    correct handling of edge cases (empty results, special characters).
    """

    def test_all_passed_valid_xml(self, tmp_path: Path) -> None:
        """PASS: All-passed results produce valid XML with zero failures.

        WHY: CI dashboards mark a build as green only when failures=0.
        If the exporter miscounts, builds will be incorrectly red.
        Expected: well-formed XML, failures="0", 3 testcase elements.
        """
        out = tmp_path / "results.xml"
        path = export_junit_xml(
            _all_passed(), output_path=str(out)
        )
        assert Path(path).exists()

        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "testsuites"

        suite = root.find("testsuite")
        assert suite is not None
        assert suite.get("tests") == "3"
        assert suite.get("failures") == "0"
        assert suite.get("errors") == "0"

        cases = suite.findall("testcase")
        assert len(cases) == 3
        for tc in cases:
            assert tc.find("failure") is None

    def test_mixed_results_failure_elements(
        self, tmp_path: Path
    ) -> None:
        """PASS: Failed tests include <failure> child elements.

        WHY: CI systems parse <failure> to show error details. Missing
        failure elements means failures are silently swallowed.
        Expected: 2 testcase elements have <failure> children.
        """
        out = tmp_path / "mixed.xml"
        export_junit_xml(_mixed_results(), output_path=str(out))

        tree = ET.parse(str(out))
        suite = tree.getroot().find("testsuite")
        assert suite is not None

        failures = [
            tc for tc in suite.findall("testcase")
            if tc.find("failure") is not None
        ]
        assert len(failures) == 2
        assert suite.get("failures") == "2"

    def test_xml_well_formed(self, tmp_path: Path) -> None:
        """PASS: Output is valid XML parseable by ET.fromstring.

        WHY: Malformed XML will crash every CI parser. This is the
        most fundamental correctness check.
        Expected: ET.fromstring succeeds without ParseError.
        """
        out = tmp_path / "wellformed.xml"
        path = export_junit_xml(
            _mixed_results(), output_path=str(out)
        )
        xml_text = Path(path).read_text(encoding="utf-8")
        root = ET.fromstring(xml_text)
        assert root.tag == "testsuites"

    def test_special_characters_escaped(
        self, tmp_path: Path
    ) -> None:
        """PASS: XML special characters in names/messages are escaped.

        WHY: Test names or messages containing <, >, &, or quotes will
        break XML parsing if not escaped. Real-world messages like
        "accuracy < threshold" contain these characters.
        Expected: XML is parseable and contains the original content.
        """
        results = [
            {
                "name": 'data.<"schema">.check',
                "passed": False,
                "duration_ms": 10.0,
                "message": "value < 0.5 & ratio > 1.0",
            },
        ]
        out = tmp_path / "special.xml"
        path = export_junit_xml(results, output_path=str(out))

        xml_text = Path(path).read_text(encoding="utf-8")
        root = ET.fromstring(xml_text)

        # Raw XML must contain escaped entities
        assert "&lt;" in xml_text
        assert "&amp;" in xml_text

        # ET parses them back to originals
        tc = root.find(".//testcase")
        assert tc is not None
        assert "<" in tc.get("name", "")
        assert '"' in tc.get("name", "")

        fail = tc.find("failure")
        assert fail is not None
        assert "&" in fail.get("message", "")
        assert "<" in fail.get("message", "")

    def test_empty_results(self, tmp_path: Path) -> None:
        """PASS: Empty results produce valid XML with tests="0".

        WHY: A filtered test run may collect zero results. The exporter
        must produce a valid (but empty) XML document rather than crash.
        Expected: tests="0", failures="0", no testcase elements.
        """
        out = tmp_path / "empty.xml"
        path = export_junit_xml([], output_path=str(out))
        assert Path(path).exists()

        tree = ET.parse(path)
        suite = tree.getroot().find("testsuite")
        assert suite is not None
        assert suite.get("tests") == "0"
        assert suite.get("failures") == "0"
        assert len(suite.findall("testcase")) == 0

    def test_output_file_created(self, tmp_path: Path) -> None:
        """PASS: XML file is created at the specified path.

        WHY: CI artifact collectors look for files at specific paths.
        If the file lands elsewhere or is not written, the pipeline
        cannot publish results.
        Expected: File exists at the exact path given.
        """
        out = tmp_path / "subdir" / "report.xml"
        path = export_junit_xml(
            _all_passed(), output_path=str(out)
        )
        assert Path(path).exists()
        assert Path(path).name == "report.xml"

    def test_suite_name_customizable(self, tmp_path: Path) -> None:
        """PASS: Suite name attribute reflects the user-specified value.

        WHY: Teams running multiple test suites (data, model, infra)
        need distinct suite names for CI grouping and filtering.
        Expected: testsuite name="custom-suite".
        """
        out = tmp_path / "custom.xml"
        export_junit_xml(
            _all_passed(),
            output_path=str(out),
            suite_name="custom-suite",
        )

        tree = ET.parse(str(out))
        suite = tree.getroot().find("testsuite")
        assert suite is not None
        assert suite.get("name") == "custom-suite"

    def test_duration_converted_ms_to_seconds(
        self, tmp_path: Path
    ) -> None:
        """PASS: duration_ms is correctly converted to seconds.

        WHY: JUnit XML spec uses seconds for the time attribute. mltk
        stores duration in milliseconds. A conversion error (e.g.,
        reporting 500 seconds instead of 0.5) would distort CI dashboards.
        Expected: testcase time="0.500000", suite time close to 0.5.
        """
        results = [
            {
                "name": "timing.test",
                "passed": True,
                "duration_ms": 500.0,
                "message": "ok",
            },
        ]
        out = tmp_path / "timing.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        suite = tree.getroot().find("testsuite")
        assert suite is not None
        assert float(suite.get("time", "0")) == 0.5

        tc = suite.find("testcase")
        assert tc is not None
        assert float(tc.get("time", "0")) == 0.5

    def test_failure_type_attribute(self, tmp_path: Path) -> None:
        """PASS: Failure elements have type="MltkAssertionError".

        WHY: CI systems use the failure type to categorize and group
        errors. A consistent type helps teams filter mltk-specific
        failures from other test framework failures in mixed suites.
        Expected: failure type="MltkAssertionError".
        """
        results = [
            {
                "name": "model.metric.f1",
                "passed": False,
                "duration_ms": 100.0,
                "message": "F1 too low",
            },
        ]
        out = tmp_path / "ftype.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        fail = tree.getroot().find(".//failure")
        assert fail is not None
        assert fail.get("type") == "MltkAssertionError"

    def test_classname_derived_from_name(
        self, tmp_path: Path
    ) -> None:
        """PASS: classname is derived from the dotted test name.

        WHY: CI dashboards group testcases by classname. Correct
        derivation (e.g., "data.schema.check" -> "mltk.data.schema")
        lets teams drill into module-level results.
        Expected: classname matches the module path prefix.
        """
        results = [
            {
                "name": "data.schema.check",
                "passed": True,
                "duration_ms": 10.0,
                "message": "ok",
            },
            {
                "name": "simple_test",
                "passed": True,
                "duration_ms": 5.0,
                "message": "ok",
            },
        ]
        out = tmp_path / "classname.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        cases = tree.getroot().findall(".//testcase")
        classnames = {
            tc.get("name"): tc.get("classname") for tc in cases
        }
        assert classnames["data.schema.check"] == "mltk.data.schema"
        assert classnames["simple_test"] == "mltk"


class TestFormatResultToJunit:
    """Tests for the format_result_to_junit helper.

    Validates individual testcase element construction in isolation.
    """

    def test_passed_result_no_failure(self) -> None:
        """PASS: A passing result produces no <failure> child.

        WHY: A <failure> child on a passing test would cause CI to
        report a false failure.
        Expected: Element has no failure subelement.
        """
        result = {
            "name": "data.completeness",
            "passed": True,
            "duration_ms": 25.0,
            "message": "all columns present",
        }
        elem = format_result_to_junit(result)
        assert elem.tag == "testcase"
        assert elem.find("failure") is None

    def test_failed_result_has_failure(self) -> None:
        """PASS: A failing result includes a <failure> with message.

        WHY: Without the failure element and its message attribute,
        CI systems cannot display what went wrong.
        Expected: failure element with message text.
        """
        result = {
            "name": "model.metric.recall",
            "passed": False,
            "duration_ms": 75.0,
            "message": "recall 0.60 < 0.70",
        }
        elem = format_result_to_junit(result)
        fail = elem.find("failure")
        assert fail is not None
        assert "recall" in fail.get("message", "")
        assert "recall" in (fail.text or "")

    def test_missing_optional_fields(self) -> None:
        """PASS: Result with only name and passed still produces valid XML.

        WHY: Not all result dicts include duration_ms or message. The
        converter must handle missing keys with sensible defaults.
        Expected: time="0.000000", no crash.
        """
        result = {"name": "minimal.test", "passed": True}
        elem = format_result_to_junit(result)
        assert elem.get("time") == "0.000000"
        assert elem.get("name") == "minimal.test"
