"""Tests for the ``mltk container scan`` CLI command."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from mltk.cli.container import app as container_app

runner = CliRunner()


def _mk_result(
    name: str,
    passed: bool,
    message: str = "",
    details: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal ``AssertionResult``-like stand-in."""
    return SimpleNamespace(
        name=name,
        passed=passed,
        message=message,
        details=details or {},
    )


class TestContainerScanCLI:
    """Tests for the Typer ``container scan`` command."""

    def test_help(self):
        # SCENARIO: --help prints usage and exits 0
        # WHY: Basic smoke test -- command is wired correctly
        # EXPECTED: exit code 0, "image" appears in help output
        result = runner.invoke(
            container_app, ["scan", "--help"],
        )
        assert result.exit_code == 0
        assert "image" in result.output.lower()

    def test_json_output_pass(self):
        # SCENARIO: Both assertions pass, --json requested
        # WHY: Verify success path produces valid JSON on stdout
        # EXPECTED: exit 0, JSON parses, "passed": true
        vuln = _mk_result(
            "container.vulnerabilities",
            passed=True,
            message="No vulnerabilities found",
            details={"critical_count": 0, "high_count": 0},
        )
        secret = _mk_result(
            "container.secrets",
            passed=True,
            message="No secrets found",
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            result = runner.invoke(
                container_app,
                ["scan", "alpine:3.18", "--json"],
            )

        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["passed"] is True
        assert data["image"] == "alpine:3.18"
        assert len(data["results"]) == 2

    def test_json_output_fail(self):
        # SCENARIO: Vulnerability scan fails, --json requested
        # WHY: Failed scan should surface as exit 1 + JSON passed=false
        # EXPECTED: exit 1, JSON passed=false
        vuln = _mk_result(
            "container.vulnerabilities",
            passed=False,
            message="1 CRITICAL CVE found",
            details={
                "critical_count": 1,
                "cves": [
                    {
                        "id": "CVE-2024-1234",
                        "severity": "CRITICAL",
                    },
                ],
            },
        )
        secret = _mk_result(
            "container.secrets", passed=True,
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            result = runner.invoke(
                container_app,
                ["scan", "alpine:3.18", "--json"],
            )

        assert result.exit_code == 1
        data = json.loads(result.stdout)
        assert data["passed"] is False

    def test_junit_xml_output(self, tmp_path: Path):
        # SCENARIO: Passing scan with --junit-xml path
        # WHY: Verify JUnit XML report is written with correct root
        # EXPECTED: exit 0, file exists, root element <testsuites>
        out_path = tmp_path / "container-report.xml"
        vuln = _mk_result(
            "container.vulnerabilities", passed=True,
        )
        secret = _mk_result(
            "container.secrets", passed=True,
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            result = runner.invoke(
                container_app,
                [
                    "scan", "alpine:3.18",
                    "--junit-xml", str(out_path),
                ],
            )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "<testsuites" in content
        assert "mltk.container" in content

    def test_junit_xml_failure_has_failure_element(
        self, tmp_path: Path,
    ):
        # SCENARIO: Failing scan with --junit-xml
        # WHY: Ensure failures surface as <failure> elements
        # EXPECTED: exit 1, XML contains "<failure"
        out_path = tmp_path / "fail-report.xml"
        vuln = _mk_result(
            "container.vulnerabilities",
            passed=False,
            message="3 HIGH CVEs found",
        )
        secret = _mk_result(
            "container.secrets", passed=True,
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            result = runner.invoke(
                container_app,
                [
                    "scan", "alpine:3.18",
                    "--junit-xml", str(out_path),
                ],
            )

        assert result.exit_code == 1
        content = out_path.read_text(encoding="utf-8")
        assert "<failure" in content
        assert "3 HIGH CVEs found" in content

    def test_import_error(self, monkeypatch):
        # SCENARIO: mltk.container.assertions cannot be imported
        # WHY: Must fail gracefully with exit 1, not a traceback
        # EXPECTED: exit code 1
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "mltk.container.assertions":
                raise ImportError("fake missing module")
            return real_import(name, *args, **kwargs)

        # Also ensure any cached module is removed so the
        # lazy import inside the command hits our hook.
        sys.modules.pop("mltk.container.assertions", None)

        monkeypatch.setattr("builtins.__import__", fake_import)

        result = runner.invoke(
            container_app, ["scan", "alpine:3.18"],
        )
        assert result.exit_code == 1

    def test_text_output_pass(self):
        # SCENARIO: Default text output (no --json), all pass
        # WHY: Ensure non-JSON path works and returns exit 0
        # EXPECTED: exit 0
        vuln = _mk_result(
            "container.vulnerabilities", passed=True,
            message="ok",
        )
        secret = _mk_result(
            "container.secrets", passed=True, message="ok",
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            result = runner.invoke(
                container_app, ["scan", "alpine:3.18"],
            )

        assert result.exit_code == 0, result.output

    def test_vuln_scan_raises_exit_2(self):
        # SCENARIO: assert_container_vulnerabilities raises
        # WHY: Exceptions mid-scan indicate infra error (exit 2)
        # EXPECTED: exit code 2
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            side_effect=RuntimeError("trivy binary missing"),
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=_mk_result(
                "container.secrets", passed=True,
            ),
        ):
            result = runner.invoke(
                container_app, ["scan", "alpine:3.18"],
            )
        assert result.exit_code == 2
