"""Tests for mltk compliance-gap CLI command.

Validates that the compliance-gap command correctly reads test results JSON,
runs gap analysis across supported compliance frameworks, and produces the
expected terminal output.

Tests use subprocess invocation (via ``_run_cli``) to exercise the full
CLI path, matching the existing test_cli.py approach.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results_json(path: Path, results: list[dict]) -> Path:
    """Write a minimal mltk results JSON file."""
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def _run_cli(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the mltk CLI via subprocess and return the result."""
    cli_args = list(args)
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {cli_args!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# TestComplianceGap
# ---------------------------------------------------------------------------

class TestComplianceGap:
    """Tests for the compliance-gap command."""

    def test_valid_json_shows_output(self, tmp_path: Path) -> None:
        """PASS: compliance-gap with valid JSON prints gap analysis header.

        WHY: The command's primary job is to read results and print a gap
        analysis summary. If the header is missing, the output format is broken.
        Expected: Output contains the gap analysis header line.
        """
        results = [
            {
                "name": "data.schema.valid",
                "passed": True,
                "severity": "info",
                "message": "Schema valid",
                "details": {},
                "duration_ms": 5.0,
            },
            {
                "name": "model.bias.demographic_parity",
                "passed": True,
                "severity": "critical",
                "message": "Bias check passed",
                "details": {},
                "duration_ms": 10.0,
            },
        ]
        results_path = _make_results_json(tmp_path / "results.json", results)
        result = _run_cli("compliance-gap", str(results_path))
        assert result.returncode == 0
        assert "Compliance Gap Analysis" in result.stdout

    def test_missing_file_errors_gracefully(self, tmp_path: Path) -> None:
        """PASS: compliance-gap exits 1 when results JSON file does not exist.

        WHY: A missing file is a common user error. The command must exit
        with code 1 and print a helpful message instead of an unhandled
        traceback.
        Expected: Process exits with code 1 and mentions "not found".
        """
        missing = tmp_path / "does_not_exist.json"
        assert not missing.exists()
        result = _run_cli("compliance-gap", str(missing))
        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_framework_owasp_filters_output(self, tmp_path: Path) -> None:
        """PASS: --framework owasp shows only OWASP section.

        WHY: Users working on a specific framework need to filter output.
        If the filter doesn't work, they get irrelevant noise from other
        frameworks that obscures the OWASP-specific gaps.
        Expected: Output contains OWASP but not EU AI Act.
        """
        results = [
            {
                "name": "llm.hallucination.rag",
                "passed": True,
                "severity": "info",
                "message": "No hallucination detected",
                "details": {},
                "duration_ms": 15.0,
            },
        ]
        results_path = _make_results_json(tmp_path / "results.json", results)
        result = _run_cli("compliance-gap", str(results_path), "--framework", "owasp")
        assert result.returncode == 0
        assert "OWASP LLM Top 10" in result.stdout
        assert "EU AI Act" not in result.stdout

    def test_framework_eu_ai_act_filters_output(self, tmp_path: Path) -> None:
        """PASS: --framework eu-ai-act shows only EU AI Act section.

        WHY: Same as OWASP filter — ensures each framework runs in isolation
        when explicitly requested.
        Expected: Output contains EU AI Act but not OWASP.
        """
        results = [
            {
                "name": "data.no_nulls",
                "passed": True,
                "severity": "info",
                "message": "No nulls",
                "details": {},
                "duration_ms": 5.0,
            },
        ]
        results_path = _make_results_json(tmp_path / "results.json", results)
        result = _run_cli("compliance-gap", str(results_path), "--framework", "eu-ai-act")
        assert result.returncode == 0
        assert "EU AI Act" in result.stdout
        assert "OWASP" not in result.stdout

    def test_framework_fda_shows_coverage(self, tmp_path: Path) -> None:
        """PASS: --framework fda shows FDA coverage with matching test names.

        WHY: FDA gap analysis uses a simple prefix match on test names.
        If prefix matching is broken, users get 0 coverage even when they
        have FDA tests.
        Expected: Output shows FDA section with the test count > 0.
        """
        results = [
            {
                "name": "fda.audit_trail",
                "passed": True,
                "severity": "info",
                "message": "Audit trail valid",
                "details": {},
                "duration_ms": 3.0,
            },
            {
                "name": "pipeline.checksum.model",
                "passed": True,
                "severity": "info",
                "message": "Checksum valid",
                "details": {},
                "duration_ms": 2.0,
            },
        ]
        results_path = _make_results_json(tmp_path / "results.json", results)
        result = _run_cli("compliance-gap", str(results_path), "--framework", "fda")
        assert result.returncode == 0
        assert "FDA" in result.stdout
        assert "2 tests" in result.stdout

    def test_all_frameworks_runs_everything(self, tmp_path: Path) -> None:
        """PASS: --framework all (default) shows all framework sections.

        WHY: The default mode must show a unified summary across all
        frameworks. If any framework section is missing, users get an
        incomplete compliance picture.
        Expected: Output contains both EU AI Act and OWASP sections.
        """
        results = [
            {
                "name": "data.schema",
                "passed": True,
                "severity": "info",
                "message": "ok",
                "details": {},
                "duration_ms": 5.0,
            },
        ]
        results_path = _make_results_json(tmp_path / "results.json", results)
        result = _run_cli("compliance-gap", str(results_path))
        assert result.returncode == 0
        assert "EU AI Act" in result.stdout
        assert "OWASP LLM Top 10" in result.stdout
        assert "FDA" in result.stdout

    def test_invalid_framework_exits_with_error(self, tmp_path: Path) -> None:
        """PASS: Unknown --framework value exits 1 with helpful message.

        WHY: A typo in the framework name must not silently succeed with
        no output. The command must tell the user which values are valid.
        Expected: Process exits with code 1 and lists valid options.
        """
        results_path = _make_results_json(tmp_path / "results.json", [])
        result = _run_cli(
            "compliance-gap", str(results_path), "--framework", "invalid-fw"
        )
        assert result.returncode == 1
        assert "unknown framework" in result.stdout.lower() or "unknown" in result.stderr.lower()

    def test_empty_results_shows_all_gaps(self, tmp_path: Path) -> None:
        """PASS: Empty results JSON shows maximum gaps for all frameworks.

        WHY: When no tests exist, every requirement is a gap. The command
        must correctly report zero coverage rather than crashing on an
        empty list.
        Expected: EU AI Act coverage shows 0/N articles.
        """
        results_path = _make_results_json(tmp_path / "results.json", [])
        result = _run_cli("compliance-gap", str(results_path), "--framework", "eu-ai-act")
        assert result.returncode == 0
        assert "0/" in result.stdout
        assert "0%" in result.stdout

    def test_wrapped_dict_format(self, tmp_path: Path) -> None:
        """PASS: compliance-gap handles {"results": [...]} dict format.

        WHY: Some exporters wrap results in a dict. The command must
        support both flat array and wrapped dict formats for compatibility.
        Expected: Output contains gap analysis header (no crash).
        """
        wrapped = {
            "results": [
                {
                    "name": "data.drift.psi",
                    "passed": True,
                    "severity": "info",
                    "message": "No drift",
                    "details": {},
                    "duration_ms": 5.0,
                },
            ]
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(wrapped), encoding="utf-8")
        result = _run_cli("compliance-gap", str(results_path))
        assert result.returncode == 0
        assert "Compliance Gap Analysis" in result.stdout
