"""Tests for mltk_workflow tool, workflow_hint in responses,
severity-conditional next steps, and error fallback parameters.

Covers the S91 additions: the 11th MCP tool (mltk_workflow),
workflow_hint presence across all success responses,
severity-based branching in mltk_scan, and fallback_parameters
in error responses for mid-chain recovery.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_has_workflow_hint,
    assert_ok,
    call_tool,
)

# ----------------------------------------------------------
# Shared test data
# ----------------------------------------------------------

_FINDING_JSON = json.dumps({
    "scanner_name": "drift",
    "result": {
        "name": "psi_check",
        "passed": False,
        "severity": "warning",
        "message": "PSI exceeds threshold",
    },
    "suggested_fixes": [
        {
            "category": "code",
            "title": "Retrain on recent data",
            "description": "Drift detected.",
            "confidence": "high",
            "code_snippet": "model.fit(X_recent, y_recent)",
        },
    ],
})

_FIX_JSON = json.dumps({
    "category": "code",
    "title": "Retrain on recent data",
    "description": "Drift detected.",
    "confidence": "high",
    "code_snippet": "model.fit(X_recent, y_recent)",
})


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def _patch_scan():
    """Context manager that mocks mltk.scan at source."""
    return patch.multiple(
        "mltk.scan",
        ScanConfig=MagicMock(),
        ScanEngine=MagicMock(),
        create=True,
    )


def _make_findings_json(severity, tmp_path, *, extra=None):
    """Write a scan report JSON file with a single finding."""
    findings = [{"severity": severity, "message": "test"}]
    if extra:
        findings.extend(extra)
    report = {
        "findings": findings,
        "scanners_run": ["drift"],
        "duration_ms": 100,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path


# ----------------------------------------------------------
# Class 1: TestMltkWorkflow
# ----------------------------------------------------------

class TestMltkWorkflow:
    """Tests for the mltk_workflow informational tool."""

    def test_workflow_returns_ok(self):
        # SCENARIO: Call mltk_workflow with no arguments
        # WHY: Informational tool must always succeed
        # EXPECTED: status=ok
        result = call_tool("mltk_workflow")
        assert_ok(result)

    def test_workflow_has_pipeline_paths(self):
        # SCENARIO: Check pipeline key in response
        # WHY: Pipeline map is the core output -- agents route
        #      based on these named paths
        # EXPECTED: "pipeline" dict with 5 named paths
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert "pipeline" in result
        pipeline = result["pipeline"]
        expected_paths = {
            "critical_path",
            "medium_path",
            "low_path",
            "eval_path",
            "test_path",
        }
        assert set(pipeline.keys()) == expected_paths

    def test_workflow_critical_path_starts_with_scan(self):
        # SCENARIO: Inspect critical_path ordering
        # WHY: Scan is always the entry point for the
        #      remediation pipeline
        # EXPECTED: critical_path[0] == "mltk_scan"
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert result["pipeline"]["critical_path"][0] == (
            "mltk_scan"
        )

    def test_workflow_critical_path_ends_with_issue(self):
        # SCENARIO: Inspect critical_path last element
        # WHY: Issue creation is the final action in the
        #      critical remediation path
        # EXPECTED: critical_path[-1] == "mltk_create_issue"
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert result["pipeline"]["critical_path"][-1] == (
            "mltk_create_issue"
        )

    def test_workflow_has_decision_tree(self):
        # SCENARIO: Check decision_tree key
        # WHY: Agents use the decision tree text to understand
        #      branching logic without calling multiple tools
        # EXPECTED: decision_tree is a non-empty string
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert "decision_tree" in result
        assert isinstance(result["decision_tree"], str)
        assert len(result["decision_tree"]) > 0

    def test_workflow_tool_count_is_eleven(self):
        # SCENARIO: Check tool_count matches the 11-tool suite
        # WHY: Agents use tool_count to verify they have the
        #      complete tool set before starting a pipeline
        # EXPECTED: tool_count == 11
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert result["tool_count"] == 11

    def test_workflow_suggested_next_step(self):
        # SCENARIO: Check suggested_next_step guidance
        # WHY: After reading workflow info the natural next
        #      action is to start scanning
        # EXPECTED: suggested_next_step mentions "mltk_scan"
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert "suggested_next_step" in result
        assert "mltk_scan" in result["suggested_next_step"]

    def test_workflow_has_workflow_hint(self):
        # SCENARIO: Verify workflow_hint in workflow response
        # WHY: Even informational responses must carry routing
        #      metadata for agent orchestrators
        # EXPECTED: workflow_hint with position="info"
        result = call_tool("mltk_workflow")
        assert_ok(result)
        assert "workflow_hint" in result
        hint = result["workflow_hint"]
        assert hint["position"] == "info"
        assert isinstance(hint["next_tools"], list)


# ----------------------------------------------------------
# Class 2: TestSeverityConditionalNextStep
# ----------------------------------------------------------

class TestSeverityConditionalNextStep:
    """Tests for severity-based branching in mltk_scan."""

    def test_scan_critical_mentions_suggest(self, tmp_path):
        # SCENARIO: Scan a JSON report with a critical finding
        # WHY: Critical findings must route the agent toward
        #      mltk_suggest for immediate fix generation
        # EXPECTED: suggested_next_step mentions "mltk_suggest"
        report = _make_findings_json("critical", tmp_path)
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_suggest" in result["suggested_next_step"]

    def test_scan_warning_mentions_issue(self, tmp_path):
        # SCENARIO: Scan a JSON report with a warning finding
        # WHY: Warnings are not urgent enough for immediate
        #      fix but should be tracked in an issue
        # EXPECTED: suggested_next_step mentions
        #           "mltk_create_issue"
        report = _make_findings_json("warning", tmp_path)
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_create_issue" in (
            result["suggested_next_step"]
        )

    def test_scan_info_mentions_report(self, tmp_path):
        # SCENARIO: Scan a JSON report with an info finding
        # WHY: Info-level findings need no action, just
        #      documentation via a report
        # EXPECTED: suggested_next_step mentions "mltk_report"
        report = _make_findings_json("info", tmp_path)
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_report" in result["suggested_next_step"]

    def test_scan_mixed_severity_uses_highest(self, tmp_path):
        # SCENARIO: Scan JSON with both critical and info
        # WHY: When multiple severities exist the agent must
        #      act on the most severe (critical wins)
        # EXPECTED: suggested_next_step mentions "mltk_suggest"
        report = _make_findings_json(
            "critical",
            tmp_path,
            extra=[{"severity": "info", "message": "low"}],
        )
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_suggest" in result["suggested_next_step"]

    def test_scan_empty_findings_defaults_to_report(
        self, tmp_path,
    ):
        # SCENARIO: Scan a JSON report with zero findings
        # WHY: Empty findings list has no severity to branch
        #      on, so the default info-level path applies
        # EXPECTED: suggested_next_step mentions "mltk_report"
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [],
            "scanners_run": ["drift"],
            "duration_ms": 100,
        }))
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_report" in result["suggested_next_step"]

    def test_scan_unknown_severity_defaults_to_report(
        self, tmp_path,
    ):
        # SCENARIO: Scan a JSON report with non-standard
        #           severity "error"
        # WHY: Unknown severity must not crash; it falls
        #      through to the info-level default path
        # EXPECTED: suggested_next_step mentions "mltk_report"
        report = _make_findings_json("error", tmp_path)
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert "mltk_report" in result["suggested_next_step"]

    def test_scan_file_mode_next_step_unchanged(
        self, tmp_path,
    ):
        # SCENARIO: Scan a .py file (not a JSON report)
        # WHY: File-mode scan uses the original static
        #      suggested_next_step, not severity branching
        # EXPECTED: suggested_next_step contains "mltk scan"
        py_file = tmp_path / "model.py"
        py_file.write_text("x = 1\n")
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(py_file),
            )
        assert_ok(result)
        assert "mltk scan" in result["suggested_next_step"]


# ----------------------------------------------------------
# Class 3: TestWorkflowHintPresence
# ----------------------------------------------------------

class TestWorkflowHintPresence:
    """Verify workflow_hint in success responses."""

    def test_scan_has_workflow_hint(self, tmp_path):
        # SCENARIO: Scan a JSON report file
        # WHY: Scan is the pipeline entry point and must
        #      carry routing metadata for the next step
        # EXPECTED: workflow_hint with position="start"
        report = _make_findings_json("info", tmp_path)
        with _patch_scan():
            result = call_tool(
                "mltk_scan", path=str(report),
            )
        assert_ok(result)
        assert_has_workflow_hint(result)
        assert result["workflow_hint"]["position"] == "start"

    def test_suggest_has_workflow_hint(self):
        # SCENARIO: Call mltk_suggest with a valid finding
        # WHY: After getting suggestions the agent needs to
        #      know whether to experiment or create a PR
        # EXPECTED: workflow_hint present with valid shape
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_JSON,
        )
        assert_ok(result)
        assert_has_workflow_hint(result)

    def test_experiment_has_workflow_hint(self):
        # SCENARIO: Call mltk_experiment with a valid finding
        # WHY: After ranking fixes the agent must know to
        #      proceed to PR creation or issue filing
        # EXPECTED: workflow_hint present with valid shape
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_JSON,
        )
        assert_ok(result)
        assert_has_workflow_hint(result)

    def test_report_has_workflow_hint(self):
        # SCENARIO: Generate a report with results
        # WHY: After reporting the agent may need to scan
        #      again or share the report
        # EXPECTED: workflow_hint present with valid shape
        results = json.dumps([
            {"name": "t1", "passed": True},
        ])
        result = call_tool(
            "mltk_report",
            title="Test Run",
            results_json=results,
        )
        assert_ok(result)
        assert_has_workflow_hint(result)

    def test_list_has_workflow_hint(self):
        # SCENARIO: List available assertions
        # WHY: After listing, the agent should know to pick
        #      an assertion or start scanning
        # EXPECTED: workflow_hint present with valid shape
        _PATCH = "mltk.cli._discovery.discover_assertions"
        mock_data = {
            "data": [
                {
                    "name": "assert_no_drift",
                    "doc": "Detect drift",
                },
            ],
        }
        with patch(_PATCH, return_value=mock_data):
            result = call_tool("mltk_list")
        assert_ok(result)
        assert_has_workflow_hint(result)


# ----------------------------------------------------------
# Class 4: TestErrorFallbackParameters
# ----------------------------------------------------------

class TestErrorFallbackParameters:
    """Tests for fallback_parameters in error responses."""

    def test_error_without_fallback_has_no_key(self):
        # SCENARIO: Standard scan error (bad path)
        # WHY: Not all errors have a fallback -- simple
        #      validation errors should not carry one
        # EXPECTED: no "fallback_parameters" key in response
        result = call_tool(
            "mltk_scan",
            path="/does/not/exist/model.py",
        )
        assert_error(result)
        assert "fallback_parameters" not in result

    @patch(
        "mltk.integrations.pr_generator.PullRequestGenerator",
    )
    @patch(
        "mltk.integrations.github_adapter.GitHubIssuesAdapter",
    )
    @patch(
        "mltk.experiment.worktree.find_git_root",
        return_value=Path("/repo"),
    )
    @patch(
        "mltk.experiment.worktree.git_available",
        return_value=True,
    )
    def test_error_with_fallback_has_tool_key(
        self,
        mock_git_avail,
        mock_git_root,
        mock_adapter,
        mock_gen,
    ):
        # SCENARIO: create_pr fails with a RuntimeError
        # WHY: PR push failures should offer a fallback so
        #      the agent can recover mid-chain
        # EXPECTED: fallback_parameters dict has "tool" key
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.side_effect = RuntimeError(
            "Failed to push branch"
        )

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo="owner/repo",
        )

        assert_error(result)
        assert "fallback_parameters" in result
        assert "tool" in result["fallback_parameters"]

    @patch(
        "mltk.integrations.pr_generator.PullRequestGenerator",
    )
    @patch(
        "mltk.integrations.github_adapter.GitHubIssuesAdapter",
    )
    @patch(
        "mltk.experiment.worktree.find_git_root",
        return_value=Path("/repo"),
    )
    @patch(
        "mltk.experiment.worktree.git_available",
        return_value=True,
    )
    def test_create_pr_fallback_points_to_issue(
        self,
        mock_git_avail,
        mock_git_root,
        mock_adapter,
        mock_gen,
    ):
        # SCENARIO: create_pr push failure
        # WHY: When a PR cannot be created the agent should
        #      fall back to filing an issue instead
        # EXPECTED: fallback tool is "mltk_create_issue"
        mock_gen_inst = MagicMock()
        mock_gen.return_value = mock_gen_inst
        mock_gen_inst.create_pr.side_effect = RuntimeError(
            "Failed to push branch"
        )

        result = call_tool(
            "mltk_create_pr",
            finding_json=_FINDING_JSON,
            fix_json=_FIX_JSON,
            repo="owner/repo",
        )

        assert_error(result)
        fb = result["fallback_parameters"]
        assert fb["tool"] == "mltk_create_issue"
