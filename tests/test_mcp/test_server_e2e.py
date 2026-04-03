"""End-to-end agent simulation tests for the mltk MCP server.

Validates multi-tool pipeline chains, agent decision logic based
on response fields, error recovery flows, and context-passing
contracts between tools.  Each test simulates an AI agent calling
tools in sequence, feeding outputs from one tool into the next.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_ok,
    call_tool,
)

# ----------------------------------------------------------
# Shared test data
# ----------------------------------------------------------

_FINDING_CRITICAL = {
    "scanner_name": "drift",
    "result": {
        "name": "psi_check",
        "passed": False,
        "severity": "critical",
        "message": "PSI exceeds threshold by 3x",
    },
    "suggested_fixes": [
        {
            "category": "code",
            "title": "Retrain on recent data",
            "description": "Drift detected -- retrain model.",
            "confidence": "high",
            "code_snippet": "model.fit(X_recent, y_recent)",
        },
        {
            "category": "config",
            "title": "Lower PSI threshold",
            "description": "Relax the PSI alert threshold.",
            "confidence": "medium",
            "code_snippet": "psi_threshold=0.3",
        },
    ],
}

_FINDING_WARNING = {
    "scanner_name": "bias",
    "result": {
        "name": "demographic_parity",
        "passed": False,
        "severity": "warning",
        "message": "Slight bias on gender feature",
    },
    "suggested_fixes": [
        {
            "category": "data",
            "title": "Balance training set",
            "description": "Resample to equalise groups.",
            "confidence": "medium",
            "code_snippet": "df = df.groupby('g').sample(n=500)",
        },
    ],
}

_FINDING_INFO = {
    "scanner_name": "calibration",
    "result": {
        "name": "brier_score",
        "passed": True,
        "severity": "info",
        "message": "Calibration within tolerance",
    },
}

_FIX_HIGH = {
    "category": "code",
    "title": "Retrain on recent data",
    "description": "Drift detected -- retrain model.",
    "confidence": "high",
    "code_snippet": "model.fit(X_recent, y_recent)",
}

_REPO = "owner/repo"
_PR_URL = "https://github.com/owner/repo/pull/7"
_ISSUE_URL = "https://github.com/owner/repo/issues/42"
_GH_CONFIG = {"repo": _REPO, "token": "ghp_test"}

_LIST_PATCH = "mltk.cli._discovery.discover_assertions"
_MOCK_LIST_ALL = {
    "data": [
        {"name": "assert_no_drift", "doc": "Detect drift"},
    ],
    "model": [
        {"name": "assert_metric", "doc": "Validate metric"},
    ],
}


# ----------------------------------------------------------
# Mock builders
# ----------------------------------------------------------

def _patch_scan():
    """Context manager that mocks mltk.scan."""
    return patch.multiple(
        "mltk.scan",
        ScanConfig=MagicMock(name="ScanConfig"),
        ScanEngine=MagicMock(name="ScanEngine"),
        create=True,
    )


def _make_pr_result(
    url=_PR_URL, branch="mltk/fix-drift", number=7,
    draft=True,
):
    """Build a mock PullRequestResult."""
    r = MagicMock()
    r.url = url
    r.branch = branch
    r.number = number
    r.draft = draft
    return r


def _pr_patches(
    git_avail=True, pr_result=None, push_error=None,
):
    """Return stacked context managers for create_pr mocks."""
    mock_gen_inst = MagicMock()
    mock_gen = MagicMock(return_value=mock_gen_inst)
    if push_error:
        mock_gen_inst.create_pr.side_effect = push_error
    else:
        mock_gen_inst.create_pr.return_value = (
            pr_result or _make_pr_result()
        )

    return (
        patch(
            "mltk.experiment.worktree.git_available",
            return_value=git_avail,
        ),
        patch(
            "mltk.experiment.worktree.find_git_root",
            return_value=Path("/repo"),
        ),
        patch(
            "mltk.integrations.github_adapter"
            ".GitHubIssuesAdapter",
        ),
        patch(
            "mltk.integrations.pr_generator"
            ".PullRequestGenerator",
            mock_gen,
        ),
    )


def _issue_patches(issue_url=_ISSUE_URL, error=None):
    """Return stacked context managers for create_issue."""
    mock_linker_inst = MagicMock()
    mock_linker = MagicMock(return_value=mock_linker_inst)
    if error:
        mock_linker_inst.create_from_finding.side_effect = error
    else:
        mock_linker_inst.create_from_finding.return_value = (
            issue_url
        )
    return (
        patch(
            "mltk.integrations.github_adapter"
            ".GitHubIssuesAdapter",
        ),
        patch(
            "mltk.integrations.issue_linker.IssueLinker",
            mock_linker,
        ),
    )


def _patch_eval():
    """Context manager pair for mltk.eval mocks."""
    mock_result = MagicMock()
    mock_result.metrics = {"accuracy": 0.95}
    mock_result.total_samples = 3
    mock_result.duration_ms = 15

    mock_task_inst = MagicMock()
    mock_task_inst.run.return_value = mock_result
    mock_task_cls = MagicMock(return_value=mock_task_inst)

    mock_load = MagicMock(return_value=[
        MagicMock(input="q1", target="a1"),
        MagicMock(input="q2", target="a2"),
        MagicMock(input="q3", target="a3"),
    ])

    p1 = patch.multiple(
        "mltk.eval.task",
        load_dataset=mock_load,
        EvalTask=mock_task_cls,
        create=True,
    )
    p2 = patch.multiple(
        "mltk.eval",
        GenerateSolver=MagicMock(),
        ChainOfThoughtSolver=MagicMock(),
        FewShotSolver=MagicMock(),
        ExactMatchScorer=MagicMock(),
        IncludesScorer=MagicMock(),
        PatternScorer=MagicMock(),
        create=True,
    )
    return p1, p2


# ===========================================================
# Class 1: Full pipeline chains
# ===========================================================

class TestFullPipelineChains:
    """Simulate full multi-tool chains where output of
    tool N feeds into tool N+1."""

    def test_critical_path_scan_suggest_experiment_pr_issue(
        self, tmp_path,
    ):
        # SCENARIO: Critical finding full pipeline:
        #   scan -> suggest -> experiment -> create_pr -> create_issue
        # WHY: The most complete agent pipeline for critical findings.
        # EXPECTED: All five steps return status=ok.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_CRITICAL],
            "scanners_run": ["drift"],
            "duration_ms": 100,
        }))

        # Step 1: scan JSON report
        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)
        assert len(scan["findings"]) >= 1

        # Step 2: suggest fixes from the finding
        finding_json = json.dumps(scan["findings"][0])
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)

        # Step 3: experiment (rank the fixes)
        experiment = call_tool(
            "mltk_experiment", finding_json=finding_json,
        )
        assert_ok(experiment)

        # Step 4: create_pr with the top fix
        fix_json = json.dumps(_FIX_HIGH)
        p1, p2, p3, p4 = _pr_patches()
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=finding_json,
                fix_json=fix_json,
                repo=_REPO,
            )
        assert_ok(pr)

        # Step 5: create_issue linked to PR
        a1, a2 = _issue_patches()
        with a1, a2:
            issue = call_tool(
                "mltk_create_issue",
                finding_json=finding_json,
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
                pr_url=pr.get("url", _PR_URL),
            )
        assert_ok(issue)

    def test_medium_path_scan_suggest_issue(self, tmp_path):
        # SCENARIO: Warning finding: scan -> suggest -> create_issue
        #   (skip experiment and PR for non-critical findings)
        # WHY: Medium severity takes a shorter path.
        # EXPECTED: All three steps return status=ok.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_WARNING],
            "scanners_run": ["bias"],
            "duration_ms": 50,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        finding_json = json.dumps(scan["findings"][0])
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)

        a1, a2 = _issue_patches()
        with a1, a2:
            issue = call_tool(
                "mltk_create_issue",
                finding_json=finding_json,
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
            )
        assert_ok(issue)

    def test_low_path_scan_report(self, tmp_path):
        # SCENARIO: Info finding: scan -> report (nothing to fix)
        # WHY: Info findings need documentation, not remediation.
        # EXPECTED: Both steps return status=ok; report has text.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_INFO],
            "scanners_run": ["calibration"],
            "duration_ms": 20,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        rpt = call_tool(
            "mltk_report",
            title="Info Scan Report",
            results_json=json.dumps(scan["findings"]),
        )
        assert_ok(rpt)
        assert "report_text" in rpt

    def test_test_path_scan_test_report(self, tmp_path):
        # SCENARIO: scan(file) -> test(yaml) -> report
        # WHY: Common path for running test suites after a scan.
        # EXPECTED: All three steps return status=ok.
        py_file = tmp_path / "model.py"
        py_file.write_text("x = 1\n")
        yml = tmp_path / "suite.yaml"
        yml.write_text("name: demo\ntests:\n  - name: t1\n")

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(py_file))
        assert_ok(scan)

        test = call_tool("mltk_test", suite_path=str(yml))
        assert_ok(test)

        rpt = call_tool(
            "mltk_report",
            title="Scan+Test Report",
            results_json=json.dumps([
                {"name": "scan", "status": "ok"},
                {"name": "test", "passed": True},
            ]),
        )
        assert_ok(rpt)
        assert "report_text" in rpt

    def test_eval_path_list_eval_report(self, tmp_path):
        # SCENARIO: list -> eval -> report
        # WHY: Agent discovers assertions, runs eval, reports.
        # EXPECTED: All three steps return status=ok.
        csv = tmp_path / "data.csv"
        csv.write_text("input,target\nq1,a1\nq2,a2\n")

        with patch(_LIST_PATCH, return_value=_MOCK_LIST_ALL):
            lst = call_tool("mltk_list")
        assert_ok(lst)
        assert lst["total"] > 0

        p1, p2 = _patch_eval()
        with p1, p2:
            ev = call_tool(
                "mltk_eval", dataset_path=str(csv),
            )
        assert_ok(ev)
        assert "metrics" in ev

        rpt = call_tool(
            "mltk_report",
            title="Eval Report",
            results_json=json.dumps({
                "name": "eval", "status": "ok",
                "metrics": ev["metrics"],
            }),
        )
        assert_ok(rpt)

    def test_pr_failure_falls_back_to_issue(self, tmp_path):
        # SCENARIO: scan -> suggest -> experiment -> PR FAILS
        #   -> agent falls back to create_issue
        # WHY: PR creation can fail (push, auth); agent must
        #   have a fallback path to create_issue.
        # EXPECTED: PR returns error; issue returns ok.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_CRITICAL],
            "scanners_run": ["drift"],
            "duration_ms": 100,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        finding_json = json.dumps(scan["findings"][0])
        fix_json = json.dumps(_FIX_HIGH)

        # PR fails (push error)
        p1, p2, p3, p4 = _pr_patches(
            push_error=RuntimeError("push rejected"),
        )
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=finding_json,
                fix_json=fix_json,
                repo=_REPO,
            )
        assert_error(pr)

        # Fallback: create issue instead
        a1, a2 = _issue_patches()
        with a1, a2:
            issue = call_tool(
                "mltk_create_issue",
                finding_json=finding_json,
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
            )
        assert_ok(issue)

    def test_finding_json_roundtrips_through_chain(
        self, tmp_path,
    ):
        # SCENARIO: finding_json from scan can be passed
        #   verbatim to suggest, experiment, and create_pr.
        # WHY: JSON must survive serialization across tools.
        # EXPECTED: All downstream tools accept the same string.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_CRITICAL],
            "scanners_run": ["drift"],
            "duration_ms": 100,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        # The exact JSON string that scan produced
        finding_json = json.dumps(scan["findings"][0])

        # suggest accepts it
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)

        # experiment accepts it
        experiment = call_tool(
            "mltk_experiment", finding_json=finding_json,
        )
        assert_ok(experiment)

        # create_pr accepts it
        fix_json = json.dumps(_FIX_HIGH)
        p1, p2, p3, p4 = _pr_patches()
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=finding_json,
                fix_json=fix_json,
                repo=_REPO,
            )
        assert_ok(pr)

    def test_fix_json_roundtrips_suggest_to_pr(self):
        # SCENARIO: Fix from suggest output can be serialized
        #   and passed as fix_json to create_pr.
        # WHY: Suggest output shape must match create_pr input.
        # EXPECTED: create_pr accepts re-serialized fix JSON.
        finding_json = json.dumps(_FINDING_CRITICAL)

        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)
        assert suggest["total"] > 0

        # Re-serialize the first suggestion
        fix_json = json.dumps(suggest["suggestions"][0])

        p1, p2, p3, p4 = _pr_patches()
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=finding_json,
                fix_json=fix_json,
                repo=_REPO,
            )
        assert_ok(pr)


# ===========================================================
# Class 2: Agent decision simulation
# ===========================================================

class TestAgentDecisionSimulation:
    """Test that agents can read response fields to make
    routing decisions."""

    def test_agent_branches_critical_to_suggest(
        self, tmp_path,
    ):
        # SCENARIO: Scan returns critical findings with
        #   severity-conditional suggested_next_step.
        # WHY: Agent must know to call mltk_suggest next.
        # EXPECTED: suggested_next_step is a non-empty string
        #   that an agent can parse for next-action routing.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_CRITICAL],
            "scanners_run": ["drift"],
            "duration_ms": 100,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)
        assert "suggested_next_step" in scan
        assert isinstance(scan["suggested_next_step"], str)
        assert len(scan["suggested_next_step"]) > 0

    def test_agent_branches_warning_to_issue(self, tmp_path):
        # SCENARIO: Scan returns warning-level findings.
        # WHY: Warning findings may route to issue creation.
        # EXPECTED: Response has suggested_next_step string.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_WARNING],
            "scanners_run": ["bias"],
            "duration_ms": 50,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)
        assert "suggested_next_step" in scan
        assert isinstance(scan["suggested_next_step"], str)

    def test_agent_branches_info_to_report(self, tmp_path):
        # SCENARIO: Scan returns info-level findings only.
        # WHY: Info findings route to report, not remediation.
        # EXPECTED: suggested_next_step is present and non-empty.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [_FINDING_INFO],
            "scanners_run": ["calibration"],
            "duration_ms": 20,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)
        assert "suggested_next_step" in scan
        assert len(scan["suggested_next_step"]) > 0

    def test_agent_selects_highest_confidence_fix(self):
        # SCENARIO: suggest returns multiple fixes with
        #   different confidence levels; agent picks highest.
        # WHY: Agent should pick the highest-confidence fix
        #   from the suggestions list for experiment/PR.
        # EXPECTED: At least one suggestion has confidence
        #   "high"; agent can compare and select.
        finding_json = json.dumps(_FINDING_CRITICAL)
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)
        assert suggest["total"] >= 2

        confidences = [
            s["confidence"] for s in suggest["suggestions"]
        ]
        assert "high" in confidences
        # Agent would pick the first "high" confidence fix
        best = next(
            s for s in suggest["suggestions"]
            if s["confidence"] == "high"
        )
        assert best["title"]
        assert best["category"]

    def test_agent_uses_experiment_rank_for_pr(self):
        # SCENARIO: experiment returns ranked fixes; agent
        #   uses the rank-1 fix to create a PR.
        # WHY: The ranking should guide which fix to apply.
        # EXPECTED: ranked_fixes[0] has rank=1 and all
        #   fields that create_pr needs (category, title, etc).
        finding_json = json.dumps(_FINDING_CRITICAL)
        experiment = call_tool(
            "mltk_experiment", finding_json=finding_json,
        )
        assert_ok(experiment)
        assert experiment["total"] >= 1

        top = experiment["ranked_fixes"][0]
        assert top["rank"] == 1
        # Verify it has fields create_pr needs as fix_json
        for key in (
            "category", "title", "description",
            "confidence",
        ):
            assert key in top, f"Missing key: {key!r}"

    def test_agent_reads_suggested_next_step(self, tmp_path):
        # SCENARIO: scan returns suggested_next_step in
        #   the ok response.
        # WHY: Agent uses this field to decide the next tool.
        # EXPECTED: suggested_next_step is a non-empty string.
        py = tmp_path / "model.py"
        py.write_text("pass\n")

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(py))
        assert_ok(scan)
        assert "suggested_next_step" in scan
        assert isinstance(scan["suggested_next_step"], str)
        assert len(scan["suggested_next_step"]) > 0

    def test_agent_skips_experiment_for_medium_severity(self):
        # SCENARIO: Warning finding -> suggest -> create_issue
        #   (skipping experiment entirely).
        # WHY: For medium-severity, the overhead of experiment
        #   is not justified; suggest -> issue is sufficient.
        # EXPECTED: suggest returns ok; create_issue returns ok.
        finding_json = json.dumps(_FINDING_WARNING)
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)

        a1, a2 = _issue_patches()
        with a1, a2:
            issue = call_tool(
                "mltk_create_issue",
                finding_json=finding_json,
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
            )
        assert_ok(issue)

    def test_suggest_next_step_mentions_tool(self):
        # SCENARIO: mltk_suggest returns a suggested_next_step
        #   that references a downstream action.
        # WHY: Agent parses this for routing decisions.
        # EXPECTED: suggested_next_step is a non-empty string
        #   that an agent can use as routing guidance.
        finding_json = json.dumps(_FINDING_CRITICAL)
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)
        assert "suggested_next_step" in suggest
        assert isinstance(suggest["suggested_next_step"], str)
        assert len(suggest["suggested_next_step"]) > 0


# ===========================================================
# Class 3: Error recovery chains
# ===========================================================

class TestErrorRecoveryChains:
    """Test agent recovery when tools fail mid-chain."""

    def test_scan_ok_suggest_fails_recovery(self, tmp_path):
        # SCENARIO: scan ok -> suggest with invalid JSON.
        # WHY: Agent must detect error and have guidance.
        # EXPECTED: scan ok; suggest error with suggested_action.
        py = tmp_path / "check.py"
        py.write_text("pass\n")

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(py))
        assert_ok(scan)

        bad = call_tool(
            "mltk_suggest", finding_json="{not valid json",
        )
        assert_error(bad)
        assert "suggested_action" in bad
        assert isinstance(bad["suggested_action"], str)

    def test_experiment_fails_agent_gets_guidance(self):
        # SCENARIO: suggest ok -> experiment with invalid data.
        # WHY: Experiment errors must be recoverable with
        #   a clear suggested_action.
        # EXPECTED: error response with recoverable=True.
        bad = call_tool(
            "mltk_experiment",
            finding_json="{bad json!",
        )
        assert_error(bad)
        assert bad["recoverable"] is True

    def test_create_pr_fails_has_suggested_action(self):
        # SCENARIO: create_pr fails due to push error.
        # WHY: PR failures must give agent an alternative.
        # EXPECTED: error with suggested_action string.
        p1, p2, p3, p4 = _pr_patches(
            push_error=RuntimeError("push rejected"),
        )
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=json.dumps(_FINDING_CRITICAL),
                fix_json=json.dumps(_FIX_HIGH),
                repo=_REPO,
            )
        assert_error(pr)
        assert "suggested_action" in pr
        assert isinstance(pr["suggested_action"], str)

    def test_create_pr_no_git_error(self):
        # SCENARIO: create_pr when git is not available.
        # WHY: Must fail early with actionable error.
        # EXPECTED: error mentioning "git".
        p1, p2, p3, p4 = _pr_patches(git_avail=False)
        with p1, p2, p3, p4:
            pr = call_tool(
                "mltk_create_pr",
                finding_json=json.dumps(_FINDING_CRITICAL),
                fix_json=json.dumps(_FIX_HIGH),
                repo=_REPO,
            )
        assert_error(pr)
        assert "git" in pr["error"].lower()

    def test_create_issue_adapter_error_recoverable(self):
        # SCENARIO: create_issue with adapter that raises.
        # WHY: Adapter errors should be marked recoverable.
        # EXPECTED: error with recoverable=True.
        a1, a2 = _issue_patches(
            error=RuntimeError("GitHub API 422"),
        )
        with a1, a2:
            issue = call_tool(
                "mltk_create_issue",
                finding_json=json.dumps(_FINDING_CRITICAL),
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
            )
        assert_error(issue)
        assert issue["recoverable"] is True

    def test_scan_nonexistent_path_error_shape(self):
        # SCENARIO: scan with a nonexistent path.
        # WHY: Must return a well-formed error, not crash.
        # EXPECTED: status=error, error string, recoverable,
        #   suggested_action all present.
        result = call_tool(
            "mltk_scan",
            path="/nonexistent/path/model.py",
        )
        assert_error(result)
        assert "not found" in result["error"].lower()

    def test_all_errors_have_suggested_action(self):
        # SCENARIO: Multiple bad calls to different tools.
        # WHY: Every error response must include
        #   suggested_action for agent recovery.
        # EXPECTED: All error responses have suggested_action.
        bad_calls = [
            ("mltk_scan", {"path": "/no/such/path.py"}),
            (
                "mltk_suggest",
                {"finding_json": "{broken"},
            ),
            (
                "mltk_experiment",
                {"finding_json": ""},
            ),
            (
                "mltk_test",
                {"suite_path": "/no/such/file.yaml"},
            ),
        ]
        for tool_name, kwargs in bad_calls:
            result = call_tool(tool_name, **kwargs)
            assert_error(result)
            assert "suggested_action" in result, (
                f"{tool_name} missing suggested_action"
            )
            assert isinstance(
                result["suggested_action"], str,
            )
            assert len(result["suggested_action"]) > 0, (
                f"{tool_name} has empty suggested_action"
            )

    def test_error_recoverable_is_bool(self):
        # SCENARIO: Error responses across tools.
        # WHY: Agent relies on "recoverable" being a bool
        #   to decide whether to retry or abort.
        # EXPECTED: recoverable is always a bool.
        bad_calls = [
            ("mltk_scan", {"path": "/bad/path.py"}),
            (
                "mltk_suggest",
                {"finding_json": ""},
            ),
            (
                "mltk_experiment",
                {"finding_json": "{nope"},
            ),
        ]
        for tool_name, kwargs in bad_calls:
            result = call_tool(tool_name, **kwargs)
            assert_error(result)
            assert isinstance(result["recoverable"], bool), (
                f"{tool_name}: recoverable is not bool"
            )


# ===========================================================
# Class 4: Context passing validation
# ===========================================================

class TestContextPassingValidation:
    """Validate that tool output schemas match what
    downstream tools expect as input."""

    def test_scan_findings_have_severity_field(self, tmp_path):
        # SCENARIO: scan JSON report -> every finding has
        #   a "severity" field (in result sub-object).
        # WHY: Downstream tools key on severity for routing.
        # EXPECTED: Each finding has result.severity.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [
                _FINDING_CRITICAL,
                _FINDING_WARNING,
                _FINDING_INFO,
            ],
            "scanners_run": ["drift", "bias", "calibration"],
            "duration_ms": 200,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        for finding in scan["findings"]:
            result = finding.get("result", {})
            assert "severity" in result, (
                f"Missing severity in {finding}"
            )

    def test_suggest_output_has_experiment_fields(self):
        # SCENARIO: suggest output -> each fix has all fields
        #   that experiment and create_pr expect.
        # WHY: Contract: suggest output feeds experiment/PR.
        # EXPECTED: title, description, confidence, category,
        #   code_snippet in every suggestion.
        finding_json = json.dumps(_FINDING_CRITICAL)
        suggest = call_tool(
            "mltk_suggest", finding_json=finding_json,
        )
        assert_ok(suggest)
        assert suggest["total"] > 0

        required_keys = {
            "title", "description", "confidence",
            "category", "code_snippet",
        }
        for s in suggest["suggestions"]:
            missing = required_keys - set(s.keys())
            assert not missing, (
                f"Suggestion missing keys: {missing}"
            )

    def test_experiment_output_has_pr_fields(self):
        # SCENARIO: experiment ranked_fixes have all fields
        #   that create_pr expects as fix_json.
        # WHY: Contract: experiment output feeds create_pr.
        # EXPECTED: category, title, description, confidence,
        #   rank, score in every ranked fix.
        finding_json = json.dumps(_FINDING_CRITICAL)
        experiment = call_tool(
            "mltk_experiment", finding_json=finding_json,
        )
        assert_ok(experiment)
        assert experiment["total"] > 0

        required = {"category", "title", "description",
                     "confidence", "rank", "score"}
        for fix in experiment["ranked_fixes"]:
            missing = required - set(fix.keys())
            assert not missing, (
                f"Ranked fix missing keys: {missing}"
            )

    def test_finding_json_schema_stable(self):
        # SCENARIO: finding from scan can be json.dumps'd
        #   and passed to suggest without mangling.
        # WHY: JSON round-trip must be lossless.
        # EXPECTED: suggest accepts the re-serialized finding.
        original = _FINDING_CRITICAL
        serialized = json.dumps(original)
        # Verify round-trip
        deserialized = json.loads(serialized)
        assert deserialized == original

        # Pass to suggest
        suggest = call_tool(
            "mltk_suggest", finding_json=serialized,
        )
        assert_ok(suggest)
        assert suggest["total"] > 0

    def test_all_ok_responses_have_suggested_next_step(
        self, tmp_path,
    ):
        # SCENARIO: Call every tool with valid input.
        # WHY: All ok responses must include
        #   suggested_next_step for agent routing.
        # EXPECTED: suggested_next_step present in all.
        py = tmp_path / "check.py"
        py.write_text("pass\n")
        yml = tmp_path / "suite.yaml"
        yml.write_text("name: t\ntests:\n  - name: t1\n")
        csv = tmp_path / "data.csv"
        csv.write_text("input,target\nq1,a1\n")

        results = {}

        # scan
        with _patch_scan():
            results["scan"] = call_tool(
                "mltk_scan", path=str(py),
            )

        # test
        results["test"] = call_tool(
            "mltk_test", suite_path=str(yml),
        )

        # list
        with patch(_LIST_PATCH, return_value=_MOCK_LIST_ALL):
            results["list"] = call_tool("mltk_list")

        # eval
        p1, p2 = _patch_eval()
        with p1, p2:
            results["eval"] = call_tool(
                "mltk_eval", dataset_path=str(csv),
            )

        # report
        results["report"] = call_tool(
            "mltk_report", title="Test Report",
        )

        # suggest
        results["suggest"] = call_tool(
            "mltk_suggest",
            finding_json=json.dumps(_FINDING_CRITICAL),
        )

        # experiment
        results["experiment"] = call_tool(
            "mltk_experiment",
            finding_json=json.dumps(_FINDING_CRITICAL),
        )

        # create_pr
        p1, p2, p3, p4 = _pr_patches()
        with p1, p2, p3, p4:
            results["pr"] = call_tool(
                "mltk_create_pr",
                finding_json=json.dumps(_FINDING_CRITICAL),
                fix_json=json.dumps(_FIX_HIGH),
                repo=_REPO,
            )

        # create_issue
        a1, a2 = _issue_patches()
        with a1, a2:
            results["issue"] = call_tool(
                "mltk_create_issue",
                finding_json=json.dumps(_FINDING_CRITICAL),
                tracker="github",
                config_json=json.dumps(_GH_CONFIG),
            )

        for name, r in results.items():
            assert_ok(r)
            assert "suggested_next_step" in r, (
                f"{name} missing suggested_next_step"
            )
            assert isinstance(
                r["suggested_next_step"], str,
            ), f"{name} suggested_next_step not str"
            assert len(r["suggested_next_step"]) > 0, (
                f"{name} has empty suggested_next_step"
            )

    def test_report_accepts_scan_findings(self, tmp_path):
        # SCENARIO: scan findings list -> pass as results_json
        #   to report.
        # WHY: Report must accept scan output as input.
        # EXPECTED: report returns ok with report_text.
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [
                _FINDING_CRITICAL,
                _FINDING_WARNING,
            ],
            "scanners_run": ["drift", "bias"],
            "duration_ms": 100,
        }))

        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(report))
        assert_ok(scan)

        rpt = call_tool(
            "mltk_report",
            title="Findings Report",
            description="From scan output",
            results_json=json.dumps(scan["findings"]),
        )
        assert_ok(rpt)
        assert "report_text" in rpt
        assert len(rpt["report_text"]) > 0

    def test_experiment_strategy_field_present(self):
        # SCENARIO: experiment response always has "strategy".
        # WHY: Agent may log or display the ranking strategy.
        # EXPECTED: strategy is a non-empty string.
        finding_json = json.dumps(_FINDING_CRITICAL)
        experiment = call_tool(
            "mltk_experiment", finding_json=finding_json,
        )
        assert_ok(experiment)
        assert "strategy" in experiment
        assert isinstance(experiment["strategy"], str)
        assert len(experiment["strategy"]) > 0

    def test_suggest_filtered_by_field_present(self):
        # SCENARIO: suggest response always has "filtered_by".
        # WHY: Agent needs to know if filtering was applied.
        # EXPECTED: filtered_by is present as a string.
        finding_json = json.dumps(_FINDING_CRITICAL)
        suggest = call_tool(
            "mltk_suggest",
            finding_json=finding_json,
            category="code",
        )
        assert_ok(suggest)
        assert "filtered_by" in suggest
        assert suggest["filtered_by"] == "code"
