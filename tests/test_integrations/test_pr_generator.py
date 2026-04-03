"""Tests for PullRequestGenerator and render_pr_body."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.result import Severity, TestResult
from mltk.integrations.pr_generator import (
    PullRequestGenerator,
    PullRequestResult,
    render_pr_body,
)
from mltk.scan.finding import FixSuggestion, ScanFinding

# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------


def _make_fix(
    *,
    title: str = "Fix drift threshold",
    description: str = "Adjust threshold to 0.3",
    confidence: str = "high",
    category: str = "code",
    code_snippet: str = "threshold = 0.3",
) -> FixSuggestion:
    return FixSuggestion(
        category=category,
        title=title,
        description=description,
        confidence=confidence,
        code_snippet=code_snippet,
    )


def _make_result(
    *,
    name: str = "test_drift",
    passed: bool = False,
    severity: Severity = Severity.WARNING,
    message: str = "PSI > 0.2",
) -> TestResult:
    return TestResult(name=name, passed=passed, severity=severity, message=message)


def _make_finding(
    *,
    scanner_name: str = "drift",
    result: TestResult | None = None,
    fix: FixSuggestion | None = None,
) -> ScanFinding:
    r = result or _make_result()
    f = fix or _make_fix()
    return ScanFinding(
        result=r,
        assertion_fn=lambda: r,
        assertion_args=(),
        assertion_kwargs={},
        scanner_name=scanner_name,
        suggested_fixes=[f],
    )


def _make_mock_worktree(branch: str = "mltk/fix-drift-abc12345") -> MagicMock:
    """Return a MagicMock that behaves as a GitWorktree context manager."""
    mock_wt = MagicMock()
    mock_wt.__enter__ = MagicMock(return_value=mock_wt)
    mock_wt.__exit__ = MagicMock(return_value=False)
    mock_wt.branch = branch
    return mock_wt


def _make_github_mock(pr_response: dict | None = None) -> MagicMock:
    response = pr_response or {
        "html_url": "https://github.com/owner/repo/pull/7",
        "number": 7,
        "draft": True,
    }
    gh = MagicMock()
    gh.create_pull_request.return_value = response
    return gh


# ---------------------------------------------------------------------------
# TestPullRequestResult
# ---------------------------------------------------------------------------


class TestPullRequestResult:
    # WHY: dataclass must expose all four fields correctly
    # SCENARIO: construct with explicit values
    # EXPECTED: attributes match what was passed
    def test_pr_result_fields(self):
        result = PullRequestResult(
            url="https://github.com/owner/repo/pull/42",
            branch="mltk/fix-drift-abc12345",
            number=42,
            draft=True,
        )
        assert result.url == "https://github.com/owner/repo/pull/42"
        assert result.branch == "mltk/fix-drift-abc12345"
        assert result.number == 42
        assert result.draft is True


# ---------------------------------------------------------------------------
# TestPullRequestGenerator
# ---------------------------------------------------------------------------


_WT_PATCH = "mltk.experiment.worktree.GitWorktree"
_FGR_PATCH = "mltk.experiment.worktree.find_git_root"


def _wt_ctx(mock_wt: MagicMock, fgr_root: Path = Path("/repo")):
    """Return a pair of patch context managers for GitWorktree + find_git_root.

    Patches at the source module (mltk.experiment.worktree) because create_pr()
    imports lazily with ``from mltk.experiment.worktree import ...``.
    """
    return (
        patch(_WT_PATCH, return_value=mock_wt),
        patch(_FGR_PATCH, return_value=fgr_root),
    )


class TestPullRequestGenerator:
    # WHY: end-to-end happy path — worktree + adapter should produce a valid result
    # SCENARIO: valid finding + fix, all mocks succeed
    # EXPECTED: PullRequestResult with correct url, number, draft, branch
    def test_create_pr_happy_path(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            result = gen.create_pr(finding, fix)

        assert result.url == "https://github.com/owner/repo/pull/7"
        assert result.number == 7
        assert result.draft is True
        gh.create_pull_request.assert_called_once()

    # WHY: branch name must follow the prescribed format for traceability
    # SCENARIO: scanner_name="drift"
    # EXPECTED: branch starts with "mltk/fix-drift-" followed by 8 hex chars
    def test_create_pr_branch_naming(self):
        gh = _make_github_mock()
        finding = _make_finding(scanner_name="drift")
        fix = _make_fix()
        captured: list[str] = []

        def capture_branch(repo_root, branch_name=None, **_kw):
            captured.append(branch_name or "")
            return _make_mock_worktree(branch=branch_name or "")

        _, fgr_patch = _wt_ctx(_make_mock_worktree())
        with patch(_WT_PATCH, side_effect=capture_branch), fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix)

        assert len(captured) == 1
        branch = captured[0]
        import re
        assert re.match(r"^mltk/fix-drift-[0-9a-f]{8}$", branch), f"Unexpected branch: {branch}"

    # WHY: scanner names with spaces/symbols must be sanitised to valid git branch chars
    # SCENARIO: scanner_name="data quality check"
    # EXPECTED: branch contains "data-quality-check" with dashes
    def test_create_pr_sanitizes_scanner_name(self):
        gh = _make_github_mock()
        finding = _make_finding(scanner_name="data quality check")
        fix = _make_fix()
        captured: list[str] = []

        def capture_branch(repo_root, branch_name=None, **_kw):
            captured.append(branch_name or "")
            return _make_mock_worktree(branch=branch_name or "")

        _, fgr_patch = _wt_ctx(_make_mock_worktree())
        with patch(_WT_PATCH, side_effect=capture_branch), fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix)

        branch = captured[0]
        assert "data-quality-check" in branch
        assert " " not in branch

    # WHY: empty code_snippet means nothing to commit — must reject early
    # SCENARIO: fix has code_snippet=""
    # EXPECTED: ValueError raised before any git operations
    def test_create_pr_empty_snippet_raises(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix(code_snippet="")

        gen = PullRequestGenerator(gh, repo_root=Path("/repo"))
        with pytest.raises(ValueError, match="code_snippet is empty"):
            gen.create_pr(finding, fix)

    # WHY: push must happen inside the `with` block so the branch is still present
    # SCENARIO: track order of calls on the mock worktree
    # EXPECTED: push call precedes __exit__
    def test_create_pr_push_inside_context(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix()
        call_order: list[str] = []

        mock_wt = MagicMock()

        def enter(_self=None):
            call_order.append("enter")
            return mock_wt

        def exit_fn(*args):
            call_order.append("exit")
            return False

        def push_fn(*cmd, **kw):
            if "push" in cmd:
                call_order.append("push")
            return MagicMock()

        mock_wt.__enter__ = MagicMock(side_effect=enter)
        mock_wt.__exit__ = MagicMock(side_effect=exit_fn)
        mock_wt.branch = "mltk/fix-drift-abc12345"
        mock_wt.run_in_worktree = MagicMock(side_effect=push_fn)

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix)

        assert "push" in call_order
        assert "exit" in call_order
        push_idx = call_order.index("push")
        exit_idx = call_order.index("exit")
        assert push_idx < exit_idx, "push must happen before __exit__"

    # WHY: draft=False should be forwarded verbatim to the adapter
    # SCENARIO: caller requests a non-draft PR
    # EXPECTED: create_pull_request called with draft=False
    def test_create_pr_draft_false(self):
        gh = _make_github_mock(
            {"html_url": "https://github.com/o/r/pull/1", "number": 1, "draft": False}
        )
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix, draft=False)

        _, kwargs = gh.create_pull_request.call_args
        assert kwargs["draft"] is False

    # WHY: labels must reach the adapter so the PR can be triaged correctly
    # SCENARIO: caller supplies labels=["autofix", "drift"]
    # EXPECTED: create_pull_request receives those labels
    def test_create_pr_custom_labels(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix, labels=["autofix", "drift"])

        _, kwargs = gh.create_pull_request.call_args
        assert kwargs["labels"] == ["autofix", "drift"]

    # WHY: a push failure must surface as a RuntimeError with a clear message
    # SCENARIO: git push subprocess exits non-zero
    # EXPECTED: RuntimeError raised mentioning the branch
    def test_create_pr_push_failure(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        def run_side_effect(*cmd, **kw):
            if "push" in cmd:
                raise subprocess.CalledProcessError(
                    returncode=1, cmd=list(cmd), stderr="remote: error"
                )
            return MagicMock()

        mock_wt.run_in_worktree = MagicMock(side_effect=run_side_effect)

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            with pytest.raises(RuntimeError, match="push"):
                gen.create_pr(finding, fix)

    # WHY: API errors from the adapter should propagate unmodified
    # SCENARIO: create_pull_request raises RuntimeError
    # EXPECTED: the same RuntimeError propagates out of create_pr
    def test_create_pr_api_error(self):
        gh = MagicMock()
        gh.create_pull_request.side_effect = RuntimeError("GitHub API error 422: Validation Failed")
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            with pytest.raises(RuntimeError, match="422"):
                gen.create_pr(finding, fix)

    # WHY: custom target_file must be used as the write path in the worktree
    # SCENARIO: target_file="fixes/my_fix.py"
    # EXPECTED: write_file is called with that exact path
    def test_create_pr_custom_target_file(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix(code_snippet="x = 1")
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix, target_file="fixes/my_fix.py")

        mock_wt.write_file.assert_called_once_with("fixes/my_fix.py", "x = 1")

    # WHY: PR title format must follow the [mltk-autofix] convention for filtering
    # SCENARIO: fix.title="Fix drift threshold"
    # EXPECTED: title passed to adapter is "[mltk-autofix] Fix drift threshold"
    def test_create_pr_title_format(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix(title="Fix drift threshold")
        mock_wt = _make_mock_worktree()

        wt_patch, fgr_patch = _wt_ctx(mock_wt)
        with wt_patch, fgr_patch:
            gen = PullRequestGenerator(gh)
            gen.create_pr(finding, fix)

        _, kwargs = gh.create_pull_request.call_args
        assert kwargs["title"] == "[mltk-autofix] Fix drift threshold"

    # WHY: when repo_root is provided explicitly, find_git_root should not be called
    # SCENARIO: PullRequestGenerator initialized with explicit repo_root
    # EXPECTED: find_git_root is never invoked
    def test_create_pr_explicit_repo_root(self):
        gh = _make_github_mock()
        finding = _make_finding()
        fix = _make_fix()
        mock_wt = _make_mock_worktree()

        with (
            patch(_WT_PATCH, return_value=mock_wt) as mock_cls,
            patch(_FGR_PATCH) as mock_fgr,
        ):
            gen = PullRequestGenerator(gh, repo_root=Path("/my/repo"))
            gen.create_pr(finding, fix)

        mock_fgr.assert_not_called()
        # GitWorktree was constructed with the provided repo_root
        args, _ = mock_cls.call_args
        assert args[0] == Path("/my/repo")


# ---------------------------------------------------------------------------
# TestRenderPrBody
# ---------------------------------------------------------------------------


class TestRenderPrBody:
    # WHY: body must contain all expected sections when snippet is present
    # SCENARIO: full finding + fix with non-empty code_snippet
    # EXPECTED: Finding, Fix, Code sections and footer all present
    def test_render_pr_body_full(self):
        finding = _make_finding(scanner_name="drift")
        fix = _make_fix(
            title="Adjust threshold",
            description="Lower PSI threshold",
            confidence="high",
            category="code",
            code_snippet="threshold = 0.1",
        )
        body = render_pr_body(finding, fix)

        assert "## Finding" in body
        assert "drift" in body
        assert "## Fix" in body
        assert "Adjust threshold" in body
        assert "Lower PSI threshold" in body
        assert "## Code" in body
        assert "```python" in body
        assert "threshold = 0.1" in body
        assert "Auto-generated by mltk experiment runner" in body

    # WHY: empty code_snippet must suppress the Code section entirely
    # SCENARIO: fix.code_snippet=""
    # EXPECTED: "## Code" not in body
    def test_render_pr_body_no_snippet(self):
        finding = _make_finding()
        fix = _make_fix(code_snippet="")
        body = render_pr_body(finding, fix)

        assert "## Finding" in body
        assert "## Fix" in body
        assert "## Code" not in body
        assert "```python" not in body

    # WHY: severity from the TestResult must appear in the body for triage
    # SCENARIO: result has severity=Severity.CRITICAL
    # EXPECTED: "CRITICAL" appears in the Finding section
    def test_render_pr_body_severity(self):
        result = _make_result(severity=Severity.CRITICAL, message="Accuracy drop > 20%")
        finding = _make_finding(result=result)
        fix = _make_fix()
        body = render_pr_body(finding, fix)

        assert "CRITICAL" in body

    # WHY: the finding message must surface in the body for full context
    # SCENARIO: result.message = "PSI > 0.2"
    # EXPECTED: body contains that exact message
    def test_render_pr_body_message(self):
        result = _make_result(message="PSI > 0.2 on feature_age")
        finding = _make_finding(result=result)
        fix = _make_fix()
        body = render_pr_body(finding, fix)

        assert "PSI > 0.2 on feature_age" in body

    # WHY: confidence and category help reviewers gauge urgency and effort
    # SCENARIO: fix with confidence="medium", category="config"
    # EXPECTED: both appear in the body
    def test_render_pr_body_confidence_and_category(self):
        finding = _make_finding()
        fix = _make_fix(confidence="medium", category="config", code_snippet="")
        body = render_pr_body(finding, fix)

        assert "medium" in body
        assert "config" in body

    # WHY: the footer is always required so reviewers know not to auto-merge
    # SCENARIO: any valid finding + fix
    # EXPECTED: footer separator and disclaimer always present
    def test_render_pr_body_footer_always_present(self):
        finding = _make_finding()
        fix = _make_fix(code_snippet="")
        body = render_pr_body(finding, fix)

        assert "---" in body
        assert "Review before merging" in body
