"""Tests for IssueLinker and _finding_to_failure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.core.result import Severity, TestResult
from mltk.integrations.adapter import IssueTrackerAdapter
from mltk.integrations.dedup import TicketDecisionEngine
from mltk.integrations.issue_linker import IssueLinker, _finding_to_failure
from mltk.scan.finding import ScanFinding

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_finding(
    name: str = "test_drift",
    passed: bool = False,
    severity: Severity = Severity.WARNING,
    message: str = "PSI > 0.2 on feature X",
    scanner_name: str = "drift",
) -> ScanFinding:
    """Create a minimal ScanFinding for test use."""
    result = TestResult(
        name=name,
        passed=passed,
        severity=severity,
        message=message,
    )
    return ScanFinding(
        result=result,
        assertion_fn=lambda: result,
        assertion_args=(),
        assertion_kwargs={},
        scanner_name=scanner_name,
    )


def _make_adapter(
    create_return: str = "https://github.com/owner/repo/issues/42",
    update_return: bool = True,
) -> MagicMock:
    """Create a mock IssueTrackerAdapter."""
    mock_adapter = MagicMock(spec=IssueTrackerAdapter)
    mock_adapter.create_issue.return_value = create_return
    mock_adapter.update_issue.return_value = update_return
    return mock_adapter


def _make_dedup(should_create: bool = True) -> MagicMock:
    """Create a mock TicketDecisionEngine."""
    mock_dedup = MagicMock(spec=TicketDecisionEngine)
    mock_dedup.should_create.return_value = should_create
    return mock_dedup


# ---------------------------------------------------------------------------
# TestIssueLinkerCreateFromFinding
# ---------------------------------------------------------------------------

class TestIssueLinkerCreateFromFinding:

    def test_create_from_finding_happy_path(self):
        # WHY: Full happy-path — dedup approves, adapter called, issue created.
        # SCENARIO: finding with WARNING severity, dedup returns True.
        # EXPECTED: adapter.create_issue is called exactly once.
        adapter = _make_adapter()
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        linker.create_from_finding(_make_finding(), project="ML")

        adapter.create_issue.assert_called_once()

    def test_create_from_finding_returns_issue_key(self):
        # WHY: Return value should be exactly what the adapter returns.
        # SCENARIO: adapter.create_issue returns a GitHub URL.
        # EXPECTED: create_from_finding returns that same URL.
        expected_key = "https://github.com/owner/repo/issues/42"
        adapter = _make_adapter(create_return=expected_key)
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        result = linker.create_from_finding(_make_finding(), project="ML")

        assert result == expected_key

    def test_create_from_finding_dedup_skip(self):
        # WHY: Dedup should short-circuit before the adapter is called.
        # SCENARIO: dedup.should_create returns False (duplicate / low severity).
        # EXPECTED: returns None and adapter.create_issue is NOT called.
        adapter = _make_adapter()
        dedup = _make_dedup(should_create=False)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        result = linker.create_from_finding(_make_finding(), project="ML")

        assert result is None
        adapter.create_issue.assert_not_called()

    def test_create_from_finding_default_dedup(self):
        # WHY: When no dedup is passed, a default TicketDecisionEngine is created.
        # SCENARIO: IssueLinker constructed without explicit dedup argument.
        # EXPECTED: linker._dedup is a TicketDecisionEngine instance.
        adapter = _make_adapter()
        linker = IssueLinker(adapter=adapter)

        assert isinstance(linker._dedup, TicketDecisionEngine)

    def test_create_from_finding_custom_template(self):
        # WHY: The template argument should be forwarded to render_ticket.
        # SCENARIO: template="drift_detection" requested explicitly.
        # EXPECTED: render_ticket is called with template_name="drift_detection".
        adapter = _make_adapter()
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        with patch("mltk.integrations.issue_linker.render_ticket") as mock_render:
            mock_render.return_value = {"title": "T", "description": "D"}
            linker.create_from_finding(
                _make_finding(), project="ML", template="drift_detection"
            )

        assert mock_render.call_args[0][0] == "drift_detection"

    def test_create_from_finding_default_template(self):
        # WHY: The default template should be "default" when not specified.
        # SCENARIO: create_from_finding called without template kwarg.
        # EXPECTED: render_ticket receives "default" as the first argument.
        adapter = _make_adapter()
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        with patch("mltk.integrations.issue_linker.render_ticket") as mock_render:
            mock_render.return_value = {"title": "T", "description": "D"}
            linker.create_from_finding(_make_finding(), project="ML")

        assert mock_render.call_args[0][0] == "default"

    def test_create_from_finding_adapter_error(self):
        # WHY: Exceptions from the adapter should propagate to the caller.
        # SCENARIO: adapter.create_issue raises RuntimeError (e.g. API failure).
        # EXPECTED: RuntimeError propagates from create_from_finding.
        adapter = _make_adapter()
        adapter.create_issue.side_effect = RuntimeError("API unavailable")
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        with pytest.raises(RuntimeError, match="API unavailable"):
            linker.create_from_finding(_make_finding(), project="ML")

    def test_create_from_finding_passes_project(self):
        # WHY: The project key must be forwarded unchanged to the adapter.
        # SCENARIO: project="DATA-TEAM" passed to create_from_finding.
        # EXPECTED: adapter.create_issue is called with "DATA-TEAM" as the first arg.
        adapter = _make_adapter()
        dedup = _make_dedup(should_create=True)
        linker = IssueLinker(adapter=adapter, dedup=dedup)

        linker.create_from_finding(_make_finding(), project="DATA-TEAM")

        call_args = adapter.create_issue.call_args
        assert call_args[0][0] == "DATA-TEAM"


# ---------------------------------------------------------------------------
# TestIssueLinkerLinkPr
# ---------------------------------------------------------------------------

class TestIssueLinkerLinkPr:

    def test_link_pr_happy_path(self):
        # WHY: link_pr must call update_issue with the correct structure.
        # SCENARIO: valid issue_id and pr_url provided, adapter returns True.
        # EXPECTED: adapter.update_issue is called with the issue_id.
        adapter = _make_adapter(update_return=True)
        linker = IssueLinker(adapter=adapter)

        linker.link_pr("ISSUE-42", "https://github.com/owner/repo/pull/7")

        adapter.update_issue.assert_called_once()
        call_args = adapter.update_issue.call_args
        assert call_args[0][0] == "ISSUE-42"

    def test_link_pr_returns_true(self):
        # WHY: Return value must reflect the adapter's success signal.
        # SCENARIO: adapter.update_issue returns True.
        # EXPECTED: link_pr returns True.
        adapter = _make_adapter(update_return=True)
        linker = IssueLinker(adapter=adapter)

        result = linker.link_pr("ISSUE-42", "https://github.com/owner/repo/pull/7")

        assert result is True

    def test_link_pr_returns_false(self):
        # WHY: A failed update must surface as False, not silently ignored.
        # SCENARIO: adapter.update_issue returns False (e.g. issue not found).
        # EXPECTED: link_pr returns False.
        adapter = _make_adapter(update_return=False)
        linker = IssueLinker(adapter=adapter)

        result = linker.link_pr("ISSUE-99", "https://github.com/owner/repo/pull/3")

        assert result is False

    def test_link_pr_comment_format(self):
        # WHY: The comment body must follow the exact documented format.
        # SCENARIO: link_pr called with a GitHub PR URL.
        # EXPECTED: comment contains "**Fix PR:**" prefix and mltk footer.
        adapter = _make_adapter(update_return=True)
        linker = IssueLinker(adapter=adapter)
        pr_url = "https://github.com/owner/repo/pull/99"

        linker.link_pr("ISSUE-1", pr_url)

        call_kwargs = adapter.update_issue.call_args[0][1]
        comment = call_kwargs["comment"]
        assert f"**Fix PR:** {pr_url}" in comment
        assert "mltk experiment runner" in comment


# ---------------------------------------------------------------------------
# TestFindingToFailure
# ---------------------------------------------------------------------------

class TestFindingToFailure:

    def test_finding_to_failure_mapping(self):
        # WHY: All required keys must be present with correct values.
        # SCENARIO: finding with known name, severity, message, scanner_name.
        # EXPECTED: returned dict maps fields correctly.
        finding = _make_finding(
            name="test_bias",
            severity=Severity.CRITICAL,
            message="Disparate impact > 0.1",
            scanner_name="bias",
        )

        failure = _finding_to_failure(finding)

        assert failure["test_name"] == "test_bias"
        assert failure["severity"] == "CRITICAL"
        assert failure["assertion_type"] == "bias"
        assert failure["metric_name"] == "bias"
        assert failure["message"] == "Disparate impact > 0.1"
        assert "timestamp" in failure

    def test_finding_to_failure_severity_mapped(self):
        # WHY: TicketDecisionEngine uses CRITICAL/HIGH/MEDIUM/LOW but
        # Severity enum uses CRITICAL/WARNING/INFO. Must translate.
        # SCENARIO: finding has Severity.WARNING → should map to "HIGH".
        # EXPECTED: failure dict contains "HIGH".
        finding = _make_finding(severity=Severity.WARNING)

        failure = _finding_to_failure(finding)

        assert failure["severity"] == "HIGH"

    def test_finding_to_failure_empty_scanner(self):
        # WHY: scanner_name may be an empty string for ad-hoc findings.
        # SCENARIO: ScanFinding constructed with scanner_name="".
        # EXPECTED: assertion_type and metric_name are both empty strings.
        finding = _make_finding(scanner_name="")

        failure = _finding_to_failure(finding)

        assert failure["assertion_type"] == ""
        assert failure["metric_name"] == ""

    def test_finding_to_failure_timestamp_is_iso_string(self):
        # WHY: Templates and trackers expect timestamp as an ISO 8601 string.
        # SCENARIO: TestResult has a default datetime.now() timestamp.
        # EXPECTED: failure["timestamp"] is a non-empty ISO 8601 string.
        finding = _make_finding()

        failure = _finding_to_failure(finding)

        assert isinstance(failure["timestamp"], str)
        assert len(failure["timestamp"]) > 0
        # ISO 8601 format contains "T" separator
        assert "T" in failure["timestamp"]
