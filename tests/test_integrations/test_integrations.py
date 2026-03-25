"""Tests for mltk.integrations — Jira adapter, dedup engine, ticket templates."""

from mltk.integrations.adapter import IssueTrackerAdapter
from mltk.integrations.dedup import TicketDecisionEngine
from mltk.integrations.templates import render_ticket


class MockAdapter(IssueTrackerAdapter):
    """Mock adapter for testing the interface."""

    def __init__(self) -> None:
        self.issues: list[dict] = []

    def create_issue(self, project, title, description, fields=None):
        key = f"{project}-{len(self.issues) + 1}"
        self.issues.append({"key": key, "title": title, "description": description})
        return key

    def search_issues(self, query):
        return [i for i in self.issues if query.lower() in i["title"].lower()]

    def update_issue(self, issue_id, updates):
        for i in self.issues:
            if i["key"] == issue_id:
                i.update(updates)
                return True
        return False


class TestIssueTrackerAdapter:
    """Verify the adapter interface works with mock implementation."""

    def test_create_issue(self) -> None:
        """Create an issue through the adapter interface."""
        adapter = MockAdapter()
        key = adapter.create_issue("ML", "Drift detected", "PSI > 0.2")
        assert key == "ML-1"
        assert len(adapter.issues) == 1

    def test_search_issues(self) -> None:
        """Search issues by title keyword."""
        adapter = MockAdapter()
        adapter.create_issue("ML", "Drift detected in income", "PSI high")
        adapter.create_issue("ML", "Schema mismatch", "Column missing")
        results = adapter.search_issues("drift")
        assert len(results) == 1

    def test_update_issue(self) -> None:
        """Update an existing issue."""
        adapter = MockAdapter()
        adapter.create_issue("ML", "Test failure", "Details")
        assert adapter.update_issue("ML-1", {"status": "resolved"}) is True
        assert adapter.update_issue("ML-999", {"status": "resolved"}) is False


class TestTicketDecisionEngine:
    """Deduplication and spam prevention tests."""

    def test_critical_creates_ticket(self) -> None:
        """CRITICAL severity passes threshold — create ticket."""
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        assert engine.should_create({"test_name": "test_x", "severity": "CRITICAL"}) is True

    def test_low_severity_skipped(self) -> None:
        """LOW severity below threshold — skip ticket."""
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        assert engine.should_create({"test_name": "test_x", "severity": "LOW"}) is False

    def test_dedup_within_cooldown(self) -> None:
        """Same failure within cooldown — skip duplicate."""
        engine = TicketDecisionEngine(severity_threshold="HIGH", cooldown_seconds=3600)
        failure = {"test_name": "test_drift", "severity": "CRITICAL", "assertion_type": "drift"}
        assert engine.should_create(failure) is True  # First time
        assert engine.should_create(failure) is False  # Duplicate within cooldown

    def test_different_failures_not_deduped(self) -> None:
        """Different test names create separate tickets."""
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        f1 = {"test_name": "test_drift", "severity": "CRITICAL"}
        f2 = {"test_name": "test_schema", "severity": "CRITICAL"}
        assert engine.should_create(f1) is True
        assert engine.should_create(f2) is True


class TestTicketTemplates:
    """Ticket template rendering tests."""

    def test_drift_template(self) -> None:
        """Drift detection template renders correctly."""
        ticket = render_ticket(
            "drift_detection",
            test_name="test_income_drift",
            method="PSI",
            statistic=0.35,
            threshold=0.2,
        )
        assert "DRIFT" in ticket["title"]
        assert "PSI" in ticket["description"]
        assert "0.35" in ticket["description"]

    def test_default_template(self) -> None:
        """Default template used for unknown types."""
        ticket = render_ticket(
            "unknown_type",
            test_name="test_something",
            assertion_type="custom",
            message="Something failed",
            severity="HIGH",
        )
        assert "MLTK" in ticket["title"]
        assert "Something failed" in ticket["description"]

    def test_missing_fields_handled(self) -> None:
        """Template gracefully handles missing fields."""
        ticket = render_ticket("model_regression", test_name="test_accuracy")
        assert "test_accuracy" in ticket["description"]
        # Missing fields show {field_name} placeholder
        assert "{metric_name}" in ticket["description"]
