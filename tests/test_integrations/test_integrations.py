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

    def test_get_hash_returns_12_char_hex(self) -> None:
        """get_hash returns a 12-character hex string for a failure dict."""
        # SCENARIO: call get_hash with a populated failure dict
        # WHY: the hash is used as a ticket label; it must be stable and the
        #      correct length so downstream tools can rely on its format
        # EXPECTED: 12-character lowercase hex string
        engine = TicketDecisionEngine()
        h = engine.get_hash({
            "test_name": "test_drift",
            "assertion_type": "drift",
            "metric_name": "psi",
        })
        assert isinstance(h, str)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_get_hash_deterministic(self) -> None:
        """Same failure dict always produces the same hash."""
        # SCENARIO: two calls with identical content
        # WHY: dedup logic depends on the hash being content-stable; a
        #      non-deterministic hash would break the cooldown logic
        # EXPECTED: both hashes are identical
        engine = TicketDecisionEngine()
        failure = {"test_name": "test_schema", "assertion_type": "schema", "metric_name": ""}
        assert engine.get_hash(failure) == engine.get_hash(failure)

    def test_get_hash_differs_for_different_failures(self) -> None:
        """Different failure content produces different hashes."""
        # SCENARIO: two failures with different test_name fields
        # WHY: if distinct failures hash to the same value, dedup would
        #      incorrectly suppress unrelated tickets
        # EXPECTED: hashes are not equal
        engine = TicketDecisionEngine()
        h1 = engine.get_hash({"test_name": "test_drift"})
        h2 = engine.get_hash({"test_name": "test_schema"})
        assert h1 != h2

    def test_cooldown_expiry_allows_recreation(self) -> None:
        """After cooldown expires, same failure is allowed again.

        SCENARIO: manually backdate the stored timestamp by more than cooldown_seconds
        WHY: the engine must allow recreation once the cooldown window has elapsed;
             without this, repeated failures would be suppressed indefinitely
        EXPECTED: should_create returns True after simulated expiry
        """
        engine = TicketDecisionEngine(severity_threshold="HIGH", cooldown_seconds=60)
        failure = {"test_name": "test_cooldown", "severity": "CRITICAL"}
        engine.should_create(failure)  # first creation, stores timestamp

        # Backdate the stored timestamp by more than the cooldown window
        content_hash = engine._hash_failure(failure)
        engine._recent_hashes[content_hash] -= 120  # 120s > 60s cooldown

        assert engine.should_create(failure) is True

    def test_zero_cooldown_always_allows_recreation(self) -> None:
        """cooldown_seconds=0 means every call is allowed regardless of recency.

        SCENARIO: same failure submitted twice with zero cooldown
        WHY: zero cooldown is a valid configuration for 'always create a ticket';
             the condition `now - last < 0` is never true so dedup never blocks
        EXPECTED: both calls return True
        """
        engine = TicketDecisionEngine(severity_threshold="HIGH", cooldown_seconds=0)
        failure = {"test_name": "test_zero_cooldown", "severity": "CRITICAL"}
        assert engine.should_create(failure) is True
        assert engine.should_create(failure) is True

    def test_medium_severity_below_high_threshold(self) -> None:
        """MEDIUM severity is below HIGH threshold — ticket skipped.

        SCENARIO: threshold is HIGH, failure is MEDIUM
        WHY: boundary condition — MEDIUM rank (1) must be less than HIGH rank (2)
        EXPECTED: should_create returns False
        """
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        assert engine.should_create({"test_name": "t", "severity": "MEDIUM"}) is False

    def test_high_severity_at_threshold(self) -> None:
        """HIGH severity exactly at HIGH threshold — ticket created.

        SCENARIO: threshold is HIGH, failure is HIGH (equal rank)
        WHY: threshold is inclusive (>=); a failure exactly at threshold must pass
        EXPECTED: should_create returns True
        """
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        assert engine.should_create({"test_name": "t2", "severity": "HIGH"}) is True

    def test_hash_uses_assertion_type_and_metric(self) -> None:
        """Failures differing only in assertion_type or metric_name hash differently.

        SCENARIO: two failures with the same test_name but different assertion_type
        WHY: the hash key is 'test_name#assertion_type#metric_name'; if only
             test_name contributed, two distinct assertions on the same test
             would be incorrectly treated as duplicates
        EXPECTED: hashes differ
        """
        engine = TicketDecisionEngine()
        h1 = engine.get_hash({"test_name": "test_x", "assertion_type": "drift"})
        h2 = engine.get_hash({"test_name": "test_x", "assertion_type": "schema"})
        assert h1 != h2

    def test_unknown_severity_treated_as_lowest_rank(self) -> None:
        """An unrecognised severity string falls back to rank 0 and is skipped.

        SCENARIO: failure dict contains an unrecognised severity value
        WHY: _severity_rank.get(severity, 0) makes the default rank 0, which is
             below any non-LOW threshold — this prevents accidental ticket spam
             from malformed data
        EXPECTED: should_create returns False (below HIGH threshold)
        """
        engine = TicketDecisionEngine(severity_threshold="HIGH")
        assert engine.should_create({"test_name": "t3", "severity": "BANANAS"}) is False


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

    def test_data_quality_template(self) -> None:
        """data_quality template renders [DATA] prefix and key fields.

        SCENARIO: render data_quality with all required fields
        WHY: this template was never exercised; its format string uses
             {assertion_type}, {message}, {timestamp} which are unique to it
        EXPECTED: title starts with [DATA], description includes assertion and message
        """
        ticket = render_ticket(
            "data_quality",
            test_name="test_null_rate",
            assertion_type="assert_no_nulls",
            message="Null rate 12% exceeds threshold 5%",
            timestamp="2026-03-26T10:00:00",
        )
        assert ticket["title"] == "[DATA] test_null_rate"
        assert "assert_no_nulls" in ticket["description"]
        assert "Null rate 12%" in ticket["description"]
        assert "2026-03-26" in ticket["description"]

    def test_model_regression_template(self) -> None:
        """model_regression template renders [MODEL] prefix and numeric fields.

        SCENARIO: render with all regression-specific fields
        WHY: model_regression has unique placeholders ({regression_pct}) not in
             other templates; they must render, not stay as raw placeholder text
        EXPECTED: description contains metric name, expected, actual, and pct values
        """
        ticket = render_ticket(
            "model_regression",
            test_name="test_f1_regression",
            metric_name="f1_score",
            expected=0.92,
            actual=0.81,
            regression_pct=11.96,
        )
        assert "[MODEL]" in ticket["title"]
        assert "f1_score" in ticket["description"]
        assert "0.92" in ticket["description"]
        assert "0.81" in ticket["description"]
        assert "11.96" in ticket["description"]

    def test_bias_violation_template(self) -> None:
        """bias_violation template renders [BIAS] prefix and group information.

        SCENARIO: render with fairness-specific fields including affected groups
        WHY: bias_violation is the only template using {groups} and {disparity};
             untested meant rendering errors there would silently produce garbled tickets
        EXPECTED: title contains [BIAS], groups appear in description
        """
        ticket = render_ticket(
            "bias_violation",
            test_name="test_demographic_parity",
            method="demographic_parity",
            disparity=0.18,
            threshold=0.10,
            groups="[male, female]",
        )
        assert "[BIAS]" in ticket["title"]
        assert "demographic_parity" in ticket["description"]
        assert "0.18" in ticket["description"]
        assert "[male, female]" in ticket["description"]

    def test_title_is_first_line_of_description(self) -> None:
        """render_ticket title is always the first line of description.

        SCENARIO: render any template and compare title to description's first line
        WHY: title is extracted as description.split('\\n')[0]; if the template
             ever changes to put the prefix on a different line, this test catches it
        EXPECTED: ticket['title'] == ticket['description'].splitlines()[0]
        """
        ticket = render_ticket(
            "drift_detection",
            test_name="test_age_drift",
            method="KS",
            statistic=0.45,
            threshold=0.05,
        )
        assert ticket["title"] == ticket["description"].splitlines()[0]

    def test_all_placeholders_when_no_kwargs(self) -> None:
        """render_ticket with no kwargs leaves all placeholders intact.

        SCENARIO: call render_ticket with only the template name
        WHY: _DefaultDict must return '{key}' for every missing key; if it raised
             KeyError instead, callers sending partial data would crash
        EXPECTED: description still contains literal placeholder strings
        """
        ticket = render_ticket("drift_detection")
        assert "{test_name}" in ticket["description"]
        assert "{method}" in ticket["description"]
        assert "{statistic}" in ticket["description"]
