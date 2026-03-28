"""Tests for the audit logging system on the mltk server.

This test suite verifies that the ``AuditLogger`` correctly records
events, filters them, exports to CSV, and that the
``assert_audit_log_complete`` assertion catches missing actions.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime

from mltk.server.audit_log import AuditLogger, assert_audit_log_complete

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ISO 8601 with timezone offset pattern: 2024-01-15T10:30:00+00:00
_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _sample_key_hash() -> str:
    """A realistic-looking SHA-256 hex digest (never a real key)."""
    return "a1b2c3d4e5f6" + "0" * 52  # 64 hex chars total


# ---------------------------------------------------------------------------
# Tests — log_action
# ---------------------------------------------------------------------------


def test_log_action_returns_valid_event():
    # SCENARIO: log a single action and inspect the returned event dict
    # WHY: every downstream consumer (query, export, assertion) depends on
    #      the event having the correct shape
    # EXPECTED: all five audit dimensions are present and correct
    logger = AuditLogger()
    event = logger.log_action(
        action="create_run",
        user_key_hash=_sample_key_hash(),
        resource="/api/runs",
        result="success",
        status_code=200,
        details={"run_id": 42},
    )
    assert event["action"] == "create_run"
    assert event["user_key_hash"] == _sample_key_hash()
    assert event["resource"] == "/api/runs"
    assert event["result"] == "success"
    assert event["status_code"] == 200
    assert event["details"] == {"run_id": 42}
    assert "id" in event
    assert "timestamp" in event


def test_log_action_generates_unique_ids():
    # SCENARIO: log two actions and compare their IDs
    # WHY: every event must be uniquely identifiable for forensic correlation
    # EXPECTED: the two UUIDs differ
    logger = AuditLogger()
    e1 = logger.log_action("a", _sample_key_hash(), "/x", "success")
    e2 = logger.log_action("b", _sample_key_hash(), "/y", "success")
    assert e1["id"] != e2["id"]


def test_timestamp_is_iso8601_utc():
    # SCENARIO: log an action and validate the timestamp format
    # WHY: compliance standards require unambiguous, timezone-aware timestamps;
    #      ISO 8601 with UTC offset is the industry standard
    # EXPECTED: timestamp matches the ISO 8601 pattern with timezone offset
    logger = AuditLogger()
    event = logger.log_action("test", _sample_key_hash(), "/api/test", "success")
    ts = event["timestamp"]
    assert _ISO8601_RE.match(ts), f"Timestamp not ISO 8601 with tz: {ts}"
    # Verify it is parseable and timezone-aware
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"


def test_api_key_hash_never_stored_raw():
    # SCENARIO: log an action and verify the stored key is a hash, not a raw key
    # WHY: if audit logs are compromised, raw API keys would give attackers
    #      direct access; hashes are one-way and safe to store
    # EXPECTED: the stored value looks like a SHA-256 hex digest (64 hex chars)
    #           and does NOT start with the mltk_ prefix
    logger = AuditLogger()
    fake_hash = _sample_key_hash()
    event = logger.log_action("test", fake_hash, "/api/test", "success")
    stored = event["user_key_hash"]
    assert not stored.startswith("mltk_"), "Raw API key leaked into audit log!"
    assert len(stored) == 64, "Expected a 64-char hex hash"
    assert all(c in "0123456789abcdef" for c in stored), "Not a valid hex string"


# ---------------------------------------------------------------------------
# Tests — get_log (filtering)
# ---------------------------------------------------------------------------


def test_get_log_filters_by_action():
    # SCENARIO: log multiple actions, then query for a specific one
    # WHY: incident responders need to isolate specific event types quickly
    # EXPECTED: only events matching the requested action are returned
    logger = AuditLogger()
    logger.log_action("create_run", _sample_key_hash(), "/api/runs", "success")
    logger.log_action("list_results", _sample_key_hash(), "/api/results", "success")
    logger.log_action("create_run", _sample_key_hash(), "/api/runs", "failure", 500)

    results = logger.get_log(action="create_run")
    assert len(results) == 2
    assert all(e["action"] == "create_run" for e in results)


def test_get_log_filters_by_user():
    # SCENARIO: log actions from two different API key hashes, then filter
    # WHY: security reviews need to audit what a specific key did
    # EXPECTED: only events for the requested user hash are returned
    logger = AuditLogger()
    hash_a = "a" * 64
    hash_b = "b" * 64
    logger.log_action("create_run", hash_a, "/api/runs", "success")
    logger.log_action("create_run", hash_b, "/api/runs", "success")
    logger.log_action("list_results", hash_a, "/api/results", "success")

    results = logger.get_log(user=hash_a)
    assert len(results) == 2
    assert all(e["user_key_hash"] == hash_a for e in results)


def test_get_log_filters_by_since():
    # SCENARIO: log events, then query with a 'since' cutoff
    # WHY: auditors often need "show me everything after incident time X"
    # EXPECTED: only events at or after the cutoff are returned
    logger = AuditLogger()
    # Log one event, capture its timestamp, then log another
    e1 = logger.log_action("early", _sample_key_hash(), "/a", "success")
    # Use a cutoff that is the exact timestamp of e1 -- it should be included
    cutoff = e1["timestamp"]
    logger.log_action("later", _sample_key_hash(), "/b", "success")

    results = logger.get_log(since=cutoff)
    actions = {e["action"] for e in results}
    # Both should be included since e1's timestamp equals the cutoff
    assert "early" in actions
    assert "later" in actions


# ---------------------------------------------------------------------------
# Tests — export_csv
# ---------------------------------------------------------------------------


def test_export_csv_creates_valid_file(tmp_path):
    # SCENARIO: log events and export them as CSV
    # WHY: compliance auditors require spreadsheet-compatible exports;
    #      the CSV must have correct headers and be parseable
    # EXPECTED: CSV has a header row + one data row per event, and all
    #           required columns are present
    logger = AuditLogger()
    logger.log_action("create_run", _sample_key_hash(), "/api/runs", "success")
    logger.log_action("list_results", _sample_key_hash(), "/api/results", "success")

    csv_path = str(tmp_path / "audit_export.csv")
    returned_path = logger.export_csv(csv_path)
    assert returned_path == csv_path

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 2
    expected_cols = {
        "id", "timestamp", "action", "user_key_hash",
        "resource", "result", "status_code", "details",
    }
    assert expected_cols == set(reader.fieldnames)  # type: ignore[arg-type]

    # Verify the details column is valid JSON
    for row in rows:
        parsed = json.loads(row["details"])
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Tests — assert_audit_log_complete
# ---------------------------------------------------------------------------


def test_assert_audit_log_complete_passes_when_all_present():
    # SCENARIO: audit log contains all expected actions
    # WHY: the assertion must confirm compliance when the trail is complete
    # EXPECTED: TestResult.passed is True
    entries = [
        {"action": "create_run"},
        {"action": "list_results"},
        {"action": "export_report"},
    ]
    result = assert_audit_log_complete(entries, ["create_run", "list_results"])
    assert result.passed is True
    assert result.name == "audit.log_complete"
    assert result.duration_ms >= 0  # timed_assertion populates this


def test_assert_audit_log_complete_fails_on_missing_action():
    # SCENARIO: audit log is missing a required action
    # WHY: the assertion must flag gaps in the audit trail -- a missing
    #      action means either the operation was not performed or the
    #      logging is broken; both are compliance failures
    # EXPECTED: TestResult.passed is False, missing actions listed
    entries = [
        {"action": "create_run"},
    ]
    result = assert_audit_log_complete(
        entries, ["create_run", "delete_webhook"]
    )
    assert result.passed is False
    assert "delete_webhook" in result.message
    assert "delete_webhook" in result.details["missing_actions"]


# ---------------------------------------------------------------------------
# Tests — JSON Lines persistence
# ---------------------------------------------------------------------------


def test_jsonlines_persistence(tmp_path):
    # SCENARIO: create an AuditLogger with a file path and log events
    # WHY: durable persistence is required so audit trails survive restarts;
    #      JSON Lines format is append-safe and survives partial writes
    # EXPECTED: the file contains one valid JSON object per line
    log_file = str(tmp_path / "audit.jsonl")
    logger = AuditLogger(storage_path=log_file)
    logger.log_action("create_run", _sample_key_hash(), "/api/runs", "success")
    logger.log_action("list_results", _sample_key_hash(), "/api/results", "success")

    with open(log_file, encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "action" in obj
        assert "timestamp" in obj
        assert "user_key_hash" in obj


# -------------------------------------------------------------------
# Parametrized & edge-case tests (hardening)
# -------------------------------------------------------------------


def test_100_actions_ordering():
    # SCENARIO: log 100 actions, verify newest-first ordering
    # WHY: get_log must return in reverse chronological order
    # EXPECTED: first returned event is the last one logged
    logger = AuditLogger()
    for i in range(100):
        logger.log_action(
            f"action_{i}",
            _sample_key_hash(),
            f"/api/{i}",
            "success",
        )
    results = logger.get_log(limit=100)
    assert len(results) == 100
    assert results[0]["action"] == "action_99"
    assert results[-1]["action"] == "action_0"


def test_filter_by_action_and_user():
    # SCENARIO: filter by both action AND user simultaneously
    # WHY: AND filtering is needed for forensic investigation
    # EXPECTED: only events matching BOTH criteria
    logger = AuditLogger()
    h1 = "a" * 64
    h2 = "b" * 64
    logger.log_action("create", h1, "/a", "success")
    logger.log_action("create", h2, "/a", "success")
    logger.log_action("delete", h1, "/a", "success")
    results = logger.get_log(action="create", user=h1)
    assert len(results) == 1
    assert results[0]["action"] == "create"
    assert results[0]["user_key_hash"] == h1


def test_export_csv_special_characters(tmp_path):
    # SCENARIO: details contain commas, quotes, newlines
    # WHY: CSV export must handle special chars safely
    # EXPECTED: CSV is parseable and data is preserved
    logger = AuditLogger()
    logger.log_action(
        "test",
        _sample_key_hash(),
        "/api",
        "success",
        details={
            "msg": 'He said "hello, world"\nnewline'
        },
    )
    csv_path = str(tmp_path / "special.csv")
    logger.export_csv(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    parsed = json.loads(rows[0]["details"])
    assert '"hello, world"' in parsed["msg"]
    assert "\n" in parsed["msg"]


def test_audit_log_complete_partial_coverage():
    # SCENARIO: 2 of 3 expected actions present
    # WHY: partial coverage must correctly report gaps
    # EXPECTED: passed=False, missing_actions has 1 item
    entries = [
        {"action": "create_run"},
        {"action": "list_results"},
    ]
    result = assert_audit_log_complete(
        entries,
        ["create_run", "list_results", "export_report"],
    )
    assert result.passed is False
    missing = result.details["missing_actions"]
    assert missing == ["export_report"]


def test_uuid_uniqueness_50_entries():
    # SCENARIO: log 50 actions, verify all IDs unique
    # WHY: UUID collisions would break forensic correlation
    # EXPECTED: 50 distinct UUIDs
    logger = AuditLogger()
    ids = set()
    for i in range(50):
        e = logger.log_action(
            f"act_{i}",
            _sample_key_hash(),
            f"/api/{i}",
            "success",
        )
        ids.add(e["id"])
    assert len(ids) == 50
