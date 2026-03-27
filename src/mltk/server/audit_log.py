"""Audit logging for the mltk server.

Every API action is recorded for compliance and security auditing.  The
audit log captures five dimensions for each event:

* **WHO** -- API key hash (never the raw key)
* **WHAT** -- action identifier (``create_run``, ``delete_webhook``, ...)
* **WHEN** -- ISO 8601 UTC timestamp
* **WHERE** -- endpoint path and HTTP method
* **RESULT** -- success/failure and HTTP status code

Why audit logging matters
-------------------------
Compliance frameworks such as SOC 2 Type II, HIPAA, and SOX all require
an immutable, queryable record of system access and data mutations.  Even
outside regulated industries, an audit trail is invaluable for:

* **Incident response** -- answering "what happened and who was affected?"
* **Access review** -- proving that only authorized keys wrote data
* **Debugging** -- correlating a bad test run with the API call that created it

Design decisions
----------------
* Events are stored in-memory *and* optionally appended to a
  `JSON Lines <https://jsonlines.org/>`_ file for durable persistence.
  JSON Lines (one JSON object per line) is chosen over full JSON arrays
  because it is append-friendly and survives partial writes.
* Timestamps are always UTC ISO 8601 with timezone offset (``+00:00``)
  so logs from different machines are directly comparable.
* The ``export_csv`` method exists because compliance auditors almost
  universally ask for spreadsheet-compatible exports.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from mltk.core.assertion import timed_assertion
from mltk.core.result import Severity, TestResult

logger = logging.getLogger(__name__)


class AuditLogger:
    """Log every API action for compliance and security auditing.

    WHY audit logging:
      SOC 2 Type II compliance requires a complete audit trail of who did
      what, when, on which resource.  Healthcare (HIPAA) and finance (SOX)
      have similar requirements.  The audit log captures:

      - **WHO**: API key hash (never the raw key)
      - **WHAT**: action (``create_run``, ``delete_webhook``, ``list_results``)
      - **WHEN**: ISO 8601 timestamp in UTC
      - **WHERE**: endpoint path + HTTP method
      - **RESULT**: ``success`` / ``failure`` + status code

    Parameters
    ----------
    storage_path:
        Optional path to a JSON Lines file.  If provided, every event is
        appended to this file in addition to being held in memory.

    Examples
    --------
    >>> logger = AuditLogger()
    >>> event = logger.log_action(
    ...     action="create_run",
    ...     user_key_hash="a1b2c3...",
    ...     resource="/api/runs",
    ...     result="success",
    ... )
    >>> event["action"]
    'create_run'
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize with optional file path for JSON Lines output.

        If *storage_path* is given, the file is opened in append mode so
        existing entries are preserved across restarts.
        """
        self._events: list[dict[str, Any]] = []
        self._storage_path = storage_path

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def log_action(
        self,
        action: str,
        user_key_hash: str,
        resource: str,
        result: str,
        status_code: int = 200,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an audit event and return the event dict.

        Every call creates a new event with a unique ID, UTC timestamp,
        and all five audit dimensions.  The event is appended to the
        in-memory list and, if a storage path was configured, to the
        JSON Lines file.

        Parameters
        ----------
        action:
            A short, machine-readable action name such as ``create_run``
            or ``delete_webhook``.
        user_key_hash:
            SHA-256 hex digest of the API key that performed the action.
            **Never** pass the raw key here.
        resource:
            The API endpoint path (e.g. ``/api/runs``).
        result:
            ``"success"`` or ``"failure"``.
        status_code:
            HTTP status code of the response (default 200).
        details:
            Optional dict of additional context (e.g. run ID, error message).

        Returns
        -------
        dict
            The full event dict including generated ``id`` and ``timestamp``.
        """
        event: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action": action,
            "user_key_hash": user_key_hash,
            "resource": resource,
            "result": result,
            "status_code": status_code,
            "details": details or {},
        }
        self._events.append(event)
        self._persist_event(event)
        logger.debug("Audit: %s %s -> %s (%d)", action, resource, result, status_code)
        return event

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_log(
        self,
        action: str | None = None,
        user: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query the audit log with optional filters.

        Filters are combined with AND semantics -- an event must match
        *all* supplied filters to be included.  Results are returned in
        reverse chronological order (newest first) up to *limit*.

        Parameters
        ----------
        action:
            If given, only return events whose ``action`` matches exactly.
        user:
            If given, only return events whose ``user_key_hash`` matches.
        since:
            If given, an ISO 8601 timestamp string.  Only events **at or
            after** this time are returned.
        limit:
            Maximum number of events to return (default 100).

        Returns
        -------
        list[dict]
            Matching events, newest first.

        Examples
        --------
        >>> logger = AuditLogger()
        >>> logger.log_action("create_run", "abc", "/api/runs", "success")
        {...}
        >>> logger.get_log(action="create_run")
        [{...}]
        """
        entries = list(reversed(self._events))  # newest first
        if action is not None:
            entries = [e for e in entries if e["action"] == action]
        if user is not None:
            entries = [e for e in entries if e["user_key_hash"] == user]
        if since is not None:
            since_dt = datetime.fromisoformat(since)
            # Ensure the cutoff is timezone-aware for correct comparison.
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
            entries = [
                e for e in entries
                if datetime.fromisoformat(e["timestamp"]) >= since_dt
            ]
        return entries[:limit]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str) -> str:
        """Export the full audit log as CSV for compliance reporting.

        The CSV contains one row per event with columns matching the
        event dict keys.  This format is universally accepted by
        auditors, SIEM tools, and spreadsheet software.

        Parameters
        ----------
        output_path:
            File path for the CSV output.

        Returns
        -------
        str
            The same *output_path*, for convenience in chaining.

        Examples
        --------
        >>> logger = AuditLogger()
        >>> logger.log_action("create_run", "abc", "/api/runs", "success")
        {...}
        >>> logger.export_csv("/tmp/audit.csv")
        '/tmp/audit.csv'
        """
        fieldnames = [
            "id", "timestamp", "action", "user_key_hash",
            "resource", "result", "status_code", "details",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for event in self._events:
                row = dict(event)
                # Serialize the details dict as a JSON string for CSV.
                row["details"] = json.dumps(row.get("details", {}))
                writer.writerow(row)
        return output_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _persist_event(self, event: dict[str, Any]) -> None:
        """Append a single event to the JSON Lines file (if configured).

        JSON Lines format writes one complete JSON object per line,
        making the file append-safe: a crash mid-write loses at most the
        current line, never corrupting earlier entries.
        """
        if self._storage_path is None:
            return
        try:
            with open(self._storage_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError:
            logger.warning("Failed to persist audit event to %s", self._storage_path)


# ---------------------------------------------------------------------------
# Audit assertion
# ---------------------------------------------------------------------------


@timed_assertion
def assert_audit_log_complete(
    audit_entries: list[dict[str, Any]],
    expected_actions: list[str],
) -> TestResult:
    """Assert that required actions appear in the audit log.

    WHY: Compliance frameworks require proof that certain actions were
    performed and logged.  This assertion verifies the audit trail
    contains all expected entries.  For example, after a deployment
    you might assert that ``create_run``, ``list_results``, and
    ``export_report`` all appear in the log.

    Parameters
    ----------
    audit_entries:
        List of audit event dicts (as returned by ``AuditLogger.get_log``).
    expected_actions:
        List of action names that **must** appear at least once.

    Returns
    -------
    TestResult
        Passing if every expected action is found, failing otherwise.

    Examples
    --------
    >>> entries = [{"action": "create_run"}, {"action": "list_results"}]
    >>> result = assert_audit_log_complete(entries, ["create_run"])
    >>> result.passed
    True
    >>> result = assert_audit_log_complete(entries, ["delete_webhook"])
    >>> result.passed
    False
    """
    recorded_actions = {e.get("action") for e in audit_entries}
    missing = [a for a in expected_actions if a not in recorded_actions]

    if missing:
        return TestResult(
            name="audit.log_complete",
            passed=False,
            severity=Severity.CRITICAL,
            message=f"Audit log missing required actions: {missing}",
            details={
                "missing_actions": missing,
                "expected_actions": expected_actions,
                "recorded_actions": sorted(recorded_actions),
            },
        )
    return TestResult(
        name="audit.log_complete",
        passed=True,
        severity=Severity.INFO,
        message="All required actions present in audit log",
        details={
            "expected_actions": expected_actions,
            "recorded_actions": sorted(recorded_actions),
        },
    )
