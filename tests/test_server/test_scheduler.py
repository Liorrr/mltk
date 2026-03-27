"""Tests for mltk.server.scheduler -- periodic test run scheduling.

The scheduler manages named test schedules with interval-based timing and
optional webhook notifications.  All subprocess calls are mocked -- no
actual commands are executed during testing.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters for ML teams>
    # EXPECTED: <what the test asserts>

Test coverage:
    1. add_schedule creates a properly structured schedule dict
    2. remove_schedule returns True when found, False when missing
    3. list_schedules returns all configured schedules
    4. should_run returns True when interval has elapsed
    5. should_run returns False when not yet due
    6. execute runs the subprocess and captures output
    7. Edge case: duplicate schedule names are rejected
    8. Edge case: empty scheduler operations are safe
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.server.scheduler import TestScheduler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler() -> TestScheduler:
    """Return a fresh TestScheduler instance for each test."""
    return TestScheduler()


@pytest.fixture
def populated_scheduler(scheduler: TestScheduler) -> TestScheduler:
    """Return a scheduler with one pre-configured schedule for convenience."""
    scheduler.add_schedule(
        name="drift-check",
        command="mltk run --tag drift",
        interval_seconds=3600,
        webhook_url="http://hooks.example.com/test",
    )
    return scheduler


# ---------------------------------------------------------------------------
# Tests: schedule management
# ---------------------------------------------------------------------------


class TestAddSchedule:
    # SCENARIO: add a new schedule with all parameters
    # WHY: the schedule dict must contain all fields needed by should_run()
    #      and execute() -- missing keys would cause runtime KeyErrors
    # EXPECTED: returns a dict with name, command, interval, webhook, timestamps
    def test_creates_schedule_dict(self, scheduler):
        result = scheduler.add_schedule(
            name="nightly",
            command="mltk run --tag nightly",
            interval_seconds=86400,
            webhook_url="http://hooks.example.com/test",
        )

        assert isinstance(result, dict), "must return a dict"
        assert result["name"] == "nightly"
        assert result["command"] == "mltk run --tag nightly"
        assert result["interval_seconds"] == 86400
        assert result["webhook_url"] == "http://hooks.example.com/test"
        assert "created_at" in result, "must record creation timestamp"
        assert result["last_run"] == 0.0, "new schedule must not have a last_run"

    # SCENARIO: attempt to add two schedules with the same name
    # WHY: duplicate names would cause ambiguity in should_run() and execute().
    #      The scheduler must reject duplicates explicitly rather than silently
    #      overwriting (which could lose webhook config).
    # EXPECTED: ValueError raised on the second add_schedule call
    def test_rejects_duplicate_names(self, populated_scheduler):
        with pytest.raises(ValueError, match="already exists"):
            populated_scheduler.add_schedule(
                name="drift-check",
                command="mltk run --tag other",
            )


class TestRemoveSchedule:
    # SCENARIO: remove an existing schedule by name
    # WHY: teams need to clean up schedules when tests are retired or renamed.
    #      The return value tells the caller whether anything was actually removed.
    # EXPECTED: returns True and the schedule is no longer in list_schedules
    def test_returns_true_when_found(self, populated_scheduler):
        assert populated_scheduler.remove_schedule("drift-check") is True
        assert len(populated_scheduler.list_schedules()) == 0

    # SCENARIO: remove a schedule that does not exist
    # WHY: calling remove on a nonexistent name should be safe (idempotent),
    #      but the caller needs to know it was a no-op (e.g. for logging).
    # EXPECTED: returns False, no exception raised
    def test_returns_false_when_missing(self, scheduler):
        assert scheduler.remove_schedule("nonexistent") is False


class TestListSchedules:
    # SCENARIO: list all schedules after adding multiple
    # WHY: the server API needs to enumerate all schedules for the admin UI.
    #      The list must reflect every add and not include removed ones.
    # EXPECTED: returns a list with the correct count and schedule names
    def test_returns_all_schedules(self, scheduler):
        scheduler.add_schedule(name="hourly", command="mltk run", interval_seconds=3600)
        scheduler.add_schedule(name="daily", command="mltk run --all", interval_seconds=86400)

        schedules = scheduler.list_schedules()
        assert len(schedules) == 2
        names = {s["name"] for s in schedules}
        assert names == {"hourly", "daily"}

    # SCENARIO: list schedules when none are configured
    # WHY: empty state must not crash -- the admin UI should show "no schedules"
    # EXPECTED: returns an empty list
    def test_empty_scheduler_returns_empty_list(self, scheduler):
        assert scheduler.list_schedules() == []


# ---------------------------------------------------------------------------
# Tests: execution logic
# ---------------------------------------------------------------------------


class TestShouldRun:
    # SCENARIO: schedule interval has elapsed since last run
    # WHY: this is the core scheduling logic.  If should_run() returns False
    #      when the interval has elapsed, scheduled tests silently stop running.
    # EXPECTED: returns True
    def test_true_when_interval_elapsed(self, populated_scheduler):
        # Set last_run to 2 hours ago, interval is 1 hour
        sched = populated_scheduler._find_schedule("drift-check")
        sched["last_run"] = 1000.0

        # current_time is 1000 + 3600 + 1 = 4601 (1 second past due)
        assert populated_scheduler.should_run("drift-check", current_time=4601.0) is True

    # SCENARIO: schedule is not yet due (interval has not elapsed)
    # WHY: premature execution wastes CI resources and floods notification
    #      channels.  The scheduler must respect the configured interval.
    # EXPECTED: returns False
    def test_false_when_not_yet_due(self, populated_scheduler):
        sched = populated_scheduler._find_schedule("drift-check")
        sched["last_run"] = 1000.0

        # Only 1800 seconds elapsed, but interval is 3600
        assert populated_scheduler.should_run("drift-check", current_time=2800.0) is False

    # SCENARIO: check should_run for a schedule that does not exist
    # WHY: a deleted schedule should not cause errors in the run loop
    # EXPECTED: returns False (safe no-op)
    def test_false_when_schedule_missing(self, scheduler):
        assert scheduler.should_run("nonexistent", current_time=99999.0) is False


class TestExecute:
    # SCENARIO: execute a schedule and capture subprocess output
    # WHY: the result dict feeds into webhook notifications, logging, and the
    #      admin UI.  All fields must be present even on success.
    # EXPECTED: result dict has all expected keys, subprocess was called
    @patch("mltk.server.scheduler.subprocess.run")
    def test_runs_subprocess_and_captures_output(self, mock_run, populated_scheduler):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "All 42 tests passed.\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = populated_scheduler.execute("drift-check")

        # Verify subprocess was called with the schedule's command
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["mltk", "run", "--tag", "drift"]
        assert call_args[1]["shell"] is False
        assert call_args[1]["capture_output"] is True

        # Verify result structure
        assert result["name"] == "drift-check"
        assert result["command"] == "mltk run --tag drift"
        assert result["returncode"] == 0
        assert result["stdout"] == "All 42 tests passed.\n"
        assert result["stderr"] == ""
        assert "executed_at" in result
        assert "duration_seconds" in result

    # SCENARIO: execute a schedule that does not exist
    # WHY: calling execute with a wrong name must fail explicitly so the
    #      error is caught early rather than silently skipping.
    # EXPECTED: ValueError raised
    def test_raises_on_missing_schedule(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            scheduler.execute("nonexistent")
