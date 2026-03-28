"""Test scheduler -- run mltk test suites on a recurring schedule.

**Why scheduled tests for ML systems:**

Machine learning systems degrade silently.  A model that passes every test
today might fail tomorrow because:

- **Data drift:** The distribution of incoming data shifted, and features that
  were predictive last month are now noise.
- **Dependency rot:** A library updated its numeric precision, changing model
  outputs just enough to cross a threshold.
- **Infrastructure decay:** A database connection pool started timing out,
  making data fetches fail intermittently.
- **Stale artifacts:** A feature store cache expired, and the fallback path
  returns different values.

Unlike traditional software where failures are immediate (crash, 500 error),
ML failures are *gradual* -- accuracy drops 0.1% per day until someone notices
weeks later.  Scheduled test runs catch these issues early.

**Architecture:**

The scheduler is intentionally simple -- no external dependencies, no daemon
process, no cron binary.  It stores schedule definitions in memory and uses
``time.time()`` to determine when a schedule is due.  Test commands are
executed via ``subprocess.run()``, capturing stdout, stderr, and return code.

For production use, pair this with:
- A system cron job or systemd timer that calls ``mltk server run-schedules``
- Or run inside the mltk server process as a background loop

**Webhook notifications:**

Each schedule can optionally specify a ``webhook_url``.  When a scheduled run
completes, the result is POSTed as JSON to that URL.  This integrates with
Slack, PagerDuty, OpsGenie, or any webhook-compatible service.

No external dependencies -- uses only ``json``, ``subprocess``, ``time``, and
``urllib`` from stdlib.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


class TestScheduler:
    """Schedule periodic mltk test runs with webhook notifications.

    **How it works:**

    1. **Add schedules** with ``add_schedule()`` -- each schedule has a name,
       a shell command, an interval, and an optional webhook URL.
    2. **Check due schedules** with ``should_run()`` -- compares the current
       time against the schedule's last run time plus the interval.
    3. **Execute** with ``execute()`` -- runs the command as a subprocess,
       captures the result, and optionally fires the webhook.

    **Why no threading or async?**

    Keeping the scheduler synchronous makes it easy to test, debug, and embed
    in any execution model (cron, systemd, server loop, CI pipeline).  The
    caller decides *when* to call ``should_run()`` and ``execute()``.

    Example::

        from mltk.server.scheduler import TestScheduler

        scheduler = TestScheduler()
        scheduler.add_schedule(
            name="nightly-drift-check",
            command="mltk run --tag drift",
            interval_seconds=86400,           # once per day
            webhook_url="https://hooks.slack.com/services/T.../B.../...",
        )

        # In a loop or cron job:
        if scheduler.should_run("nightly-drift-check"):
            result = scheduler.execute("nightly-drift-check")
            print(f"Return code: {result['returncode']}")
    """

    def __init__(self) -> None:
        self._schedules: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    def add_schedule(
        self,
        name: str,
        command: str,
        interval_seconds: int = 3600,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """Add a scheduled test run.

        Each schedule is identified by a unique *name*.  If a schedule with
        the same name already exists, a ``ValueError`` is raised -- use
        ``remove_schedule`` first if you need to update an existing one.

        **Design note:** ``interval_seconds`` uses wall-clock seconds rather
        than cron syntax because it is simpler to reason about and test.  For
        common intervals: hourly = 3600, daily = 86400, weekly = 604800.

        Args:
            name: Unique identifier for this schedule (e.g.
                ``"nightly-drift"``).  Used as the key for all other methods.
            command: Shell command to execute (e.g. ``"mltk run --tag drift"``).
                Executed via ``subprocess.run(shlex.split(command), shell=False)``.
            interval_seconds: Minimum seconds between consecutive runs.
                Defaults to 3600 (one hour).
            webhook_url: Optional URL to POST results to after each run.
                Set to ``None`` to disable webhook notifications.

        Returns:
            The schedule definition dictionary containing all fields plus
            ``created_at`` and ``last_run`` timestamps.

        Raises:
            ValueError: If *name* is empty or already exists, or if
                *command* is empty, or if *interval_seconds* is not positive.
        """
        if not name:
            raise ValueError("Schedule name must not be empty")
        if not command:
            raise ValueError("Schedule command must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if any(s["name"] == name for s in self._schedules):
            raise ValueError(f"Schedule {name!r} already exists")

        schedule: dict[str, Any] = {
            "name": name,
            "command": command,
            "interval_seconds": interval_seconds,
            "webhook_url": webhook_url,
            "created_at": time.time(),
            "last_run": 0.0,
        }
        self._schedules.append(schedule)
        return schedule

    def remove_schedule(self, name: str) -> bool:
        """Remove a schedule by name.

        Args:
            name: The schedule identifier to remove.

        Returns:
            ``True`` if a schedule with that name was found and removed,
            ``False`` if no such schedule exists.
        """
        for i, sched in enumerate(self._schedules):
            if sched["name"] == name:
                self._schedules.pop(i)
                return True
        return False

    def list_schedules(self) -> list[dict[str, Any]]:
        """List all configured schedules.

        Returns:
            A list of schedule dictionaries.  Each dictionary contains:
            ``name``, ``command``, ``interval_seconds``, ``webhook_url``,
            ``created_at``, and ``last_run``.  Returns an empty list if no
            schedules are configured.

        Note:
            The returned list is a shallow copy -- mutating it does not
            affect the scheduler's internal state.
        """
        return list(self._schedules)

    def _find_schedule(self, name: str) -> dict[str, Any] | None:
        """Look up a schedule by name.  Returns ``None`` if not found."""
        for sched in self._schedules:
            if sched["name"] == name:
                return sched
        return None

    # ------------------------------------------------------------------
    # Execution logic
    # ------------------------------------------------------------------

    def should_run(self, name: str, current_time: float | None = None) -> bool:
        """Check if a schedule is due to run.

        A schedule is due when the elapsed time since its last run exceeds
        the configured ``interval_seconds``.  Schedules that have never run
        (``last_run == 0.0``) are always due.

        **Why accept ``current_time`` as a parameter?**

        Injecting the current time makes this method fully deterministic in
        tests.  Production code omits it and gets ``time.time()`` by default.

        Args:
            name: The schedule identifier to check.
            current_time: Override for the current timestamp (seconds since
                epoch).  If ``None``, uses ``time.time()``.

        Returns:
            ``True`` if the schedule exists and is due, ``False`` otherwise
            (including when the schedule does not exist).
        """
        sched = self._find_schedule(name)
        if sched is None:
            return False

        now = current_time if current_time is not None else time.time()
        elapsed = now - sched["last_run"]
        return elapsed >= sched["interval_seconds"]

    def execute(self, name: str) -> dict[str, Any]:
        """Execute a scheduled test run via subprocess.

        Runs the schedule's command in a shell subprocess, captures its
        stdout, stderr, and return code, then optionally POSTs the result
        to the configured webhook URL.

        **Subprocess safety:** Commands run with ``shell=False`` (using shlex.split) because mltk
        commands often include flags and pipes.  The command string comes from
        the schedule creator (an admin), not from end-user input.

        Args:
            name: The schedule identifier to execute.

        Returns:
            A result dictionary with keys:

            - ``name`` -- schedule name
            - ``command`` -- the command that was run
            - ``returncode`` -- process exit code (0 = success)
            - ``stdout`` -- captured standard output (decoded UTF-8)
            - ``stderr`` -- captured standard error (decoded UTF-8)
            - ``executed_at`` -- timestamp when the run started
            - ``duration_seconds`` -- wall-clock execution time
            - ``webhook_sent`` -- ``True`` if webhook was sent successfully

        Raises:
            ValueError: If no schedule with the given *name* exists.
        """
        sched = self._find_schedule(name)
        if sched is None:
            raise ValueError(f"Schedule {name!r} not found")

        start = time.time()
        try:
            proc = subprocess.run(
                shlex.split(sched["command"]),
                shell=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            returncode = -1
            stdout = ""
            stderr = "Command timed out after 3600 seconds"

        duration = time.time() - start
        sched["last_run"] = time.time()

        result: dict[str, Any] = {
            "name": name,
            "command": sched["command"],
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "executed_at": start,
            "duration_seconds": duration,
            "webhook_sent": False,
        }

        # Fire webhook notification if configured
        if sched["webhook_url"]:
            result["webhook_sent"] = self._send_webhook(
                sched["webhook_url"], result
            )

        return result

    # ------------------------------------------------------------------
    # Webhook delivery
    # ------------------------------------------------------------------

    @staticmethod
    def _send_webhook(url: str, payload: dict[str, Any]) -> bool:
        """POST the execution result to a webhook URL.

        **Why fire-and-forget?**

        Webhook delivery is best-effort.  If the remote server is down, we
        log the failure but do not retry -- the test result is still recorded
        locally.  Retry logic belongs in a dedicated queue (e.g. Celery),
        not in the scheduler itself.

        Args:
            url: The webhook endpoint URL.
            payload: The execution result dictionary to send as JSON.

        Returns:
            ``True`` if the webhook returned HTTP 2xx, ``False`` otherwise.
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False
