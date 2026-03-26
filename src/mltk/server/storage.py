"""SQLite storage for test results and reports."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


class Storage:
    """SQLite-backed persistence layer for mltk test runs and results."""

    def __init__(self, db_path: str = "mltk_server.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    project     TEXT    NOT NULL DEFAULT 'default',
                    timestamp   TEXT    NOT NULL,
                    total       INTEGER NOT NULL DEFAULT 0,
                    passed      INTEGER NOT NULL DEFAULT 0,
                    failed      INTEGER NOT NULL DEFAULT 0,
                    score       REAL    NOT NULL DEFAULT 0.0,
                    duration_ms REAL    NOT NULL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       INTEGER NOT NULL REFERENCES runs(id),
                    name         TEXT    NOT NULL,
                    passed       INTEGER NOT NULL DEFAULT 0,
                    severity     TEXT    NOT NULL DEFAULT 'info',
                    message      TEXT    NOT NULL DEFAULT '',
                    details_json TEXT    NOT NULL DEFAULT '{}',
                    duration_ms  REAL    NOT NULL DEFAULT 0.0
                )
            """)
            conn.commit()

    def save_run(self, project: str, results: list[dict]) -> int:  # type: ignore[type-arg]
        """Save a test run with all results. Returns run_id.

        Args:
            project: Project name for grouping runs.
            results: List of result dicts with keys: name, passed, severity,
                     message, details, duration_ms.

        Returns:
            The auto-assigned run_id.
        """
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        failed = total - passed
        score = (passed / total * 100.0) if total > 0 else 0.0
        duration_ms = sum(float(r.get("duration_ms", 0.0)) for r in results)
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (project, timestamp, total, passed, failed, score, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (project, timestamp, total, passed, failed, score, duration_ms),
            )
            run_id = cursor.lastrowid

            for r in results:
                details_json = json.dumps(r.get("details", {}), default=str)
                conn.execute(
                    """
                    INSERT INTO results
                        (run_id, name, passed, severity, message, details_json, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(r.get("name", "")),
                        1 if r.get("passed", False) else 0,
                        str(r.get("severity", "info")),
                        str(r.get("message", "")),
                        details_json,
                        float(r.get("duration_ms", 0.0)),
                    ),
                )
            conn.commit()

        return run_id  # type: ignore[return-value]

    def get_runs(
        self,
        project: str | None = None,
        limit: int = 50,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Get recent test runs, ordered by most-recent first.

        Args:
            project: If provided, filter to this project only.
            limit: Maximum number of runs to return.

        Returns:
            List of run summary dicts.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if project is not None:
                rows = conn.execute(
                    """
                    SELECT id, project, timestamp, total, passed, failed, score, duration_ms
                    FROM runs
                    WHERE project = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (project, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, project, timestamp, total, passed, failed, score, duration_ms
                    FROM runs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict | None:  # type: ignore[type-arg]
        """Get a single run with all its results.

        Args:
            run_id: The run to retrieve.

        Returns:
            Dict with run summary and 'results' list, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            run_row = conn.execute(
                "SELECT id, project, timestamp, total, passed, failed, score, duration_ms "
                "FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()

            if run_row is None:
                return None

            run = dict(run_row)

            result_rows = conn.execute(
                "SELECT id, name, passed, severity, message, details_json, duration_ms "
                "FROM results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()

        run_results = []
        for row in result_rows:
            r = dict(row)
            try:
                r["details"] = json.loads(r.pop("details_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                r["details"] = {}
            r["passed"] = bool(r["passed"])
            run_results.append(r)

        run["results"] = run_results
        return run

    def get_trends(
        self,
        project: str,
        limit: int = 20,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Get score trend over time for a project.

        Args:
            project: Project name to query.
            limit: Maximum number of data points (most recent first).

        Returns:
            List of dicts with keys: id, timestamp, score, passed, failed, total.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, timestamp, score, passed, failed, total
                FROM runs
                WHERE project = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (project, limit),
            ).fetchall()

        # Return in chronological order so charts render left-to-right
        return list(reversed([dict(row) for row in rows]))
