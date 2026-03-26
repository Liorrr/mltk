"""SQLite storage for test results and reports."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from mltk.server.webhooks import WebhookConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Schema migrations — each entry is (version, description, SQL list)
# ------------------------------------------------------------------
_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "Initial schema: runs, results, api_keys, webhooks tables with indexes",
        [
            # Core tables
            """
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
            """,
            """
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
            """,
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash   TEXT    NOT NULL UNIQUE,
                project    TEXT    NOT NULL DEFAULT 'default',
                created_at TEXT    NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS webhooks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT    NOT NULL,
                events_json TEXT    NOT NULL DEFAULT '[]',
                project     TEXT,
                created_at  TEXT    NOT NULL
            )
            """,
            # Indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)",
        ],
    ),
]


class Storage:
    """SQLite-backed persistence layer for mltk test runs and results.

    Uses a singleton connection (``check_same_thread=False``) so the same
    ``Storage`` instance can be shared across FastAPI async workers.
    WAL journal mode is enabled for concurrent-read performance, and
    FOREIGN KEY constraints are enforced at the connection level.

    Schema changes are managed by a versioned migration system.  Each
    migration is recorded in a ``schema_versions`` table so it is only
    applied once.  Future upgrades simply add a new entry to the
    module-level ``_MIGRATIONS`` list.
    """

    def __init__(self, db_path: str = "mltk_server.db") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False,
        )
        self._init_db()

    # ------------------------------------------------------------------
    # Schema migration engine
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_schema_versions_table(conn: sqlite3.Connection) -> None:
        """Create the ``schema_versions`` tracking table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version    INTEGER PRIMARY KEY,
                applied_at TEXT    NOT NULL
            )
        """)

    @staticmethod
    def _get_current_version(conn: sqlite3.Connection) -> int:
        """Return the highest applied migration version, or 0 if none."""
        row = conn.execute(
            "SELECT MAX(version) FROM schema_versions"
        ).fetchone()
        return row[0] if row[0] is not None else 0

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Run any pending migrations in order and record them.

        Each migration is a list of SQL statements executed inside a
        transaction.  The version number is recorded in
        ``schema_versions`` so the migration is never re-applied.
        """
        self._ensure_schema_versions_table(conn)
        current = self._get_current_version(conn)

        for version, description, statements in _MIGRATIONS:
            if version <= current:
                continue
            logger.info("Applying migration v%d: %s", version, description)
            for sql in statements:
                conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_versions (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(tz=timezone.utc).isoformat()),
            )
            logger.info("Migration v%d applied successfully", version)

        conn.commit()

    def _init_db(self) -> None:
        """Initialise the database — configure pragmas then run migrations."""
        conn = self._conn

        # --- Pragmas (connection-level, not versioned) -------------------
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = ON")

        # --- Versioned migrations ----------------------------------------
        self._migrate(conn)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def save_run(self, project: str, results: list[dict]) -> int:  # type: ignore[type-arg]
        """Save a test run with all results. Returns run_id.

        Uses ``executemany`` for batch-inserting individual results, which is
        significantly faster than issuing one ``execute`` per row.

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

        conn = self._conn
        cursor = conn.execute(
            """
            INSERT INTO runs (project, timestamp, total, passed, failed, score, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project, timestamp, total, passed, failed, score, duration_ms),
        )
        run_id = cursor.lastrowid

        if results:
            result_rows = [
                (
                    run_id,
                    str(r.get("name", "")),
                    1 if r.get("passed", False) else 0,
                    str(r.get("severity", "info")),
                    str(r.get("message", "")),
                    json.dumps(r.get("details", {}), default=str),
                    float(r.get("duration_ms", 0.0)),
                )
                for r in results
            ]
            conn.executemany(
                """
                INSERT INTO results
                    (run_id, name, passed, severity, message, details_json, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                result_rows,
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
        conn = self._conn
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
        conn.row_factory = None
        return [dict(row) for row in rows]

    def get_run(self, run_id: int) -> dict | None:  # type: ignore[type-arg]
        """Get a single run with all its results.

        Args:
            run_id: The run to retrieve.

        Returns:
            Dict with run summary and 'results' list, or None if not found.
        """
        conn = self._conn
        conn.row_factory = sqlite3.Row

        run_row = conn.execute(
            "SELECT id, project, timestamp, total, passed, failed, score, duration_ms "
            "FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        if run_row is None:
            conn.row_factory = None
            return None

        run = dict(run_row)

        result_rows = conn.execute(
            "SELECT id, name, passed, severity, message, details_json, duration_ms "
            "FROM results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        conn.row_factory = None

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
        conn = self._conn
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
        conn.row_factory = None

        # Return in chronological order so charts render left-to-right
        return list(reversed([dict(row) for row in rows]))

    # ------------------------------------------------------------------
    # API key management
    # ------------------------------------------------------------------

    def save_api_key(self, key_hash: str, project: str) -> None:
        """Persist a hashed API key bound to a project.

        Args:
            key_hash: SHA-256 hex digest of the raw key.
            project: Project name associated with this key.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        conn = self._conn
        conn.execute(
            "INSERT INTO api_keys (key_hash, project, created_at) VALUES (?, ?, ?)",
            (key_hash, project, timestamp),
        )
        conn.commit()

    def verify_api_key(self, key_hash: str) -> str | None:
        """Look up a key hash and return the associated project name.

        Args:
            key_hash: SHA-256 hex digest to look up.

        Returns:
            Project name if found, else None.
        """
        conn = self._conn
        row = conn.execute(
            "SELECT project FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Webhook management
    # ------------------------------------------------------------------

    def save_webhook(self, url: str, events: list[str], project: str | None = None) -> int:
        """Register a new webhook configuration.

        Args:
            url: Target URL to POST to.
            events: List of event names (e.g. ["on_failure", "on_success"]).
            project: If provided, restrict webhook to this project only.

        Returns:
            Auto-assigned webhook id.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        events_json = json.dumps(events)
        conn = self._conn
        cursor = conn.execute(
            "INSERT INTO webhooks (url, events_json, project, created_at) VALUES (?, ?, ?, ?)",
            (url, events_json, project, timestamp),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_webhooks(self, project: str | None = None) -> list[WebhookConfig]:
        """Return all registered webhooks, optionally filtered by project.

        A webhook with project=NULL matches every project.

        Args:
            project: If given, return webhooks for this project plus global ones.

        Returns:
            List of WebhookConfig objects.
        """
        conn = self._conn
        conn.row_factory = sqlite3.Row
        if project is not None:
            rows = conn.execute(
                "SELECT id, url, events_json, project FROM webhooks "
                "WHERE project = ? OR project IS NULL",
                (project,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, url, events_json, project FROM webhooks"
            ).fetchall()
        conn.row_factory = None

        configs: list[WebhookConfig] = []
        for row in rows:
            try:
                events = json.loads(row["events_json"])
            except (json.JSONDecodeError, TypeError):
                events = []
            configs.append(
                WebhookConfig(
                    id=row["id"],
                    url=row["url"],
                    events=events,
                    project=row["project"],
                )
            )
        return configs

    def delete_webhook(self, webhook_id: int) -> bool:
        """Remove a webhook by id.

        Args:
            webhook_id: Primary key of the webhook to remove.

        Returns:
            True if a row was deleted, False if the id did not exist.
        """
        conn = self._conn
        cursor = conn.execute(
            "DELETE FROM webhooks WHERE id = ?",
            (webhook_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
