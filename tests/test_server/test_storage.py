"""Tests for mltk.server.storage — SQLite persistence layer."""
from __future__ import annotations

import sqlite3

import pytest

from mltk.server.storage import _MIGRATIONS, Storage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    {
        "name": "test_a", "passed": True, "severity": "info",
        "message": "ok", "details": {}, "duration_ms": 10.0,
    },
    {
        "name": "test_b", "passed": False, "severity": "error",
        "message": "fail", "details": {"score": 0.4}, "duration_ms": 20.0,
    },
    {
        "name": "test_c", "passed": True, "severity": "info",
        "message": "ok", "details": {}, "duration_ms": 5.0,
    },
]


@pytest.fixture
def storage(tmp_path):
    """In-memory-like storage backed by a temp file."""
    # SCENARIO: fresh storage for every test
    # WHY: avoid cross-test state bleed
    # EXPECTED: each test gets a clean database
    db_file = str(tmp_path / "test.db")
    return Storage(db_path=db_file)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_get_run(storage):
    # SCENARIO: save a run and retrieve it by id
    # WHY: core round-trip must be lossless
    # EXPECTED: run dict matches what was saved, results count matches
    run_id = storage.save_run("myproject", SAMPLE_RESULTS)
    assert isinstance(run_id, int)
    assert run_id > 0

    run = storage.get_run(run_id)
    assert run is not None
    assert run["project"] == "myproject"
    assert run["total"] == 3
    assert run["passed"] == 2
    assert run["failed"] == 1
    assert len(run["results"]) == 3

    names = {r["name"] for r in run["results"]}
    assert names == {"test_a", "test_b", "test_c"}

    # details JSON round-trips correctly
    test_b = next(r for r in run["results"] if r["name"] == "test_b")
    assert test_b["details"] == {"score": 0.4}
    assert test_b["passed"] is False


def test_list_runs(storage):
    # SCENARIO: save multiple runs then list them
    # WHY: list_runs must return all saved runs in reverse-chronological order
    # EXPECTED: most-recently saved run appears first
    id1 = storage.save_run("proj", SAMPLE_RESULTS[:1])
    id2 = storage.save_run("proj", SAMPLE_RESULTS[:2])
    id3 = storage.save_run("proj", SAMPLE_RESULTS)

    runs = storage.get_runs()
    assert len(runs) >= 3
    ids = [r["id"] for r in runs]
    # most recent first
    assert ids.index(id3) < ids.index(id2) < ids.index(id1)


def test_trends(storage):
    # SCENARIO: save three runs for a project then fetch trends
    # WHY: trends must surface score progression in chronological order
    # EXPECTED: trend list length == number of runs, scores are floats
    for i in range(3):
        results = [{"name": f"t{j}", "passed": j <= i, "severity": "info",
                    "message": "", "details": {}, "duration_ms": 1.0}
                   for j in range(3)]
        storage.save_run("trend_proj", results)

    trends = storage.get_trends("trend_proj")
    assert len(trends) == 3
    for point in trends:
        assert "score" in point
        assert "timestamp" in point
        assert isinstance(point["score"], float)

    # chronological order: ids should be ascending
    ids = [p["id"] for p in trends]
    assert ids == sorted(ids)


def test_empty_db(storage):
    # SCENARIO: query a freshly created database
    # WHY: callers must handle zero-results gracefully
    # EXPECTED: get_runs returns [] and get_run returns None
    assert storage.get_runs() == []
    assert storage.get_run(999) is None
    assert storage.get_trends("nosuchproject") == []


def test_project_filter(storage):
    # SCENARIO: save runs for two different projects then filter by project
    # WHY: list_runs(project=X) must not return runs from other projects
    # EXPECTED: each project only sees its own runs
    storage.save_run("alpha", SAMPLE_RESULTS[:1])
    storage.save_run("alpha", SAMPLE_RESULTS[:1])
    storage.save_run("beta", SAMPLE_RESULTS[:2])

    alpha_runs = storage.get_runs(project="alpha")
    beta_runs = storage.get_runs(project="beta")

    assert len(alpha_runs) == 2
    assert len(beta_runs) == 1
    assert all(r["project"] == "alpha" for r in alpha_runs)
    assert all(r["project"] == "beta" for r in beta_runs)


def test_run_not_found(storage):
    # SCENARIO: request a run id that was never saved
    # WHY: get_run must return None rather than raising an exception
    # EXPECTED: None returned for unknown id
    result = storage.get_run(99999)
    assert result is None


def test_score_calculation(storage):
    # SCENARIO: save a fully-passing and a fully-failing run
    # WHY: score field must reflect pass-rate accurately
    # EXPECTED: 100% pass → score 100.0, 0% pass → score 0.0
    all_pass = [{"name": "t", "passed": True, "severity": "info",
                 "message": "", "details": {}, "duration_ms": 1.0}]
    all_fail = [{"name": "t", "passed": False, "severity": "error",
                 "message": "", "details": {}, "duration_ms": 1.0}]

    id_pass = storage.save_run("score_proj", all_pass)
    id_fail = storage.save_run("score_proj", all_fail)

    run_pass = storage.get_run(id_pass)
    run_fail = storage.get_run(id_fail)

    assert run_pass["score"] == pytest.approx(100.0)
    assert run_fail["score"] == pytest.approx(0.0)


def test_save_run_empty_results(storage):
    # SCENARIO: save a run with an empty results list
    # WHY: CI may submit a run before any tests execute; storage must not crash
    # EXPECTED: run is saved, total=0, passed=0, failed=0, score=0.0
    run_id = storage.save_run("empty_proj", [])
    assert isinstance(run_id, int)
    run = storage.get_run(run_id)
    assert run is not None
    assert run["total"] == 0
    assert run["passed"] == 0
    assert run["failed"] == 0
    assert run["score"] == pytest.approx(0.0)
    assert run["results"] == []


def test_save_run_large_payload(storage):
    # SCENARIO: save a run with 500 results
    # WHY: large test suites must be stored without truncation or error
    # EXPECTED: all 500 results are retrievable with correct counts
    results = [
        {"name": f"test_{i}", "passed": i % 2 == 0, "severity": "info",
         "message": "ok", "details": {}, "duration_ms": 1.0}
        for i in range(500)
    ]
    run_id = storage.save_run("large_proj", results)
    run = storage.get_run(run_id)
    assert run is not None
    assert run["total"] == 500
    assert run["passed"] == 250
    assert run["failed"] == 250
    assert len(run["results"]) == 500


def test_get_runs_limit(storage):
    # SCENARIO: save 10 runs then list with limit=3
    # WHY: callers must be able to cap results for pagination
    # EXPECTED: exactly 3 runs returned, the 3 most recent
    for i in range(10):
        storage.save_run("limit_proj", [
            {"name": f"t{i}", "passed": True, "severity": "info",
             "message": "", "details": {}, "duration_ms": 1.0}
        ])
    runs = storage.get_runs(project="limit_proj", limit=3)
    assert len(runs) == 3
    # Most recent first
    ids = [r["id"] for r in runs]
    assert ids == sorted(ids, reverse=True)


def test_details_json_roundtrip_complex(storage):
    # SCENARIO: save a result with nested/complex details dict
    # WHY: details_json column must faithfully round-trip nested structures
    #      including lists, booleans, and null values
    # EXPECTED: retrieved details matches the original dict exactly
    complex_details = {
        "scores": [0.91, 0.87, 0.93],
        "metadata": {"model": "v3", "threshold": 0.85},
        "is_baseline": False,
        "note": None,
    }
    results = [{
        "name": "test_complex",
        "passed": True,
        "severity": "info",
        "message": "ok",
        "details": complex_details,
        "duration_ms": 5.0,
    }]
    run_id = storage.save_run("detail_proj", results)
    run = storage.get_run(run_id)
    saved_result = run["results"][0]
    assert saved_result["details"] == complex_details


def test_webhook_crud_roundtrip(storage):
    # SCENARIO: save a webhook, list it, then delete it
    # WHY: full CRUD round-trip must work without errors
    # EXPECTED: webhook appears in get_webhooks, then disappears after delete
    wh_id = storage.save_webhook("https://example.com/hook", ["on_failure"], "proj-a")
    assert isinstance(wh_id, int)

    hooks = storage.get_webhooks("proj-a")
    assert len(hooks) == 1
    assert hooks[0].url == "https://example.com/hook"
    assert hooks[0].events == ["on_failure"]
    assert hooks[0].project == "proj-a"

    deleted = storage.delete_webhook(wh_id)
    assert deleted is True

    hooks_after = storage.get_webhooks("proj-a")
    assert len(hooks_after) == 0


def test_delete_nonexistent_webhook_returns_false(storage):
    # SCENARIO: attempt to delete a webhook id that was never registered
    # WHY: callers must be able to distinguish "deleted" from "not found"
    # EXPECTED: returns False without raising
    result = storage.delete_webhook(99999)
    assert result is False


def test_global_webhook_matches_all_projects(storage):
    # SCENARIO: save a global webhook (project=None), then query with a project name
    # WHY: global webhooks must be included in per-project queries — they
    #      subscribe to ALL projects
    # EXPECTED: global webhook appears in project-filtered results
    storage.save_webhook("https://global.example.com/hook", ["on_failure"], None)
    storage.save_webhook("https://specific.example.com/hook", ["on_success"], "proj-x")

    # Global webhook must appear in proj-x query
    hooks_for_proj_x = storage.get_webhooks("proj-x")
    urls = {h.url for h in hooks_for_proj_x}
    assert "https://global.example.com/hook" in urls
    assert "https://specific.example.com/hook" in urls

    # proj-y query gets only the global webhook (not proj-x specific one)
    hooks_for_proj_y = storage.get_webhooks("proj-y")
    urls_y = {h.url for h in hooks_for_proj_y}
    assert "https://global.example.com/hook" in urls_y
    assert "https://specific.example.com/hook" not in urls_y


# ---------------------------------------------------------------------------
# P1-16: WAL journal mode
# ---------------------------------------------------------------------------


def test_wal_mode_enabled(storage):
    # SCENARIO: after Storage init, SQLite journal mode should be WAL
    # WHY: WAL enables concurrent readers and improves write performance
    # EXPECTED: PRAGMA journal_mode returns "wal"
    row = storage._conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"


# ---------------------------------------------------------------------------
# P1-26: FOREIGN KEY enforcement
# ---------------------------------------------------------------------------


def test_foreign_key_enforcement(storage):
    # SCENARIO: insert a result row referencing a non-existent run_id
    # WHY: FOREIGN KEY constraints must be enforced to prevent orphan rows
    # EXPECTED: sqlite3.IntegrityError is raised
    with pytest.raises(sqlite3.IntegrityError):
        storage._conn.execute(
            "INSERT INTO results (run_id, name, passed, severity,"
            " message, details_json, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (999999, "orphan_test", 1, "info", "should not exist", "{}", 0.0),
        )
    storage._conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# P1-25: Singleton connection / close()
# ---------------------------------------------------------------------------


def test_close_method(tmp_path):
    # SCENARIO: create a Storage, close it, then verify the connection is shut
    # WHY: callers must be able to cleanly release the database file
    # EXPECTED: after close(), executing a query raises ProgrammingError
    db_file = str(tmp_path / "close_test.db")
    s = Storage(db_path=db_file)
    s.save_run("proj", [{"name": "t", "passed": True, "severity": "info",
                         "message": "", "details": {}, "duration_ms": 1.0}])
    s.close()
    with pytest.raises(sqlite3.ProgrammingError):
        s._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# P1-17: Database indexes exist
# ---------------------------------------------------------------------------


def test_indexes_created(storage):
    # SCENARIO: after Storage init, performance indexes should exist
    # WHY: indexes on runs(project), results(run_id), api_keys(key_hash)
    #      are critical for query performance at scale
    # EXPECTED: all three indexes are present in sqlite_master
    rows = storage._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    index_names = {r[0] for r in rows}
    assert "idx_runs_project" in index_names
    assert "idx_results_run" in index_names
    assert "idx_api_keys_hash" in index_names


# ---------------------------------------------------------------------------
# P1-18: Schema migration system
# ---------------------------------------------------------------------------


def test_schema_versions_table_exists(storage):
    # SCENARIO: after Storage init, a schema_versions table must exist
    # WHY: migration tracking requires this table
    # EXPECTED: table is present in sqlite_master
    rows = storage._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_versions'"
    ).fetchall()
    assert len(rows) == 1


def test_migration_v1_recorded(storage):
    # SCENARIO: fresh database should have migration v1 applied
    # WHY: v1 is the initial schema; _migrate() must record it
    # EXPECTED: schema_versions contains exactly one row with version=1
    rows = storage._conn.execute(
        "SELECT version, applied_at FROM schema_versions ORDER BY version"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1]  # applied_at is non-empty


def test_migration_is_idempotent(tmp_path):
    # SCENARIO: create Storage twice on the same database file
    # WHY: re-opening a database must NOT re-apply already-applied migrations
    # EXPECTED: schema_versions still has exactly one v1 row
    db_file = str(tmp_path / "idempotent.db")
    s1 = Storage(db_path=db_file)
    s1.close()

    s2 = Storage(db_path=db_file)
    rows = s2._conn.execute(
        "SELECT version FROM schema_versions ORDER BY version"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    s2.close()


def test_migrate_old_schema_adds_tracking(tmp_path):
    # SCENARIO: database has the old schema (tables exist but no schema_versions)
    # WHY: existing deployments created tables before the migration system;
    #      _migrate() must detect missing schema_versions and bootstrap it
    # EXPECTED: after re-opening with Storage, schema_versions exists and
    #           migration v1 is recorded; existing data is preserved
    db_file = str(tmp_path / "legacy.db")

    # Simulate a pre-migration database: create tables manually
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL DEFAULT 'default',
            timestamp TEXT NOT NULL,
            total INTEGER NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            score REAL NOT NULL DEFAULT 0.0,
            duration_ms REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(id),
            name TEXT NOT NULL,
            passed INTEGER NOT NULL DEFAULT 0,
            severity TEXT NOT NULL DEFAULT 'info',
            message TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL DEFAULT '{}',
            duration_ms REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE TABLE api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL UNIQUE,
            project TEXT NOT NULL DEFAULT 'default',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE webhooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            events_json TEXT NOT NULL DEFAULT '[]',
            project TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Insert some legacy data
    conn.execute(
        "INSERT INTO runs (project, timestamp, total, passed, failed, score, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("legacy-proj", "2025-01-01T00:00:00+00:00", 1, 1, 0, 100.0, 5.0),
    )
    conn.commit()
    conn.close()

    # Now open with Storage — migration system should bootstrap
    storage = Storage(db_path=db_file)

    # schema_versions should exist with v1 recorded
    rows = storage._conn.execute(
        "SELECT version FROM schema_versions"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1

    # Indexes should have been created by the migration
    idx_rows = storage._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    ).fetchall()
    idx_names = {r[0] for r in idx_rows}
    assert "idx_runs_project" in idx_names
    assert "idx_results_run" in idx_names
    assert "idx_api_keys_hash" in idx_names

    # Legacy data must still be present
    runs = storage.get_runs(project="legacy-proj")
    assert len(runs) == 1
    assert runs[0]["project"] == "legacy-proj"

    storage.close()


def test_get_current_version_empty_table(tmp_path):
    # SCENARIO: schema_versions table exists but has no rows
    # WHY: _get_current_version must return 0, not crash, when empty
    # EXPECTED: returns 0
    db_file = str(tmp_path / "empty_ver.db")
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE schema_versions (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    conn.commit()
    assert Storage._get_current_version(conn) == 0
    conn.close()


def test_migrations_list_is_ordered():
    # SCENARIO: the _MIGRATIONS list must have strictly increasing versions
    # WHY: out-of-order versions would cause migrations to be skipped or
    #      applied in the wrong sequence
    # EXPECTED: each version > the previous one
    versions = [v for v, _, _ in _MIGRATIONS]
    assert versions == sorted(versions)
    assert len(versions) == len(set(versions)), "Duplicate migration versions"
