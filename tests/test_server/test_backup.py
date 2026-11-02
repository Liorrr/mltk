"""Tests for safe online SQLite backups."""
from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing

import pytest

pytest.importorskip("fastapi", reason="server tests require fastapi")

from mltk.server.backup import backup_database
from mltk.server.storage import Storage


def test_live_wal_source_is_backed_up_without_wal_sidecar(tmp_path):
    db = tmp_path / "source.db"
    out = tmp_path / "backup.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE items (value TEXT)")
    writer.executemany("INSERT INTO items VALUES (?)", [("one",), ("two",)])
    writer.commit()
    try:
        plain = tmp_path / "plain.db"
        shutil.copyfile(db, plain)
        with closing(sqlite3.connect(plain)) as conn, conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT COUNT(*) FROM items")

        assert backup_database(db, out) == os.path.getsize(out)
        with closing(sqlite3.connect(out)) as conn, conn:
            assert conn.execute(
                "SELECT value FROM items ORDER BY value"
            ).fetchall() == [("one",), ("two",)]
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert not out.with_name(out.name + "-wal").exists()
    finally:
        writer.close()


def test_real_server_schema_round_trips(tmp_path):
    db = tmp_path / "server.db"
    out = tmp_path / "backup.db"
    storage = Storage(str(db))
    run_id = storage.save_run("project", [{"name": "check", "passed": True}])
    try:
        backup_database(db, out)
        restored = Storage(str(out))
        try:
            assert restored.get_run(run_id)["project"] == "project"
        finally:
            restored.close()
    finally:
        storage.close()


def test_missing_source_is_not_created(tmp_path):
    db = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="database not found"):
        backup_database(db, tmp_path / "out.db")
    assert not db.exists()


def test_existing_output_requires_force(tmp_path):
    db = tmp_path / "source.db"
    out = tmp_path / "out.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE items (value TEXT)")
        conn.execute("INSERT INTO items VALUES ('new')")
    out.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="already exists"):
        backup_database(db, out)
    assert out.read_bytes() == b"original"
    backup_database(db, out, force=True)
    assert out.read_bytes() != b"original"


def test_same_path_and_missing_output_directory(tmp_path):
    db = tmp_path / "source.db"
    sqlite3.connect(db).close()
    with pytest.raises(ValueError, match="database itself"):
        backup_database(db, db)
    with pytest.raises(FileNotFoundError, match="output directory does not exist"):
        backup_database(db, tmp_path / "missing" / "out.db")


def test_directory_output_is_refused_even_with_force(tmp_path):
    db = tmp_path / "source.db"
    sqlite3.connect(db).close()
    target = tmp_path / "backups"
    target.mkdir()
    with pytest.raises(IsADirectoryError, match="is a directory"):
        backup_database(db, target, force=True)
    assert target.is_dir()
    assert list(tmp_path.glob(".mltk-backup-*.tmp")) == []


def test_non_database_source_fails_without_leaving_files(tmp_path):
    db = tmp_path / "notes.txt"
    db.write_text("this is not a sqlite database. " * 10, encoding="utf-8")
    out = tmp_path / "out.db"
    with pytest.raises(sqlite3.DatabaseError):
        backup_database(db, out)
    assert not out.exists()
    assert list(tmp_path.glob(".mltk-backup-*.tmp")) == []


def test_failed_backup_removes_temp_file(tmp_path, monkeypatch):
    db = tmp_path / "source.db"
    out = tmp_path / "out.db"
    sqlite3.connect(db).close()

    def fail_backup(*args, **kwargs):
        raise RuntimeError("backup failed")

    monkeypatch.setattr("mltk.server.backup._backup", fail_backup)
    with pytest.raises(RuntimeError, match="backup failed"):
        backup_database(db, out)
    assert list(tmp_path.glob(".mltk-backup-*.tmp")) == []
