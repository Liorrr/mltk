"""Tests for the server-backup CLI command."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

# server-backup imports mltk.server, whose package import loads the FastAPI app.
pytest.importorskip("fastapi", reason="server-backup requires the server extra")


def _run_cli(*args: str, cwd: str | None = None):
    import subprocess
    import sys

    code = (
        f"import sys; sys.argv=['mltk'] + {list(args)!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )


def _make_db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE items (value TEXT)")
        conn.execute("INSERT INTO items VALUES ('value')")


def test_success_prints_size_and_writes_backup(tmp_path):
    db = tmp_path / "db.sqlite"
    out = tmp_path / "backup.sqlite"
    _make_db(db)
    result = _run_cli("server-backup", "--db", str(db), "--out", str(out))
    assert result.returncode == 0
    assert result.stdout == f"Backed up {db} to {out} ({out.stat().st_size} bytes)\n"
    assert out.exists()


def test_missing_db_and_existing_output_errors(tmp_path):
    db = tmp_path / "missing.db"
    out = tmp_path / "backup.db"
    result = _run_cli("server-backup", "--db", str(db), "--out", str(out))
    assert result.returncode == 1
    assert "Error: database not found" in result.stderr
    assert "Traceback" not in result.stderr

    _make_db(db)
    out.write_bytes(b"old")
    result = _run_cli("server-backup", "--db", str(db), "--out", str(out))
    assert result.returncode == 1
    assert "--force" in result.stderr
    assert out.read_bytes() == b"old"
    forced = _run_cli("server-backup", "--db", str(db), "--out", str(out), "--force")
    assert forced.returncode == 0


def test_out_is_required(tmp_path):
    db = tmp_path / "db.sqlite"
    _make_db(db)
    result = _run_cli("server-backup", "--db", str(db))
    assert result.returncode != 0


def test_non_database_file_exits_1_without_traceback(tmp_path):
    db = tmp_path / "notes.txt"
    db.write_text("this is not a sqlite database. " * 10, encoding="utf-8")
    out = tmp_path / "backup.db"
    result = _run_cli("server-backup", "--db", str(db), "--out", str(out))
    assert result.returncode == 1
    assert "Error: cannot back up" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()
