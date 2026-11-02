"""Tests for the server-backup CLI command."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

# server-backup imports mltk.server, whose package import loads the FastAPI app.
pytest.importorskip("fastapi", reason="server-backup requires the server extra")


def _run_cli(
    *args: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    prelude: str = "",
):
    import subprocess
    import sys

    code = (
        f"{prelude}import sys; sys.argv=['mltk'] + {list(args)!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=_env(env),
    )


def _env(extra: dict[str, str] | None) -> dict[str, str]:
    """Child environment without an inherited MLTK_DB_PATH, plus ``extra``."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "MLTK_DB_PATH"}
    env.update(extra or {})
    return env


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


def test_server_backup_reads_mltk_db_path(tmp_path):
    db = tmp_path / "from_env.db"
    out = tmp_path / "backup.db"
    _make_db(db)
    result = _run_cli(
        "server-backup", "--out", str(out), env={"MLTK_DB_PATH": str(db)}, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"Backed up {db} to {out} (")


def test_explicit_db_wins_over_mltk_db_path(tmp_path):
    db = tmp_path / "explicit.db"
    out = tmp_path / "backup.db"
    _make_db(db)
    result = _run_cli(
        "server-backup", "--db", str(db), "--out", str(out),
        env={"MLTK_DB_PATH": str(tmp_path / "missing.db")},
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(f"Backed up {db} to {out} (")


def test_server_reads_mltk_db_path(tmp_path):
    """`mltk server` must open the MLTK_DB_PATH database, as the compose setup expects."""
    db = tmp_path / "compose.db"
    prelude = (
        "import types, sys; "
        "fake = types.ModuleType('uvicorn'); fake.run = lambda *a, **k: None; "
        "sys.modules['uvicorn'] = fake; "
        "import mltk.server as s; "
        "s.create_app = lambda db_path: print('DB=' + db_path); "
    )
    result = _run_cli(
        "server", env={"MLTK_DB_PATH": str(db)}, prelude=prelude, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert f"DB={db}" in result.stdout


def test_server_create_key_reads_mltk_db_path(tmp_path):
    db = tmp_path / "keys.db"
    result = _run_cli(
        "server-create-key", "--project", "p", env={"MLTK_DB_PATH": str(db)}, cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0] == 1
