"""Tests for the server-create-key role option."""
from __future__ import annotations

import sqlite3
import subprocess
import sys

import pytest

pytest.importorskip("typer")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the mltk CLI in a child interpreter (same pattern as test_cli)."""
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {list(args)!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
    )


def _stored_roles(path):
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT role FROM api_keys ORDER BY id")]


def test_server_create_key_role_option(tmp_path):
    db_path = tmp_path / "keys.db"
    reader = _run_cli("server-create-key", "--db", str(db_path), "--role", "reader")
    assert reader.returncode == 0, reader.stdout + reader.stderr
    assert "role 'reader'" in reader.stdout
    default = _run_cli("server-create-key", "--db", str(db_path))
    assert default.returncode == 0, default.stdout + default.stderr
    assert _stored_roles(db_path) == ["reader", "writer"]
    bogus = _run_cli("server-create-key", "--db", str(db_path), "--role", "bogus")
    assert bogus.returncode != 0
    assert _stored_roles(db_path) == ["reader", "writer"]
