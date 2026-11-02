"""Online backups for the mltk server SQLite database."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path


def _backup(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
    """Copy a database using SQLite's online backup API."""
    source.backup(destination)


def backup_database(
    db_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    force: bool = False,
) -> int:
    """Create a self-contained online backup and return its size in bytes.

    The server opens its database in WAL mode, so recent commits can live in
    ``<db>-wal`` rather than the main file, and a plain file copy of a live
    database can lose them. This copies through SQLite's online backup API,
    which is safe while the server is running. The copy is switched to
    rollback-journal mode, so it is one self-contained file, and must pass
    ``PRAGMA integrity_check``. It is written to a temp file in the output
    directory and renamed into place, so a failed backup leaves no partial
    file and never touches an existing output.

    Raises:
        FileNotFoundError: ``db_path`` or the output directory does not exist.
        IsADirectoryError: ``out_path`` is a directory.
        FileExistsError: ``out_path`` exists and ``force`` is False.
        ValueError: ``out_path`` is the database itself.
        RuntimeError: the copy failed ``PRAGMA integrity_check``.
        sqlite3.DatabaseError: ``db_path`` is not a SQLite database.
    """
    source_path = Path(db_path)
    if not source_path.exists():
        raise FileNotFoundError(f"database not found: {source_path}")

    source_path = source_path.resolve()
    destination_path = Path(out_path).resolve()
    if source_path == destination_path:
        raise ValueError(f"backup output is the database itself: {destination_path}")
    if destination_path.is_dir():
        raise IsADirectoryError(f"backup output is a directory: {destination_path}")
    if destination_path.exists() and not force:
        raise FileExistsError(f"backup file already exists: {destination_path}")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(
            f"output directory does not exist: {destination_path.parent}"
        )

    temp_path: str | None = None
    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    try:
        fd, temp_path = tempfile.mkstemp(
            dir=str(destination_path.parent),
            prefix=".mltk-backup-",
            suffix=".tmp",
        )
        os.close(fd)

        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(temp_path)
        _backup(source, destination)
        destination.execute("PRAGMA journal_mode=DELETE")
        result = destination.execute("PRAGMA integrity_check").fetchone()
        integrity = result[0] if result else None
        if integrity != "ok":
            raise RuntimeError(f"backup failed integrity check: {integrity}")
        destination.commit()
        destination.close()
        destination = None
        source.close()
        source = None
        os.replace(temp_path, destination_path)
        temp_path = None
        return os.path.getsize(destination_path)
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
