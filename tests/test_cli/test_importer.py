"""Tests for the ``mltk import`` CLI command."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[A-Za-z]|\x1b\]8;[^\x1b]*\x1b\\"
)
REPO_ROOT = Path(__file__).parents[2]
TINY_CSV_REL = "tests/test_importer/fixtures/tiny.csv"
TINY_CSV = REPO_ROOT / TINY_CSV_REL


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _run_cli(
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the mltk CLI via subprocess."""
    cli_args = list(args)
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {cli_args!r}; "
        "from mltk.cli.app import main; main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(cwd) if cwd is not None else None,
        env={
            **os.environ,
            "COLUMNS": "200",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
    )
    result.stdout = _strip_ansi(result.stdout)
    result.stderr = _strip_ansi(result.stderr)
    return result


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def test_import_happy_path_emits_pytest_file(tmp_path: Path) -> None:
    output_path = tmp_path / "test_tiny_import.py"

    result = _run_cli(
        "import",
        TINY_CSV_REL,
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, _combined_output(result)
    assert "column | role | sample" in result.stdout
    assert "qa_rag" in result.stdout
    assert "suite:" in result.stdout.lower()
    assert "registered assertions" in result.stdout
    assert str(output_path) in result.stdout
    assert output_path.exists()
    ast.parse(output_path.read_text(encoding="utf-8"))


def test_import_default_output_name_uses_source_and_task_type(
    tmp_path: Path,
) -> None:
    result = _run_cli("import", str(TINY_CSV), cwd=tmp_path)

    output_path = tmp_path / "test_tiny_qa_rag.py"
    assert result.returncode == 0, _combined_output(result)
    assert output_path.exists()
    ast.parse(output_path.read_text(encoding="utf-8"))


def test_import_refuses_to_overwrite_without_force(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "test_tiny_import.py"
    first = _run_cli(
        "import",
        TINY_CSV_REL,
        "--output",
        str(output_path),
    )
    assert first.returncode == 0, _combined_output(first)

    sentinel = "# user edits must survive\n"
    output_path.write_text(sentinel, encoding="utf-8")

    second = _run_cli(
        "import",
        TINY_CSV_REL,
        "--output",
        str(output_path),
    )
    assert second.returncode == 1
    assert "already exists" in _combined_output(second).lower()
    assert output_path.read_text(encoding="utf-8") == sentinel

    forced = _run_cli(
        "import",
        TINY_CSV_REL,
        "--output",
        str(output_path),
        "--force",
    )
    assert forced.returncode == 0, _combined_output(forced)
    assert output_path.read_text(encoding="utf-8") != sentinel
    ast.parse(output_path.read_text(encoding="utf-8"))


def test_import_no_emit_skips_pytest_file(tmp_path: Path) -> None:
    result = _run_cli(
        "import",
        str(TINY_CSV),
        "--no-emit",
        cwd=tmp_path,
    )

    assert result.returncode == 0, _combined_output(result)
    assert "column | role | sample" in result.stdout
    assert "task type:" in result.stdout.lower()
    assert not list(tmp_path.glob("test_*.py"))


def test_import_input_column_override_is_forwarded() -> None:
    result = _run_cli(
        "import",
        TINY_CSV_REL,
        "--input-column",
        "passage",
        "--no-emit",
    )

    assert result.returncode == 0, _combined_output(result)
    assert "passage | input |" in result.stdout


def test_import_input_column_override_is_emitted(tmp_path: Path) -> None:
    output_path = tmp_path / "test_tiny_import.py"

    result = _run_cli(
        "import",
        TINY_CSV_REL,
        "--input-column",
        "passage",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, _combined_output(result)
    content = output_path.read_text(encoding="utf-8")
    ast.parse(content)
    assert "input_column='passage'," in content


def test_import_missing_file_exits_with_helpful_message(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.csv"

    result = _run_cli("import", str(missing), "--no-emit")

    assert result.returncode == 1
    assert "not found" in _combined_output(result).lower()


def test_import_no_input_mapping_exits_with_validation_problem(
    tmp_path: Path,
) -> None:
    no_input = tmp_path / "bookkeeping.csv"
    no_input.write_text(
        "id,created_date\n1,2026-07-05\n",
        encoding="utf-8",
    )

    result = _run_cli("import", str(no_input), "--no-emit")

    combined = _combined_output(result).lower()
    assert result.returncode == 1
    assert "no column has role input" in combined
    assert "warning:" in combined


def test_import_help_does_not_require_importer_extra() -> None:
    root_help = _run_cli("--help")
    assert root_help.returncode == 0, _combined_output(root_help)
    assert "import" in root_help.stdout

    command_help = _run_cli("import", "--help")
    assert command_help.returncode == 0, _combined_output(command_help)
    for option in (
        "--split",
        "--input-column",
        "--target-column",
        "--output",
        "--force",
        "--no-emit",
    ):
        assert option in command_help.stdout


def test_importing_cli_app_does_not_import_importer_package() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mltk.cli.app; "
            "assert 'mltk.importer' not in sys.modules; "
            "assert 'datasets' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "COLUMNS": "200",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
    )
    assert proc.returncode == 0, proc.stderr
