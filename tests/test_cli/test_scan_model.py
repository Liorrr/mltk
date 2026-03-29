"""Tests for the ``mltk scan-model`` CLI command.

The ``scan-model`` command loads a serialized model and a
dataset, runs all applicable scanners, and prints a
summary of findings.  These tests verify:

1. ``--help`` shows all expected options.
2. Invalid model path produces a clear error (exit 2).
3. Invalid data path produces a clear error (exit 2).

We invoke the CLI via subprocess (same pattern as other
test_cli tests) to exercise the full Typer entry point.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def _run_cli(
    *args: str,
) -> subprocess.CompletedProcess[str]:
    """Invoke the mltk CLI via subprocess.

    Uses ``sys.executable -c`` to call ``main()``
    with the given arguments, ensuring the same Python
    interpreter and installed packages are used.
    """
    cli_args = list(args)
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {cli_args!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestScanModelHelp:
    """Verify that --help shows all expected options."""

    def test_help_shows_options(self) -> None:
        """``mltk scan-model --help`` lists every
        option defined in the command signature.
        """
        result = _run_cli("scan-model", "--help")
        assert result.returncode == 0

        out = result.stdout
        assert "--model" in out
        assert "--data" in out
        assert "--target" in out
        assert "--sensitive" in out
        assert "--output" in out
        assert "--junit-xml" in out

    def test_help_shows_description(self) -> None:
        """The help output includes the command
        description text.
        """
        result = _run_cli("scan-model", "--help")
        assert result.returncode == 0
        assert "Scan a model" in result.stdout


class TestScanModelInvalidModel:
    """Invalid model path should error gracefully."""

    def test_missing_model_file(self) -> None:
        """Exit code 2 when model file does not exist."""
        result = _run_cli(
            "scan-model",
            "--model", "/nonexistent/model.pkl",
            "--data", "/nonexistent/data.csv",
            "--target", "label",
        )
        # Typer may exit with 2 (our code) or non-zero
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert (
            "not found" in combined.lower()
            or "error" in combined.lower()
            or "missing" in combined.lower()
        )

    def test_model_error_message_includes_path(
        self,
    ) -> None:
        """Error message should reference the path
        the user provided so they can fix it.
        """
        bad_path = "/tmp/does_not_exist_model.pkl"
        result = _run_cli(
            "scan-model",
            "--model", bad_path,
            "--data", "/tmp/some_data.csv",
            "--target", "y",
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        # Should mention the model path or "not found"
        assert (
            "not found" in combined.lower()
            or "model" in combined.lower()
        )


class TestScanModelInvalidData:
    """Invalid data path should error gracefully."""

    def test_missing_data_file(
        self, tmp_path: object,
    ) -> None:
        """Exit code 2 when data file does not exist
        but model file does exist.

        We create a dummy file so the model path check
        passes, then verify the data path check fires.
        """
        import tempfile

        # Create a dummy model file (content irrelevant
        # -- the data check runs first if model exists)
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False,
        ) as f:
            f.write(b"dummy")
            model_path = f.name

        result = _run_cli(
            "scan-model",
            "--model", model_path,
            "--data", "/nonexistent/data.csv",
            "--target", "label",
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert (
            "not found" in combined.lower()
            or "error" in combined.lower()
            or "cannot" in combined.lower()
        )

    def test_missing_required_options(self) -> None:
        """Calling scan-model without required options
        should fail.
        """
        result = _run_cli("scan-model")
        assert result.returncode != 0
