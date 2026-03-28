"""Tests for ``mltk list`` CLI command.

The ``list`` command scans mltk subpackages for ``assert_*`` functions and
prints them grouped by category.  These tests exercise:

1. Default table output contains a total count and known assertions
2. ``--filter`` narrows results by name / module / docstring
3. ``--format json`` produces valid, well-structured JSON
4. Unknown ``--format`` exits with an error message
5. Known assertions (assert_schema, assert_no_drift) appear in output
6. Filter with zero matches still exits cleanly
7. Direct unit tests of the discovery module
"""

from __future__ import annotations

import json
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(
    *args: str,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the mltk CLI via subprocess and return the result."""
    cli_args = list(args)
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {cli_args!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# TestListTable
# ---------------------------------------------------------------------------

class TestListTable:
    """``mltk list`` default table output."""

    def test_shows_total_count(self) -> None:
        """Table header includes 'mltk assertions (N total)'."""
        result = _run_cli("list")
        assert result.returncode == 0, result.stderr
        assert "mltk assertions (" in result.stdout
        assert "total)" in result.stdout

    def test_contains_known_assertion_schema(self) -> None:
        """assert_schema appears in the Data Quality section."""
        result = _run_cli("list")
        assert result.returncode == 0, result.stderr
        assert "assert_schema" in result.stdout

    def test_contains_known_assertion_no_drift(self) -> None:
        """assert_no_drift appears in the output."""
        result = _run_cli("list")
        assert result.returncode == 0, result.stderr
        assert "assert_no_drift" in result.stdout

    def test_categories_present(self) -> None:
        """At least Data Quality and Model Quality categories show."""
        result = _run_cli("list")
        assert result.returncode == 0, result.stderr
        assert "Data Quality" in result.stdout
        assert "Model Quality" in result.stdout


# ---------------------------------------------------------------------------
# TestListFilter
# ---------------------------------------------------------------------------

class TestListFilter:
    """``mltk list <keyword>`` filters results."""

    def test_filter_drift(self) -> None:
        """Filtering by 'drift' returns only drift-related assertions."""
        result = _run_cli("list", "drift")
        assert result.returncode == 0, result.stderr
        assert "assert_no_drift" in result.stdout
        # A non-drift assertion should be absent.
        assert "assert_schema" not in result.stdout

    def test_filter_no_match(self) -> None:
        """A filter with zero results still exits cleanly."""
        result = _run_cli("list", "zzz_nonexistent_xyz")
        assert result.returncode == 0, result.stderr
        assert "0 total" in result.stdout

    def test_filter_case_insensitive(self) -> None:
        """Filter matching is case-insensitive."""
        result = _run_cli("list", "DRIFT")
        assert result.returncode == 0, result.stderr
        assert "assert_no_drift" in result.stdout


# ---------------------------------------------------------------------------
# TestListJson
# ---------------------------------------------------------------------------

class TestListJson:
    """``mltk list --format json`` output."""

    def test_valid_json(self) -> None:
        """Output is valid JSON with expected top-level keys."""
        result = _run_cli("list", "--format", "json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert "total" in payload
        assert "modules" in payload
        assert isinstance(payload["total"], int)
        assert isinstance(payload["modules"], dict)

    def test_json_total_positive(self) -> None:
        """Total count is a positive integer."""
        result = _run_cli("list", "--format", "json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["total"] > 0

    def test_json_entry_shape(self) -> None:
        """Each entry has name, module, and doc keys."""
        result = _run_cli("list", "--format", "json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        for _cat, items in payload["modules"].items():
            for entry in items:
                assert "name" in entry
                assert "module" in entry
                assert "doc" in entry
                assert entry["name"].startswith("assert_")

    def test_json_filter(self) -> None:
        """Filter also works in JSON mode."""
        result = _run_cli("list", "schema", "--format", "json")
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        all_names = [
            e["name"]
            for items in payload["modules"].values()
            for e in items
        ]
        assert "assert_schema" in all_names
        # Unrelated assertions should be absent.
        assert "assert_no_toxicity" not in all_names


# ---------------------------------------------------------------------------
# TestListErrors
# ---------------------------------------------------------------------------

class TestListErrors:
    """Error handling for bad arguments."""

    def test_unknown_format(self) -> None:
        """An unknown format exits with code 1 and a message."""
        result = _run_cli("list", "--format", "yaml")
        assert result.returncode != 0
        assert "Unknown format" in result.stdout


# ---------------------------------------------------------------------------
# TestDiscoveryUnit — direct tests of the discovery module
# ---------------------------------------------------------------------------

class TestDiscoveryUnit:
    """Unit tests for mltk.cli._discovery without subprocess."""

    def test_discover_returns_dict(self) -> None:
        """discover_assertions returns a non-empty dict."""
        from mltk.cli._discovery import discover_assertions

        result = discover_assertions()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_discover_total_exceeds_100(self) -> None:
        """mltk ships 100+ assertions; verify discovery finds them."""
        from mltk.cli._discovery import discover_assertions

        result = discover_assertions()
        total = sum(len(v) for v in result.values())
        assert total > 100, f"Expected >100 assertions, got {total}"

    def test_discover_filter_narrows(self) -> None:
        """Filtering reduces the total count."""
        from mltk.cli._discovery import discover_assertions

        all_entries = discover_assertions()
        filtered = discover_assertions("drift")
        total_all = sum(len(v) for v in all_entries.values())
        total_filt = sum(len(v) for v in filtered.values())
        assert total_filt < total_all
        assert total_filt > 0

    def test_discover_entry_has_required_keys(self) -> None:
        """Every entry dict has name, module, doc."""
        from mltk.cli._discovery import discover_assertions

        for _cat, items in discover_assertions().items():
            for entry in items:
                assert set(entry.keys()) == {"name", "module", "doc"}
                assert entry["name"].startswith("assert_")
