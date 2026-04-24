"""Tests for scripts/bump.py."""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable without an __init__.py
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bump  # noqa: E402
from bump import (  # noqa: E402
    LiveCounts,
    _replace_counts_in_text,
    _roll_changelog,
    cmd_refresh,
    cmd_verify,
)


def _make_counts(
    *,
    assertions: int = 230,
    cli: int = 24,
    mcp: int = 11,
    scanners: int = 8,
    tests: int = 4247,
    version: str = "0.9.0",
) -> LiveCounts:
    return LiveCounts(
        assertions=assertions,
        cli=cli,
        mcp=mcp,
        scanners=scanners,
        tests=tests,
        version=version,
    )


class TestRefreshIdempotent:
    def test_refresh_twice_no_second_changes(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts()
        doc = tmp_path / "doc.md"
        doc.write_text("We have 224 assertions and 3388+ tests and 8 scanners.", encoding="utf-8")

        monkeypatch.setattr(bump, "COUNT_TARGETS", ["doc.md"])
        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "get_live_counts", lambda: counts)

        first_modified = cmd_refresh(counts=counts)
        assert len(first_modified) == 1

        second_modified = cmd_refresh(counts=counts)
        assert second_modified == [], "Second refresh should find nothing stale"


class TestVerifyDrift:
    def test_exits_nonzero_on_drift(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts(assertions=230, tests=4247)
        doc = tmp_path / "stale.md"
        doc.write_text("Only 224 assertions here.", encoding="utf-8")

        monkeypatch.setattr(bump, "COUNT_TARGETS", ["stale.md"])
        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "get_live_counts", lambda: counts)

        exit_code = cmd_verify()
        assert exit_code == 1

    def test_exits_zero_when_current(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts(assertions=230, tests=4247)
        doc = tmp_path / "fresh.md"
        doc.write_text("We have 230 assertions and 4247+ tests.", encoding="utf-8")

        monkeypatch.setattr(bump, "COUNT_TARGETS", ["fresh.md"])
        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "get_live_counts", lambda: counts)

        exit_code = cmd_verify()
        assert exit_code == 0


class TestDryRunNoWrites:
    def test_dry_run_leaves_files_unchanged(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts(assertions=230, tests=4247, version="1.0.0")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('version = "0.9.0"\n', encoding="utf-8")
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n", encoding="utf-8")
        doc = tmp_path / "README.md"
        doc.write_text("224 assertions, 3388+ tests\n", encoding="utf-8")

        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "PYPROJECT", pyproject)
        monkeypatch.setattr(bump, "CHANGELOG", changelog)
        monkeypatch.setattr(bump, "VERSION_TARGETS", ["pyproject.toml"])
        monkeypatch.setattr(bump, "COUNT_TARGETS", ["README.md"])
        monkeypatch.setattr(bump, "get_live_counts", lambda: counts)
        monkeypatch.setattr(bump, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(bump, "_git_add", lambda _: None)

        original_pyproject = pyproject.read_text(encoding="utf-8")
        original_changelog = changelog.read_text(encoding="utf-8")
        original_doc = doc.read_text(encoding="utf-8")

        exit_code = bump.cmd_release("1.0.0", dry_run=True)

        assert exit_code == 0
        assert pyproject.read_text(encoding="utf-8") == original_pyproject
        assert changelog.read_text(encoding="utf-8") == original_changelog
        assert doc.read_text(encoding="utf-8") == original_doc


class TestSinceExclusion:
    def test_since_version_not_rewritten(self) -> None:
        counts = _make_counts(assertions=230)
        text = "Has 224 assertions total. See Since: v0.9.0 for history."
        result = _replace_counts_in_text(text, counts)
        # The standalone count should update
        assert "230 assertions" in result
        # The version after Since: must remain untouched
        assert "Since: v0.9.0" in result

    def test_count_after_since_not_rewritten(self) -> None:
        counts = _make_counts(assertions=230)
        # Simulate a table row where the count is right after a Since cell
        # The count is within 30 chars of "Since"
        text = "| Since | 224 assertions |\n| also | 224 assertions |\n"
        result = _replace_counts_in_text(text, counts)
        # The second occurrence (not near Since) must be updated
        assert "230 assertions" in result

    def test_assertion_count_in_normal_text_updated(self) -> None:
        counts = _make_counts(assertions=230)
        text = "We have 224 assertions and they are great."
        result = _replace_counts_in_text(text, counts)
        assert "230 assertions" in result
        assert "224 assertions" not in result


class TestChangelogRoll:
    def test_unreleased_becomes_versioned(self) -> None:
        today = __import__("datetime").date.today().isoformat()
        text = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- something\n"
        result = _roll_changelog(text, "1.0.0")
        assert f"## [1.0.0] — {today}" in result
        assert "## [Unreleased]" in result

    def test_fresh_stub_inserted_above_versioned(self) -> None:
        text = "# Changelog\n\n## [Unreleased]\n\n### Added\n"
        result = _roll_changelog(text, "1.0.0")
        idx_unreleased = result.index("## [Unreleased]")
        idx_versioned = result.index("## [1.0.0]")
        assert idx_unreleased < idx_versioned

    def test_stub_sections_present(self) -> None:
        text = "# Changelog\n\n## [Unreleased]\n\n### Added\n"
        result = _roll_changelog(text, "1.0.0")
        assert "### Added" in result
        assert "### Changed" in result
        assert "### Fixed" in result

    def test_raises_if_no_unreleased(self) -> None:
        import pytest
        text = "# Changelog\n\n## [0.9.0] — 2025-01-01\n"
        with pytest.raises(ValueError, match="Unreleased"):
            _roll_changelog(text, "1.0.0")


class TestVersionBumpPyproject:
    def test_pyproject_version_updated(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts(version="1.0.0")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "0.9.0"\n', encoding="utf-8")
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n", encoding="utf-8")

        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "PYPROJECT", pyproject)
        monkeypatch.setattr(bump, "CHANGELOG", changelog)
        monkeypatch.setattr(bump, "VERSION_TARGETS", ["pyproject.toml"])
        monkeypatch.setattr(bump, "COUNT_TARGETS", [])
        monkeypatch.setattr(bump, "get_live_counts", lambda: counts)
        monkeypatch.setattr(bump, "get_current_version", lambda: "0.9.0")
        monkeypatch.setattr(bump, "_git_add", lambda _: None)

        exit_code = bump.cmd_release("1.0.0", dry_run=False)

        assert exit_code == 0
        content = pyproject.read_text(encoding="utf-8")
        assert 'version = "1.0.0"' in content
        assert 'version = "0.9.0"' not in content

    def test_invalid_version_exits_nonzero(self, monkeypatch: object) -> None:
        exit_code = bump.cmd_release("not-a-version", dry_run=False)
        assert exit_code == 1
