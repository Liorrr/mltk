"""GitHub Release notes come from the tag's CHANGELOG section (`bump.py notes`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """\
# Changelog

## [Unreleased]

### Added

### Changed

### Fixed

## [1.1.0] — 2026-02-01

### Added

### Fixed

## [1.0.0] — 2026-01-01

### Added
- **Thing** — it works.

### Fixed
- A bug.

## [0.9.0] — 2025-12-01
- Oldest entry.
"""


@pytest.fixture
def bump() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_mltk_bump_notes_under_test", ROOT / "scripts" / "bump.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mltk_bump_notes_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_section_stops_at_next_heading(bump: ModuleType) -> None:
    body = bump._changelog_section(SAMPLE, "1.0.0")
    assert body.startswith("### Added\n- **Thing** — it works.")
    assert body.endswith("- A bug.")
    assert "## [" not in body


def test_last_section_runs_to_end_of_file(bump: ModuleType) -> None:
    assert bump._changelog_section(SAMPLE, "0.9.0") == "- Oldest entry."


@pytest.mark.parametrize("version", ["2.0.0", "1.0", "1.0.00", "0.9"])
def test_missing_section_raises(bump: ModuleType, version: str) -> None:
    with pytest.raises(ValueError, match="No '## \\["):
        bump._changelog_section(SAMPLE, version)


def test_section_with_only_headings_raises(bump: ModuleType) -> None:
    with pytest.raises(ValueError, match="no entries"):
        bump._changelog_section(SAMPLE, "1.1.0")


def test_release_notes_accept_tag_name_and_add_install_footer(bump: ModuleType) -> None:
    notes = bump._release_notes(SAMPLE, "v1.0.0")
    assert notes.startswith("### Added\n- **Thing**")
    assert "`pip install mlspec==1.0.0`" in notes
    assert "`ghcr.io/liorrr/mltk:v1.0.0`" in notes
    assert "v1.0.0-full" in notes


def test_real_changelog_has_notes_for_the_released_version(bump: ModuleType) -> None:
    version = bump.get_current_version()
    notes = bump._release_notes(bump.CHANGELOG.read_text(encoding="utf-8"), version)
    assert f"mlspec=={version}" in notes


def test_cmd_notes_writes_file(
    bump: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(bump, "CHANGELOG", changelog)
    out = tmp_path / "notes.md"
    assert bump.cmd_notes("1.0.0", out) == 0
    assert out.read_text(encoding="utf-8").startswith("### Added\n- **Thing**")


def test_cmd_notes_fails_without_a_section(
    bump: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setattr(bump, "CHANGELOG", changelog)
    out = tmp_path / "notes.md"
    assert bump.cmd_notes("1.1.0", out) == 1
    assert not out.exists()
    assert "no entries" in capsys.readouterr().err
