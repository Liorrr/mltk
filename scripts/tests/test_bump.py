"""Tests for scripts/bump.py."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable without an __init__.py
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bump  # noqa: E402
from bump import (  # noqa: E402
    LiveCounts,
    _bump_formula,
    _replace_counts_in_text,
    _roll_changelog,
    cmd_refresh,
    cmd_verify,
)

REAL_FORMULA = (Path(__file__).parent / "fixtures" / "homebrew_mltk_0_16_0.rb").read_text(
    encoding="utf-8"
)


def _make_counts(
    *,
    assertions: int = 230,
    cli: int = 24,
    cli_summary: str = "19 top-level + 5 groups",
    mcp: int = 11,
    scanners: int = 8,
    tests: int = 4247,
    version: str = "0.9.0",
) -> LiveCounts:
    return LiveCounts(
        assertions=assertions,
        cli=cli,
        cli_summary=cli_summary,
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
        monkeypatch.setattr(bump, "get_live_counts", lambda **_kw: counts)

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
        monkeypatch.setattr(bump, "get_live_counts", lambda **_kw: counts)

        exit_code = cmd_verify()
        assert exit_code == 1

    def test_exits_zero_when_current(self, tmp_path: Path, monkeypatch: object) -> None:
        counts = _make_counts(assertions=230, tests=4247)
        doc = tmp_path / "fresh.md"
        doc.write_text("We have 230 assertions and 4247+ tests.", encoding="utf-8")

        monkeypatch.setattr(bump, "COUNT_TARGETS", ["fresh.md"])
        monkeypatch.setattr(bump, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(bump, "get_live_counts", lambda **_kw: counts)

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
        monkeypatch.setattr(bump, "get_live_counts", lambda **_kw: counts)
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
        monkeypatch.setattr(bump, "get_live_counts", lambda **_kw: counts)
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


class TestHomebrewFormula:
    def test_bumps_only_top_level_lines(self) -> None:
        updated = _bump_formula(REAL_FORMULA, "https://example/mlspec-0.17.0.tar.gz", "newhash")
        old_url = (
            'url "https://files.pythonhosted.org/packages/80/86/'
            '248f1dd2b796920649d226c0843c9192923a8ef74086b119f2a0f1e4139f/'
            'mlspec-0.16.0.tar.gz"'
        )
        old_sha = 'sha256 "2a927686cbd60583f62abb83265b13c067b59e2e7b45b7474e804b15a4c132d3"'
        assert updated.replace('url "https://example/mlspec-0.17.0.tar.gz"', old_url).replace(
            'sha256 "newhash"', old_sha
        ) == REAL_FORMULA
        assert 'version ' not in updated

    @pytest.mark.parametrize("text", [
        'class Mltk < Formula\n  sha256 "old"\nend\n',
        'class Mltk < Formula\n  url "one"\n  url "two"\n  sha256 "old"\nend\n',
    ])
    def test_requires_exactly_one_top_level_url(self, text: str) -> None:
        with pytest.raises(ValueError, match="top-level url"):
            _bump_formula(text, "new-url", "new-hash")

    @pytest.mark.parametrize("text", [
        'class Mltk < Formula\n  url "old"\nend\n',
        'class Mltk < Formula\n  url "one"\n  sha256 "one"\n  sha256 "two"\nend\n',
    ])
    def test_requires_exactly_one_top_level_sha256(self, text: str) -> None:
        with pytest.raises(ValueError, match="top-level sha256"):
            _bump_formula(text, "new-url", "new-hash")

    @pytest.mark.parametrize("stale_line", [
        '  version "0.16.0"',
        '  revision 1',
        '  mirror "https://example.com/mlspec.tar.gz"',
        '  bottle do',
    ])
    def test_rejects_stale_top_level_lines(self, stale_line: str) -> None:
        text = REAL_FORMULA.replace(
            '  sha256 "2a927686cbd60583f62abb83265b13c067b59e2e7b45b7474e804b15a4c132d3"\n',
            '  sha256 "2a927686cbd60583f62abb83265b13c067b59e2e7b45b7474e804b15a4c132d3"\n'
            f"{stale_line}\n",
            1,
        )
        with pytest.raises(ValueError, match="manual update"):
            _bump_formula(text, "https://example/mlspec-0.17.0.tar.gz", "new-hash")

    def test_replacement_values_are_literal(self) -> None:
        text = 'class Mltk < Formula\n  url "old"\n  sha256 "old-hash"\nend\n'
        result = _bump_formula(text, r"new\url", r"new\sha\1")
        assert r'  url "new\url"' in result
        assert r'  sha256 "new\sha\1"' in result

    def test_does_not_change_resource_block(self) -> None:
        text = (
            'class Mltk < Formula\n  url "old"\n  sha256 "old-hash"\n'
            '    resource "x" do\n      url "nested"\n      sha256 "nested-hash"\n'
            '    end\nend\n'
        )
        result = _bump_formula(text, "new", "new-hash")
        assert '  url "new"' in result
        assert '  sha256 "new-hash"' in result
        assert '      url "nested"' in result
        assert '      sha256 "nested-hash"' in result

    def test_preserves_crlf(self) -> None:
        text = 'class Mltk < Formula\r\n  url "old"\r\n  sha256 "old-hash"\r\nend\r\n'
        result = _bump_formula(text, "new", "new-hash")
        assert result == 'class Mltk < Formula\r\n  url "new"\r\n  sha256 "new-hash"\r\nend\r\n'


class TestHomebrewCommand:
    def _setup_tap(self, tmp_path: Path) -> tuple[Path, Path]:
        formula = tmp_path / "Formula" / "mltk.rb"
        formula.parent.mkdir()
        formula.write_bytes(REAL_FORMULA.encode())
        return tmp_path, formula

    def test_end_to_end_and_idempotence(
        self, tmp_path: Path, monkeypatch: object, capsys: object
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        data = b"real sdist bytes"
        url = "https://files.pythonhosted.org/packages/mlspec-0.17.0.tar.gz"
        digest = hashlib.sha256(data).hexdigest()
        payload = json.dumps({"urls": [{
            "packagetype": "sdist", "url": url, "digests": {"sha256": digest}
        }]}).encode()
        monkeypatch.setattr(
            bump, "_fetch", lambda requested: payload if requested.endswith("/json") else data
        )

        assert bump.cmd_homebrew("v0.17.0", tap=tap) == 0
        assert 'mlspec-0.17.0.tar.gz' in formula.read_text()
        assert digest in formula.read_text()
        assert b"\r" not in formula.read_bytes()
        assert bump.cmd_homebrew("0.17.0", tap=tap) == 0
        assert "already at 0.17.0" in capsys.readouterr().out

    @pytest.mark.parametrize("sdist_count", [0, 2])
    def test_rejects_wrong_sdist_count(
        self,
        tmp_path: Path,
        monkeypatch: object,
        capsys: object,
        sdist_count: int,
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        original = formula.read_bytes()
        data = b"real sdist bytes"
        digest = hashlib.sha256(data).hexdigest()
        urls = [{
            "packagetype": "sdist",
            "url": "https://example/mlspec-0.17.0.tar.gz",
            "digests": {"sha256": digest},
        }] * sdist_count
        payload = json.dumps({"urls": urls}).encode()
        monkeypatch.setattr(
            bump, "_fetch", lambda requested: payload if requested.endswith("/json") else data
        )
        assert bump.cmd_homebrew("0.17.0", tap=tap) == 1
        assert formula.read_bytes() == original
        assert f"found {sdist_count}" in capsys.readouterr().err

    def test_rejects_digest_mismatch_without_writing(
        self, tmp_path: Path, monkeypatch: object
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        original = formula.read_bytes()
        url = "https://example/mlspec-0.17.0.tar.gz"
        metadata = {"urls": [{"packagetype": "sdist", "url": url, "digests": {"sha256": "wrong"}}]}
        monkeypatch.setattr(
            bump,
            "_fetch",
            lambda requested: json.dumps(metadata).encode()
            if requested.endswith("/json") else b"bytes",
        )
        assert bump.cmd_homebrew("0.17.0", tap=tap) == 1
        assert formula.read_bytes() == original

    def test_rejects_filename_version_mismatch(
        self, tmp_path: Path, monkeypatch: object, capsys: object
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        original = formula.read_bytes()
        data = b"real sdist bytes"
        digest = hashlib.sha256(data).hexdigest()
        url = "https://example/mlspec-0.16.0.tar.gz"
        metadata = {
            "urls": [{
                "packagetype": "sdist",
                "url": url,
                "digests": {"sha256": digest},
            }]
        }
        monkeypatch.setattr(
            bump,
            "_fetch",
            lambda requested: json.dumps(metadata).encode()
            if requested.endswith("/json") else data,
        )
        assert bump.cmd_homebrew("0.17.0", tap=tap) == 1
        assert formula.read_bytes() == original
        assert "mlspec-0.16.0.tar.gz" in capsys.readouterr().err

    def test_rejects_yanked_sdist_without_writing(
        self, tmp_path: Path, monkeypatch: object, capsys: object
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        original = formula.read_bytes()
        data = b"real sdist bytes"
        url = "https://example/mlspec-0.17.0.tar.gz"
        digest = hashlib.sha256(data).hexdigest()
        metadata = {"urls": [{
            "packagetype": "sdist",
            "url": url,
            "digests": {"sha256": digest},
            "yanked": True,
            "yanked_reason": "security issue",
        }]}
        monkeypatch.setattr(
            bump,
            "_fetch",
            lambda requested: json.dumps(metadata).encode()
            if requested.endswith("/json") else data,
        )

        assert bump.cmd_homebrew("0.17.0", tap=tap) == 1
        assert formula.read_bytes() == original
        assert "yanked" in capsys.readouterr().err

    def test_dry_run_and_missing_formula(
        self, tmp_path: Path, monkeypatch: object, capsys: object
    ) -> None:
        tap, formula = self._setup_tap(tmp_path)
        data = b"bytes"
        url = "https://example/mlspec-0.17.0.tar.gz"
        digest = hashlib.sha256(data).hexdigest()
        metadata = {"urls": [{"packagetype": "sdist", "url": url, "digests": {"sha256": digest}}]}
        monkeypatch.setattr(
            bump,
            "_fetch",
            lambda requested: json.dumps(metadata).encode()
            if requested.endswith("/json") else data,
        )
        original = formula.read_bytes()
        assert bump.cmd_homebrew("0.17.0", tap=tap, dry_run=True) == 0
        assert formula.read_bytes() == original
        assert "---" in capsys.readouterr().out
        formula.unlink()
        assert bump.cmd_homebrew("0.17.0", tap=tap) == 1
        assert "pass --tap <dir>" in capsys.readouterr().err

    def test_end_to_end_preserves_crlf(self, tmp_path: Path, monkeypatch: object) -> None:
        tap, formula = self._setup_tap(tmp_path)
        original = formula.read_bytes().replace(b"\n", b"\r\n")
        formula.write_bytes(original)
        data = b"real sdist bytes"
        url = "https://example/mlspec-0.17.0.tar.gz"
        digest = hashlib.sha256(data).hexdigest()
        metadata = {"urls": [{"packagetype": "sdist", "url": url, "digests": {"sha256": digest}}]}
        monkeypatch.setattr(
            bump,
            "_fetch",
            lambda requested: json.dumps(metadata).encode()
            if requested.endswith("/json") else data,
        )
        expected = _bump_formula(original.decode(), url, digest).encode()

        assert bump.cmd_homebrew("0.17.0", tap=tap) == 0
        result = formula.read_bytes()
        assert result == expected
