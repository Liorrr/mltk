"""Tests for mltk.integrations.dvc -- DVC data versioning assertions.

These tests use pytest's ``tmp_path`` fixture to create mock DVC project
structures on disk. No actual DVC installation is needed -- we create the
``.dvc`` files and ``.gitignore`` entries manually, exactly as DVC would.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>
"""

from __future__ import annotations

from pathlib import Path

from mltk.integrations.dvc import (
    assert_dvc_data_version,
    assert_dvc_file_tracked,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dvc_file(
    dvc_root: Path,
    file_path: str,
    md5: str = "d41d8cd98f00b204e9800998ecf8427e",
    size: int = 12345,
) -> Path:
    """Create a mock .dvc file with standard DVC YAML structure.

    Mimics the output of ``dvc add <file_path>``::

        outs:
        - md5: d41d8cd98f00b204e9800998ecf8427e
          size: 12345
          path: train.csv

    Args:
        dvc_root: Root directory of the mock DVC repo.
        file_path: Relative path to the data file (e.g., "data/train.csv").
        md5: MD5 hash to write into the .dvc file.
        size: File size to write into the .dvc file.

    Returns:
        Path to the created .dvc file.
    """
    dvc_file = dvc_root / f"{file_path}.dvc"
    dvc_file.parent.mkdir(parents=True, exist_ok=True)

    file_name = Path(file_path).name
    content = (
        f"outs:\n"
        f"- md5: {md5}\n"
        f"  size: {size}\n"
        f"  path: {file_name}\n"
    )
    dvc_file.write_text(content, encoding="utf-8")
    return dvc_file


def _create_gitignore(
    dvc_root: Path,
    directory: str,
    entries: list[str],
) -> Path:
    """Create a .gitignore file with the given entries.

    DVC adds entries like ``/train.csv`` to the .gitignore in the same
    directory as the tracked file.

    Args:
        dvc_root: Root directory of the mock DVC repo.
        directory: Subdirectory where .gitignore lives (e.g., "data").
        entries: Lines to write (e.g., ["/train.csv"]).

    Returns:
        Path to the created .gitignore file.
    """
    gitignore = dvc_root / directory / ".gitignore"
    gitignore.parent.mkdir(parents=True, exist_ok=True)
    gitignore.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return gitignore


# ---------------------------------------------------------------------------
# Tests: assert_dvc_file_tracked
# ---------------------------------------------------------------------------

class TestDvcFileTracked:
    """assert_dvc_file_tracked -- verify a file is properly DVC-tracked."""

    def test_file_tracked_passes(self, tmp_path: Path) -> None:
        # SCENARIO: A data file has both a .dvc file and a .gitignore entry.
        # WHY: This is the happy path -- ``dvc add`` was run correctly.
        #      The assertion must pass when everything is in order.
        # EXPECTED: TestResult.passed is True, name is correct.

        _create_dvc_file(tmp_path, "data/train.csv")
        _create_gitignore(tmp_path, "data", ["/train.csv"])

        result = assert_dvc_file_tracked("data/train.csv", dvc_root=str(tmp_path))

        assert result.passed is True
        assert result.name == "integrations.dvc.file_tracked"
        assert "properly tracked" in result.message

    def test_file_tracked_missing_dvc_file(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc metadata file does not exist (dvc add was never run).
        # WHY: This is the most common DVC mistake: adding a data file to the
        #      repo without running ``dvc add``. The assertion must catch it.
        # EXPECTED: TestResult.passed is False, message mentions missing .dvc file.

        # Only create .gitignore, no .dvc file
        _create_gitignore(tmp_path, "data", ["/train.csv"])

        result = assert_dvc_file_tracked(
            "data/train.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "Missing .dvc file" in result.message

    def test_file_tracked_missing_gitignore_entry(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc file exists but the file is not in .gitignore.
        # WHY: If the data file is not gitignored, it might accidentally get
        #      committed to git (bloating the repo with large binary files).
        #      This can happen if someone manually creates a .dvc file without
        #      using ``dvc add``.
        # EXPECTED: TestResult.passed is False, message mentions .gitignore.

        _create_dvc_file(tmp_path, "data/train.csv")
        # Create .gitignore but without the data file entry
        _create_gitignore(tmp_path, "data", ["/other_file.csv"])

        result = assert_dvc_file_tracked(
            "data/train.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "not found in" in result.message

    def test_file_tracked_no_gitignore_at_all(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc file exists but there is no .gitignore file at all.
        # WHY: A missing .gitignore is an even stronger signal that ``dvc add``
        #      was not used (it always creates/updates .gitignore). The assertion
        #      must fail and mention the missing gitignore entry.
        # EXPECTED: TestResult.passed is False.

        _create_dvc_file(tmp_path, "data/train.csv")
        # No .gitignore created

        result = assert_dvc_file_tracked(
            "data/train.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "not found in" in result.message


# ---------------------------------------------------------------------------
# Tests: assert_dvc_data_version
# ---------------------------------------------------------------------------

class TestDvcDataVersion:
    """assert_dvc_data_version -- verify DVC file hash integrity."""

    def test_hash_matches(self, tmp_path: Path) -> None:
        # SCENARIO: The stored MD5 in the .dvc file matches expected_md5.
        # WHY: This is the happy path for version-pinned data assertions.
        #      When expected hash matches stored hash, the data is exactly
        #      the version we expect -- reproducibility is guaranteed.
        # EXPECTED: TestResult.passed is True, message confirms match.

        known_hash = "abc123def456789"
        _create_dvc_file(tmp_path, "data/train.csv", md5=known_hash)

        result = assert_dvc_data_version(
            "data/train.csv",
            expected_md5=known_hash,
            dvc_root=str(tmp_path),
        )

        assert result.passed is True
        assert "matches" in result.message
        assert result.details["stored_md5"] == known_hash

    def test_hash_mismatch(self, tmp_path: Path) -> None:
        # SCENARIO: The stored MD5 does not match expected_md5.
        # WHY: A hash mismatch means the data changed but the .dvc file
        #      was not updated (or vice versa). This is a data version drift
        #      that causes unreproducible training results.
        # EXPECTED: TestResult.passed is False, message shows both hashes.

        _create_dvc_file(tmp_path, "data/train.csv", md5="stored_hash_111")

        result = assert_dvc_data_version(
            "data/train.csv",
            expected_md5="expected_hash_222",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "mismatch" in result.message
        assert result.details["stored_md5"] == "stored_hash_111"
        assert result.details["expected_md5"] == "expected_hash_222"

    def test_valid_hash_no_expected(self, tmp_path: Path) -> None:
        # SCENARIO: No expected_md5 is provided; just check that the .dvc
        #           file exists and has a valid hash.
        # WHY: Sometimes you want to verify "is this file set up correctly
        #      with DVC?" without pinning to a specific version. This is
        #      useful in setup/validation scripts that run before training.
        # EXPECTED: TestResult.passed is True as long as .dvc has a hash.

        _create_dvc_file(tmp_path, "data/train.csv", md5="any_valid_hash")

        result = assert_dvc_data_version(
            "data/train.csv",
            dvc_root=str(tmp_path),
        )

        assert result.passed is True
        assert "valid hash" in result.message
        assert result.details["stored_md5"] == "any_valid_hash"

    def test_dvc_file_valid_yaml_structure(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc file has the standard YAML structure with outs,
        #           md5, size, and path fields.
        # WHY: DVC files follow a specific YAML schema. The parser must
        #      correctly extract the md5 from the ``outs`` list. If parsing
        #      fails on valid DVC YAML, real repos will break.
        # EXPECTED: Parser extracts the correct md5 from standard structure.

        dvc_content = (
            "outs:\n"
            "- md5: e99a18c428cb38d5f260853678922e03\n"
            "  size: 98765\n"
            "  path: embeddings.npy\n"
        )
        dvc_file = tmp_path / "data" / "embeddings.npy.dvc"
        dvc_file.parent.mkdir(parents=True, exist_ok=True)
        dvc_file.write_text(dvc_content, encoding="utf-8")

        result = assert_dvc_data_version(
            "data/embeddings.npy",
            expected_md5="e99a18c428cb38d5f260853678922e03",
            dvc_root=str(tmp_path),
        )

        assert result.passed is True
        assert result.details["stored_md5"] == "e99a18c428cb38d5f260853678922e03"

    def test_missing_dvc_file(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc file does not exist at all.
        # WHY: If there is no .dvc file, the data file was never added to
        #      DVC. This is distinct from a hash mismatch -- it means DVC
        #      tracking was never set up for this file.
        # EXPECTED: TestResult.passed is False, message says "not found".

        result = assert_dvc_data_version(
            "data/nonexistent.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "not found" in result.message

    def test_missing_dvc_root(self, tmp_path: Path) -> None:
        # SCENARIO: The dvc_root directory itself does not exist.
        # WHY: In CI environments, the working directory might be wrong or
        #      the repo might not be checked out yet. The assertion should
        #      fail gracefully with a clear message, not crash with an
        #      unhandled OSError.
        # EXPECTED: TestResult.passed is False (not an unhandled exception).

        nonexistent_root = str(tmp_path / "does_not_exist")

        result = assert_dvc_data_version(
            "data/train.csv",
            dvc_root=nonexistent_root,
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        assert "not found" in result.message

    def test_corrupted_dvc_file(self, tmp_path: Path) -> None:
        # SCENARIO: The .dvc file exists but contains invalid/corrupted content.
        # WHY: Merge conflicts or manual edits can corrupt .dvc files. The
        #      parser must handle this gracefully instead of raising an
        #      unhandled exception that crashes the test suite.
        # EXPECTED: TestResult.passed is False, message mentions parse failure.

        dvc_file = tmp_path / "data" / "train.csv.dvc"
        dvc_file.parent.mkdir(parents=True, exist_ok=True)
        dvc_file.write_text("<<<< HEAD\ngarbage\n>>>> branch\n", encoding="utf-8")

        result = assert_dvc_data_version(
            "data/train.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )

        assert result.passed is False
        # Should fail due to parse error or missing hash
        assert "parse" in result.message.lower() or "no md5" in result.message.lower()


# -------------------------------------------------------------------
# Parametrized & edge-case tests (hardening)
# -------------------------------------------------------------------


class TestDvcMultipleFilesInDir:
    """Multiple .dvc files in the same directory."""

    def test_two_dvc_files_same_dir(
        self, tmp_path: Path
    ) -> None:
        """Each file tracked independently in same dir."""
        _create_dvc_file(
            tmp_path, "data/train.csv", md5="aaa"
        )
        _create_dvc_file(
            tmp_path, "data/val.csv", md5="bbb"
        )
        _create_gitignore(
            tmp_path,
            "data",
            ["/train.csv", "/val.csv"],
        )

        r1 = assert_dvc_file_tracked(
            "data/train.csv", dvc_root=str(tmp_path)
        )
        r2 = assert_dvc_file_tracked(
            "data/val.csv", dvc_root=str(tmp_path)
        )
        assert r1.passed is True
        assert r2.passed is True


class TestDvcMultipleOutputs:
    """.dvc file with multiple outs entries."""

    def test_multi_output_dvc_file(
        self, tmp_path: Path
    ) -> None:
        """Parser extracts first md5 from multi-out file."""
        content = (
            "outs:\n"
            "- md5: first111\n"
            "  size: 100\n"
            "  path: file_a.csv\n"
            "- md5: second222\n"
            "  size: 200\n"
            "  path: file_b.csv\n"
        )
        dvc_file = tmp_path / "data" / "file_a.csv.dvc"
        dvc_file.parent.mkdir(parents=True, exist_ok=True)
        dvc_file.write_text(content, encoding="utf-8")

        r = assert_dvc_data_version(
            "data/file_a.csv",
            expected_md5="first111",
            dvc_root=str(tmp_path),
        )
        assert r.passed is True
        assert r.details["stored_md5"] == "first111"


class TestDvcRelativeVsAbsolutePaths:
    """Relative vs absolute dvc_root paths."""

    def test_absolute_dvc_root(
        self, tmp_path: Path
    ) -> None:
        """Absolute dvc_root resolves correctly."""
        _create_dvc_file(
            tmp_path, "models/bert.bin", md5="abs123"
        )
        abs_root = str(tmp_path.resolve())
        r = assert_dvc_data_version(
            "models/bert.bin", dvc_root=abs_root
        )
        assert r.passed is True
        assert r.details["stored_md5"] == "abs123"


class TestDvcGitignoreMultipleEntries:
    """.gitignore with multiple entries."""

    def test_gitignore_multi_entries(
        self, tmp_path: Path
    ) -> None:
        """Target found among many gitignore lines."""
        _create_dvc_file(tmp_path, "data/test.csv")
        _create_gitignore(
            tmp_path,
            "data",
            ["/other.csv", "/test.csv", "/extra.bin"],
        )
        r = assert_dvc_file_tracked(
            "data/test.csv", dvc_root=str(tmp_path)
        )
        assert r.passed is True


class TestDvcMissingYamlKey:
    """Missing YAML key in .dvc file."""

    def test_dvc_file_missing_md5_key(
        self, tmp_path: Path
    ) -> None:
        """DVC file with outs but no md5 field."""
        content = (
            "outs:\n"
            "- size: 999\n"
            "  path: broken.csv\n"
        )
        dvc_file = tmp_path / "data" / "broken.csv.dvc"
        dvc_file.parent.mkdir(parents=True, exist_ok=True)
        dvc_file.write_text(content, encoding="utf-8")

        r = assert_dvc_data_version(
            "data/broken.csv",
            dvc_root=str(tmp_path),
            severity="warning",  # type: ignore[arg-type]
        )
        assert r.passed is False
