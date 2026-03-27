"""DVC (Data Version Control) assertions -- verify data files are tracked and versioned.

WHY DVC assertions matter for ML testing:
DVC is the standard tool for versioning large files (datasets, model weights,
embeddings) alongside git. Git tracks code; DVC tracks data. Together they
give you reproducible experiments.

But DVC has a common failure mode: someone adds a new dataset file, trains a
model with it, commits the code... and forgets to ``dvc add`` the data file.
The code works on their machine (the file exists locally), but when a teammate
clones the repo and runs ``dvc pull``, the file is missing. The training
pipeline crashes with a cryptic FileNotFoundError.

These assertions catch that problem BEFORE it reaches CI or a teammate:

1. ``assert_dvc_file_tracked`` -- is this data file known to DVC?
2. ``assert_dvc_data_version`` -- does the DVC metadata match the expected hash?

How DVC tracking works (for context):
When you run ``dvc add data/train.csv``, DVC:
  1. Computes the file's MD5 hash
  2. Creates ``data/train.csv.dvc`` (a YAML file containing the hash)
  3. Adds ``data/train.csv`` to ``.gitignore`` (so git ignores the large file)
  4. Copies the file into ``.dvc/cache/`` (keyed by hash)

You then ``git add data/train.csv.dvc .gitignore`` and commit.
The ``.dvc`` file is tiny (a few lines of YAML) and lives in git.
The actual data lives in DVC remote storage (S3, GCS, etc.).

These assertions verify steps 2 and 3 happened correctly.
"""

from __future__ import annotations

from pathlib import Path

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _parse_dvc_yaml(dvc_path: Path) -> dict:
    """Parse a .dvc file and return its contents as a dict.

    DVC files are simple YAML with a predictable structure::

        outs:
        - md5: d41d8cd98f00b204e9800998ecf8427e
          size: 12345
          path: train.csv

    We parse this with PyYAML if available, falling back to a minimal
    manual parser for the common case (single-output .dvc files).

    WHY not always require PyYAML:
    mltk aims to have minimal required dependencies. PyYAML is common but
    not universal. The manual fallback handles the 95% case (single ``md5``
    line) so that users can run DVC assertions without installing PyYAML.

    Args:
        dvc_path: Path to the ``.dvc`` file.

    Returns:
        Parsed dict. Always contains an ``"outs"`` key with a list of
        output dicts if the file is valid DVC YAML.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    content = dvc_path.read_text(encoding="utf-8")

    # Try PyYAML first (preferred -- handles all edge cases)
    try:
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass

    # Fallback: minimal line-based parser for standard .dvc files
    result: dict = {"outs": []}
    current_out: dict = {}

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- md5:"):
            if current_out:
                result["outs"].append(current_out)
            current_out = {"md5": stripped.split(":", 1)[1].strip()}
        elif stripped.startswith("size:") and current_out:
            try:
                current_out["size"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("path:") and current_out:
            current_out["path"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("md5:") and not stripped.startswith("- md5:"):
            # Top-level md5 (file-level hash in some DVC versions)
            result["md5"] = stripped.split(":", 1)[1].strip()

    if current_out:
        result["outs"].append(current_out)

    if not result["outs"] and "md5" not in result:
        raise ValueError(f"Could not parse DVC file: {dvc_path}")

    return result


def _get_md5_from_dvc(dvc_data: dict) -> str | None:
    """Extract the first MD5 hash from parsed .dvc data.

    Handles both formats:
    - ``outs[0]["md5"]`` (standard single-output .dvc file)
    - ``md5`` at root level (older DVC versions)

    Args:
        dvc_data: Parsed .dvc file contents.

    Returns:
        MD5 hash string, or None if not found.
    """
    outs = dvc_data.get("outs", [])
    if outs and isinstance(outs, list) and isinstance(outs[0], dict):
        return outs[0].get("md5")
    return dvc_data.get("md5")


@timed_assertion
def assert_dvc_file_tracked(
    file_path: str,
    dvc_root: str = ".",
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a data file is properly tracked by DVC.

    WHAT this checks:
    A file is "tracked by DVC" when ALL of these are true:
    1. A ``{file_path}.dvc`` metadata file exists (created by ``dvc add``)
    2. The file is listed in ``.gitignore`` (so git does not track the raw data)

    WHY this matters:
    If a data file is NOT tracked by DVC, it means:
    - It will not be available via ``dvc pull`` on other machines
    - It might accidentally get committed to git (bloating the repo)
    - Reproducibility is broken: someone else cannot recreate your experiment

    This is the #1 DVC mistake in ML teams. Catch it in CI, not in production.

    Args:
        file_path: Path to the data file (relative to dvc_root or absolute).
            Example: ``"data/train.csv"`` or ``"models/bert-base.bin"``.
        dvc_root: Root directory of the DVC repo. Defaults to current directory.
            The ``.dvc`` file is expected at ``{dvc_root}/{file_path}.dvc``.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with pass/fail and details about what was checked.

    Example::

        # In a pytest test or CI script:
        result = assert_dvc_file_tracked("data/train.csv", dvc_root=".")
        assert result.passed

    See Also:
        :func:`assert_dvc_data_version` -- verify the hash matches expectations.
    """
    root = Path(dvc_root)
    target = Path(file_path)
    dvc_file = root / f"{file_path}.dvc"

    checks: dict[str, bool] = {}
    errors: list[str] = []

    # Check 1: .dvc metadata file exists
    dvc_exists = dvc_file.exists()
    checks["dvc_file_exists"] = dvc_exists
    if not dvc_exists:
        errors.append(f"Missing .dvc file: {dvc_file}")

    # Check 2: file is in .gitignore
    gitignore_path = root / target.parent / ".gitignore"
    gitignore_has_entry = False

    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text(encoding="utf-8")
        file_name = target.name
        # DVC adds the bare filename to .gitignore in the same directory
        for line in gitignore_content.splitlines():
            stripped = line.strip()
            if stripped == f"/{file_name}" or stripped == file_name:
                gitignore_has_entry = True
                break

    checks["in_gitignore"] = gitignore_has_entry
    if not gitignore_has_entry:
        errors.append(
            f"File '{target.name}' not found in {gitignore_path} "
            f"(DVC should have added it)"
        )

    passed = dvc_exists and gitignore_has_entry
    message = (
        f"File '{file_path}' is properly tracked by DVC"
        if passed
        else "; ".join(errors)
    )

    return assert_true(
        passed,
        name="integrations.dvc.file_tracked",
        message=message,
        severity=severity,
        file_path=str(file_path),
        dvc_file=str(dvc_file),
        checks=checks,
    )


@timed_assertion
def assert_dvc_data_version(
    file_path: str,
    expected_md5: str | None = None,
    dvc_root: str = ".",
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a DVC-tracked file has the expected content hash.

    WHAT this checks:
    1. The ``{file_path}.dvc`` metadata file exists and is valid YAML
    2. The ``.dvc`` file contains an ``md5`` hash (proof of ``dvc add``)
    3. If ``expected_md5`` is provided, the stored hash must match exactly

    WHY this matters:
    The MD5 hash in the ``.dvc`` file is DVC's record of the data content.
    A mismatch means one of these happened:
    - Someone modified the data locally but forgot ``dvc push`` (data drift)
    - The ``.dvc`` file was manually edited (metadata corruption)
    - A merge conflict corrupted the hash (silent data version mismatch)

    Any of these causes training on different data than expected, leading to
    unreproducible results. This assertion catches the mismatch early.

    When ``expected_md5`` is None, the assertion just verifies the ``.dvc``
    file exists and has a structurally valid hash -- useful for "is this file
    set up correctly?" checks without pinning to a specific version.

    Args:
        file_path: Path to the data file (relative to dvc_root or absolute).
        expected_md5: Expected MD5 hash string. When provided, the stored
            hash must match exactly. When None, any valid hash passes.
        dvc_root: Root directory of the DVC repo. Defaults to current directory.
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with details including the stored and expected hashes.

    Example::

        # Pin to a specific data version:
        result = assert_dvc_data_version(
            "data/train.csv",
            expected_md5="d41d8cd98f00b204e9800998ecf8427e",
        )
        assert result.passed

        # Just verify the .dvc file is valid (no specific hash):
        result = assert_dvc_data_version("data/train.csv")
        assert result.passed

    See Also:
        :func:`assert_dvc_file_tracked` -- verify the file is tracked at all.
    """
    root = Path(dvc_root)
    dvc_file = root / f"{file_path}.dvc"

    # Check 1: .dvc file exists
    if not dvc_file.exists():
        return assert_true(
            False,
            name="integrations.dvc.data_version",
            message=f"DVC metadata file not found: {dvc_file}",
            severity=severity,
            file_path=str(file_path),
            dvc_file=str(dvc_file),
            expected_md5=expected_md5,
            stored_md5=None,
        )

    # Check 2: parse the .dvc file
    try:
        dvc_data = _parse_dvc_yaml(dvc_file)
    except (ValueError, OSError) as exc:
        return assert_true(
            False,
            name="integrations.dvc.data_version",
            message=f"Failed to parse DVC file {dvc_file}: {exc}",
            severity=severity,
            file_path=str(file_path),
            dvc_file=str(dvc_file),
            expected_md5=expected_md5,
            stored_md5=None,
        )

    # Check 3: extract MD5 hash
    stored_md5 = _get_md5_from_dvc(dvc_data)

    if stored_md5 is None:
        return assert_true(
            False,
            name="integrations.dvc.data_version",
            message=f"No MD5 hash found in {dvc_file}",
            severity=severity,
            file_path=str(file_path),
            dvc_file=str(dvc_file),
            expected_md5=expected_md5,
            stored_md5=None,
        )

    # Check 4: if expected_md5 provided, compare
    if expected_md5 is not None:
        matches = stored_md5 == expected_md5
        message = (
            f"DVC hash matches for '{file_path}': {stored_md5}"
            if matches
            else (
                f"DVC hash mismatch for '{file_path}': "
                f"stored={stored_md5}, expected={expected_md5}"
            )
        )
        return assert_true(
            matches,
            name="integrations.dvc.data_version",
            message=message,
            severity=severity,
            file_path=str(file_path),
            dvc_file=str(dvc_file),
            expected_md5=expected_md5,
            stored_md5=stored_md5,
        )

    # No expected_md5 -- just verify a valid hash exists
    return assert_true(
        True,
        name="integrations.dvc.data_version",
        message=f"DVC file '{file_path}' has valid hash: {stored_md5}",
        severity=severity,
        file_path=str(file_path),
        dvc_file=str(dvc_file),
        expected_md5=None,
        stored_md5=stored_md5,
    )
