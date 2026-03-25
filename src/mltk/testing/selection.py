"""Smart test selection — only re-run tests affected by changes."""
from __future__ import annotations

import ast
from pathlib import Path


def _extract_imports(filepath: Path) -> set[str]:
    """Return the set of dotted module names imported by *filepath*.

    Uses :mod:`ast` to parse the source.  Both ``import foo.bar`` and
    ``from foo.bar import baz`` forms are captured.  Returns an empty set
    if the file cannot be parsed.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, OSError):
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                # Also record fully-qualified sub-module candidates.
                # e.g. "from mypkg import schema" -> "mypkg.schema"
                # This lets us match source file mypkg/schema.py correctly.
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return names


def _module_name_from_path(filepath: Path, base_dir: Path) -> str:
    """Convert a filesystem path to a dotted module name relative to *base_dir*.

    Example::

        base_dir = Path("src")
        filepath = Path("src/mltk/data/drift.py")
        => "mltk.data.drift"
    """
    try:
        rel = filepath.relative_to(base_dir)
    except ValueError:
        rel = filepath

    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Drop __init__
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_test_map(
    test_dir: str | Path,
    src_dir: str | Path,
) -> dict[str, list[str]]:
    """Build a dependency map from source files to the test files that import them.

    Parses every ``test_*.py`` file under *test_dir* with :mod:`ast` and
    records which source modules it imports.

    Args:
        test_dir: Root directory containing test files.
        src_dir: Root directory of the source tree (used to resolve module names).

    Returns:
        Dict mapping source-file path strings to lists of test-file path strings.
        Example::

            {
                "src/mltk/data/drift.py": ["tests/test_data/test_drift.py"],
            }

    Example:
        >>> mapping = build_test_map("tests/", "src/")
        >>> mapping.get("src/mltk/data/drift.py", [])
        ['tests/test_data/test_drift.py']
    """
    test_root = Path(test_dir)
    src_root = Path(src_dir)

    # Build reverse index: module_name -> source_file_path
    module_to_src: dict[str, str] = {}
    for src_file in src_root.rglob("*.py"):
        mod_name = _module_name_from_path(src_file, src_root)
        module_to_src[mod_name] = str(src_file)

    result: dict[str, list[str]] = {}

    for test_file in test_root.rglob("test_*.py"):
        imported_modules = _extract_imports(test_file)
        for mod in imported_modules:
            # Match when the imported module *is* or *is a sub-module of* the
            # source module.  "mltk.data.drift" matches source "mltk.data.drift"
            # (exact) or source "mltk.data" (parent package).  We do NOT match
            # the reverse — importing "mypkg" must not pull in every sub-module.
            for mod_name, src_path in module_to_src.items():
                is_match = mod == mod_name or mod.startswith(mod_name + ".")
                if is_match:
                    if src_path not in result:
                        result[src_path] = []
                    test_str = str(test_file)
                    if test_str not in result[src_path]:
                        result[src_path].append(test_str)

    return result


def select_affected_tests(
    changed_files: list[str],
    test_map: dict[str, list[str]],
) -> list[str]:
    """Return the test files that need re-running given a list of changed source files.

    Args:
        changed_files: List of source file paths that were modified.
        test_map: Dependency map produced by :func:`build_test_map`.

    Returns:
        Deduplicated list of test file path strings that cover the changed files.
        Returns an empty list when no tests are affected.

    Example:
        >>> test_map = {"src/mltk/data/drift.py": ["tests/test_data/test_drift.py"]}
        >>> select_affected_tests(["src/mltk/data/drift.py"], test_map)
        ['tests/test_data/test_drift.py']
    """
    affected: list[str] = []
    seen: set[str] = set()

    for changed in changed_files:
        # Normalise separators for cross-platform matching
        changed_norm = str(Path(changed))
        for src_path, test_files in test_map.items():
            src_norm = str(Path(src_path))
            if changed_norm == src_norm:
                for tf in test_files:
                    if tf not in seen:
                        seen.add(tf)
                        affected.append(tf)

    return affected
