"""Test impact analysis — run only the tests that matter for your change.

Running an entire test suite on every commit is wasteful. If you modified
``data/drift.py``, you do not need to re-run ``test_fairness.py`` or
``test_golden.py``.  Test impact analysis builds a dependency graph from
Python imports and returns only the test files whose execution paths
include the changed source code.

This is complementary to :mod:`mltk.testing.selection` which provides the
lower-level primitives (``build_test_map`` / ``select_affected_tests``).
The functions here add **transitive dependency resolution** and a
**coverage assertion** that verifies your CI pipeline actually ran
everything it should have.

Typical usage in CI::

    changed = ["src/mltk/data/drift.py"]
    impacted = analyze_impact(changed, project_root=".", test_dir="tests")
    # -> ["tests/test_data/test_drift.py", "tests/test_monitor/test_concept_drift.py"]

    # After the test run, verify nothing was skipped:
    assert_impact_coverage(changed, executed_tests=impacted)

"""
from __future__ import annotations

import ast
from pathlib import Path

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _extract_imports(filepath: Path) -> set[str]:
    """Parse a Python file and return all imported module names.

    Uses :mod:`ast` to statically analyse ``import X`` and
    ``from X import Y`` statements.  Returns an empty set if the file
    cannot be parsed (syntax errors, encoding issues, etc.).

    Both the base module and the fully-qualified sub-import are recorded::

        from mltk.data import drift
        -> {"mltk.data", "mltk.data.drift"}

    Args:
        filepath: Path to a ``.py`` file.

    Returns:
        Set of dotted module name strings found in import statements.
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
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return names


def _module_name_from_path(filepath: Path, base_dir: Path) -> str:
    """Convert a filesystem path to a dotted Python module name.

    Strips the *base_dir* prefix and the ``.py`` extension, then joins
    the remaining directory parts with dots.  ``__init__`` segments are
    removed so that package directories become their parent module name.

    Example::

        _module_name_from_path(
            Path("src/mltk/data/drift.py"),
            Path("src"),
        )
        # -> "mltk.data.drift"
    """
    try:
        rel = filepath.relative_to(base_dir)
    except ValueError:
        rel = filepath

    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_source_module_index(src_root: Path) -> dict[str, str]:
    """Map dotted module names to their source file paths.

    Scans every ``.py`` file under *src_root* and creates a lookup table::

        {"mltk.data.drift": "src/mltk/data/drift.py", ...}

    Args:
        src_root: Root of the source tree (e.g. ``Path("src")``).

    Returns:
        Dict from module name to source path string.
    """
    index: dict[str, str] = {}
    for src_file in src_root.rglob("*.py"):
        mod_name = _module_name_from_path(src_file, src_root)
        if mod_name:
            index[mod_name] = str(src_file)
    return index


def _build_test_dependency_graph(
    test_root: Path,
    module_to_src: dict[str, str],
) -> dict[str, set[str]]:
    """Build a mapping from each source file to the test files that depend on it.

    For each ``test_*.py`` file, parses its imports and matches them to
    known source modules.  A test *depends on* a source file if it imports
    that module or any child of it.

    Args:
        test_root: Directory containing test files.
        module_to_src: Module-name-to-path index from :func:`_build_source_module_index`.

    Returns:
        Dict mapping source path strings to sets of test path strings.
    """
    graph: dict[str, set[str]] = {}

    for test_file in test_root.rglob("test_*.py"):
        imported = _extract_imports(test_file)
        for imp in imported:
            for mod_name, src_path in module_to_src.items():
                if imp == mod_name or imp.startswith(mod_name + "."):
                    graph.setdefault(src_path, set()).add(str(test_file))

    return graph


def _resolve_transitive_deps(
    src_root: Path,
    module_to_src: dict[str, str],
) -> dict[str, set[str]]:
    """Build a transitive dependency map between source files.

    If ``drift.py`` imports from ``schema.py``, and ``monitor.py`` imports
    from ``drift.py``, then a change to ``schema.py`` transitively
    affects both ``drift.py`` and ``monitor.py``.

    This function performs a fixed-point expansion: for each source file,
    it follows the import chain until no new dependencies are discovered.

    Args:
        src_root: Root of the source tree.
        module_to_src: Module-name-to-path index.

    Returns:
        Dict mapping each source path to the set of other source paths
        that transitively depend on it.
    """
    # Reverse the index for quick lookup: src_path -> module_name
    src_to_module: dict[str, str] = {v: k for k, v in module_to_src.items()}

    # Direct forward deps: src_path -> set of src_paths it imports
    forward: dict[str, set[str]] = {}
    for src_file in src_root.rglob("*.py"):
        src_path = str(src_file)
        if src_path not in src_to_module:
            continue
        imported = _extract_imports(src_file)
        deps: set[str] = set()
        for imp in imported:
            for mod_name, mpath in module_to_src.items():
                if (imp == mod_name or imp.startswith(mod_name + ".")) and mpath != src_path:
                    deps.add(mpath)
        forward[src_path] = deps

    # Invert to: src_path -> set of src_paths that import from it (direct)
    reverse: dict[str, set[str]] = {}
    for src_path, deps in forward.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(src_path)

    # Transitive closure via BFS on the reverse graph
    transitive: dict[str, set[str]] = {}
    for origin in reverse:
        visited: set[str] = set()
        frontier = list(reverse.get(origin, set()))
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            frontier.extend(reverse.get(current, set()) - visited)
        transitive[origin] = visited

    return transitive


def analyze_impact(
    changed_files: list[str],
    project_root: str = ".",
    test_dir: str = "tests",
) -> list[str]:
    """Determine which test files should run based on changed source files.

    **Why test impact analysis matters:**

    Running ALL 1500+ tests on every code change is slow and wasteful.
    If you changed ``data/drift.py``, you only need to run
    ``test_data/test_drift.py`` and any test that imports from
    ``data/drift``.  This function builds a dependency graph from Python
    imports and returns only the impacted test files.

    **Algorithm:**

    1. Scan the ``src/`` tree to build a module-name -> file-path index.
    2. For each ``test_*.py`` file, parse its imports (``ast.parse`` ->
       ``ImportFrom`` nodes) and map them to source files.
    3. Build a transitive dependency graph between source files (if A
       imports B and B imports C, then changing C impacts tests of A too).
    4. For each changed file, collect all test files that import from it
       directly **or** from any source file that transitively depends on it.

    Args:
        changed_files: List of changed source file paths (relative to
            *project_root* or absolute).
        project_root: Root of the project (default ``"."``).
        test_dir: Subdirectory containing tests (default ``"tests"``).

    Returns:
        Sorted, deduplicated list of test file paths that should be
        executed.  Empty list if no tests are impacted.

    Example:
        >>> impacted = analyze_impact(
        ...     ["src/mltk/data/drift.py"],
        ...     project_root=".",
        ...     test_dir="tests",
        ... )
        >>> # Returns tests that import from data.drift, directly or transitively
    """
    root = Path(project_root)
    test_root = root / test_dir

    if not test_root.exists():
        return []

    # Discover the source directory (prefer "src/" if it exists)
    src_root = root / "src" if (root / "src").exists() else root

    # Step 1: Build module index
    module_to_src = _build_source_module_index(src_root)
    if not module_to_src:
        return []

    # Step 2: Build direct test-dependency graph
    test_deps = _build_test_dependency_graph(test_root, module_to_src)

    # Step 3: Build transitive source-to-source dependency graph
    transitive_deps = _resolve_transitive_deps(src_root, module_to_src)

    # Step 4: For each changed file, find all impacted test files
    impacted: set[str] = set()

    for changed in changed_files:
        changed_norm = str(Path(changed))

        # Direct test dependents
        for src_path, test_files in test_deps.items():
            if str(Path(src_path)) == changed_norm:
                impacted.update(test_files)

        # Transitive: source files that depend on the changed file
        for src_path, dependents in transitive_deps.items():
            if str(Path(src_path)) == changed_norm:
                # Every source file that transitively depends on the changed
                # file could have test files associated with it
                for dep_src in dependents:
                    for td_src, td_tests in test_deps.items():
                        if str(Path(td_src)) == str(Path(dep_src)):
                            impacted.update(td_tests)

    return sorted(impacted)


@timed_assertion
def assert_impact_coverage(
    changed_files: list[str],
    executed_tests: list[str],
    project_root: str = ".",
    test_dir: str = "tests",
) -> TestResult:
    """Assert that all impacted tests were actually executed.

    **Why this assertion exists:**

    CI pipelines sometimes skip tests due to misconfigured path filters,
    parallelism bugs, or shard imbalances.  This catches the situation
    where you changed ``drift.py`` but ``test_drift.py`` was never
    executed -- a silent coverage gap that defeats the purpose of having
    tests at all.

    Internally calls :func:`analyze_impact` to determine the required
    test set, then compares it against *executed_tests*.

    Args:
        changed_files: Source files that were modified in this change.
        executed_tests: Test files that were actually run (from CI logs
            or pytest collection).
        project_root: Project root directory.
        test_dir: Test subdirectory name.

    Returns:
        :class:`~mltk.core.result.TestResult` -- passes when every
        impacted test file appears in *executed_tests*.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: When impacted
            tests were not executed (CRITICAL severity).

    Example:
        >>> result = assert_impact_coverage(
        ...     changed_files=["src/mltk/data/drift.py"],
        ...     executed_tests=["tests/test_data/test_drift.py"],
        ... )
    """
    required = analyze_impact(
        changed_files,
        project_root=project_root,
        test_dir=test_dir,
    )

    executed_norm = {str(Path(t)) for t in executed_tests}
    missing = [t for t in required if str(Path(t)) not in executed_norm]

    passed = len(missing) == 0
    message = (
        f"All {len(required)} impacted tests were executed"
        if passed
        else (
            f"{len(missing)} impacted test(s) were NOT executed: "
            + ", ".join(missing)
        )
    )

    return assert_true(
        passed,
        name="testing.impact.coverage",
        message=message,
        severity=Severity.CRITICAL,
        required_tests=required,
        executed_tests=sorted(executed_norm),
        missing_tests=missing,
        changed_files=changed_files,
    )
