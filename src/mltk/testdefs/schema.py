"""YAML test definition schema — parse test suite YAML files into structured dataclasses.

A test suite YAML file declares a data source and a list of assertions to run.
This module handles loading, parsing, and env-var resolution so the runner
receives clean, typed objects it can dispatch directly.

File format::

    data_source: path/to/data.csv          # or env:MY_DATA_PATH
    tests:
      - name: Schema check
        assertion: schema
        params:
          expected:
            id: int64
            label: int64

      - name: No nulls anywhere
        assertion: no_nulls

      - name: Score range
        assertion: range
        params:
          column: score
          min_val: 0.0
          max_val: 1.0

The ``data_source`` field supports two forms:

- A plain file path: ``data/features.csv``
- An env-var reference: ``env:MY_DATA_PATH`` — resolved from ``os.environ`` at
  load time, raising ``KeyError`` with a clear message if the variable is unset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestDef:
    """Specification for a single test inside a YAML suite.

    Args:
        name: Human-readable label shown in test output.
        assertion: Key identifying the assertion type (e.g., ``"schema"``,
            ``"no_nulls"``, ``"range"``).
        params: Extra parameters forwarded verbatim to the assertion function.

    Example:
        >>> td = TestDef(name="Score in range", assertion="range",
        ...              params={"column": "score", "min_val": 0.0, "max_val": 1.0})
    """

    name: str
    assertion: str
    params: dict = field(default_factory=dict)


@dataclass
class TestSuiteYaml:
    """Parsed YAML test suite ready for the runner.

    Args:
        data_source: Resolved path to the data file (CSV or Parquet). Any
            ``env:VAR`` references are expanded before this object is created.
        tests: Ordered list of :class:`TestDef` entries to execute.

    Example:
        >>> suite = load_test_suite("tests.yaml")
        >>> print(suite.data_source, len(suite.tests))
    """

    data_source: str
    tests: list[TestDef] = field(default_factory=list)


def _resolve_data_source(raw: str) -> str:
    """Resolve a data_source value, expanding ``env:VAR`` references.

    Args:
        raw: Raw data_source string from YAML (e.g., ``"env:MY_PATH"`` or
            ``"data/features.csv"``).

    Returns:
        Resolved path string.

    Raises:
        KeyError: If an ``env:VAR`` reference points to an unset variable.

    Example:
        >>> os.environ["MY_DATA"] = "/tmp/data.csv"
        >>> _resolve_data_source("env:MY_DATA")
        '/tmp/data.csv'
        >>> _resolve_data_source("data/features.csv")
        'data/features.csv'
    """
    if raw.startswith("env:"):
        var_name = raw[len("env:"):]
        value = os.environ.get(var_name)
        if value is None:
            raise KeyError(
                f"data_source references environment variable '{var_name}' "
                f"which is not set. Export it before running: "
                f"export {var_name}=/path/to/data.csv"
            )
        return value
    return raw


def load_test_suite(path: str | Path) -> TestSuiteYaml:
    """Load a YAML test suite definition from a file.

    Parses the YAML, resolves any ``env:VAR`` references in ``data_source``,
    and returns a typed :class:`TestSuiteYaml` ready to pass to
    :func:`~mltk.testdefs.runner.run_test_suite`.

    Args:
        path: Path to the ``.yaml`` (or ``.json``) test suite file.

    Returns:
        Parsed :class:`TestSuiteYaml` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is missing required fields (``data_source``,
            ``tests``) or if an assertion entry is missing its ``assertion`` key.
        KeyError: If an ``env:VAR`` reference in ``data_source`` is unset.

    Example:
        >>> suite = load_test_suite("tests/suite.yaml")
        >>> print(suite.data_source)
        data/features.csv
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test suite file not found: {p}")

    text = p.read_text(encoding="utf-8")

    try:
        import yaml

        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Test suite file must be a YAML mapping, got {type(raw).__name__}: {p}"
        )

    # Validate required top-level keys
    if "data_source" not in raw:
        raise ValueError(f"Test suite is missing required key 'data_source': {p}")
    if "tests" not in raw:
        raise ValueError(f"Test suite is missing required key 'tests': {p}")

    data_source = _resolve_data_source(str(raw["data_source"]))

    raw_tests = raw["tests"]
    if not isinstance(raw_tests, list):
        raise ValueError(
            f"'tests' must be a list, got {type(raw_tests).__name__}: {p}"
        )

    tests: list[TestDef] = []
    for i, entry in enumerate(raw_tests):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Test entry {i} must be a mapping, got {type(entry).__name__}: {p}"
            )
        if "assertion" not in entry:
            raise ValueError(
                f"Test entry {i} is missing required key 'assertion': {p}"
            )

        tests.append(
            TestDef(
                name=str(entry.get("name", f"test_{i}")),
                assertion=str(entry["assertion"]),
                params=entry.get("params") or {},
            )
        )

    return TestSuiteYaml(data_source=data_source, tests=tests)
