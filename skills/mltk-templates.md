---
description: >
  mltk development templates — patterns for adding assertions, scanners,
  MCP tools, and CLI commands. Use when building new mltk features.
---

# mltk Development Templates

## 1. Add New Assertion

File: `src/mltk/{domain}/{module}.py`

```python
"""Module docstring."""
from __future__ import annotations

import numpy as np  # or whatever domain needs

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_something(
    data: np.ndarray,
    threshold: float = 0.05,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """One-line description.

    Args:
        data: What this parameter is.
        threshold: Pass/fail boundary.
        severity: CRITICAL raises, WARNING/INFO just reports.
    """
    # Domain-specific computation
    value = float(np.mean(data))
    passed = value < threshold

    return assert_true(
        condition=passed,
        name="domain.something",
        message=f"Value {value:.4f} {'<' if passed else '>='} {threshold}",
        severity=severity,
        value=value,
        threshold=threshold,
    )
```

**Wiring:**
- Add to `src/mltk/{domain}/__init__.py`: import + `__all__` entry
- Test: `tests/test_{domain}/test_{module}.py`

## 2. Add New Scanner

File: `src/mltk/scan/scanners/{name}.py`

```python
"""Scanner docstring."""
from __future__ import annotations

from mltk.scan.config import ScanContext
from mltk.scan.finding import FixSuggestion, ScanFinding
from mltk.scan.scanners.base import Scanner


class FooScanner(Scanner):
    name = "foo"
    category = "data_quality"  # or fairness, performance, security
    requires: frozenset[str] = frozenset({"df"})

    def scan(self, ctx: ScanContext) -> list[ScanFinding]:
        findings: list[ScanFinding] = []
        # Use ctx.df, ctx.model_fn, ctx.X, ctx.y, etc.
        # Build ScanFinding with assertion_fn + args for replay
        return findings

    def _gen_fix(self, finding: ScanFinding) -> list[FixSuggestion]:
        return [FixSuggestion(
            category="data",
            title="Fix title",
            description="What to do.",
            confidence=0.8,
            code_snippet="# example fix code",
        )]
```

**Wiring:**
- Add import to `src/mltk/scan/scanners/__init__.py`
- Test: `tests/test_scan/scanners/test_{name}.py`

## 3. Add New MCP Tool

File: `src/mltk/mcp/server.py` inside `_register_tools(mcp)`

```python
@mcp.tool()
def mltk_foo(param: str, option: str = "default") -> str:
    """One-line description for agent discovery.

    Args:
        param: What this is.
        option: Optional config.
    """
    try:
        # Lazy imports inside closure
        from mltk.some_module import SomeClass

        result = SomeClass().do_thing(param)
        return _ok(_with_hint("mltk_foo", {
            "result": result,
            "suggested_next_step": "Run mltk_report to export.",
        }))
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc))
```

**Wiring:**
- Add entry to `_WORKFLOW_HINTS` dict (position + next_tools)
- Update docstring tool count
- Add to `tests/test_mcp/conftest.py` docstring
- Update `tests/test_mcp/test_server_creation.py` EXPECTED_TOOLS
- Update `tests/test_mcp/test_server_integration.py` expected tool count

**Test pattern:** `tests/test_mcp/test_server_{name}.py`
```python
from __future__ import annotations
from tests.test_mcp._helpers import assert_error, assert_ok, call_tool

class TestMltkFoo:
    def test_success(self, registered_tools):
        # Patch at SOURCE module, not mcp.server
        with patch("mltk.some_module.SomeClass") as mock:
            mock.return_value.do_thing.return_value = "ok"
            result = call_tool("mltk_foo", param="test")
            assert_ok(result)

    def test_error(self, registered_tools):
        result = call_tool("mltk_foo", param="")
        assert_error(result)
```

## 4. Add New CLI Command

File: `src/mltk/cli/app.py` inside `main()`

```python
@app.command("foo-bar")
def foo_bar(
    path: str = typer.Argument(..., help="Input path"),
    verbose: bool = typer.Option(False, help="Show details"),
) -> None:
    """One-line description shown in --help."""
    from rich.console import Console
    console = Console()
    # Lazy import heavy deps
    from mltk.some_module import do_thing
    result = do_thing(path)
    console.print(result)
```

For sub-commands, use the appropriate sub-app:
- `contract_app.command()` for contract subcommands
- `docs_app.command()` for docs subcommands
- `registry_app.command()` for registry subcommands
- `notify_app.command()` for notify subcommands

**Test:** `tests/test_cli/test_{name}.py`

## 5. Wiring Checklist

After any addition:
- [ ] Export in `__init__.py` + `__all__`
- [ ] CHANGELOG.md entry under `[Unreleased]`
- [ ] BACKLOG.md count update
- [ ] `python scripts/generate_skill_index.py`
- [ ] `ruff check src/ tests/`
- [ ] `python -m pytest tests/ -x -q`

## Key Rules
- `from __future__ import annotations` = first code line (after docstring)
- MCP tools: lazy imports inside closures
- MCP tests: patch at source module, never at `mltk.mcp.server`
- All assertions return `TestResult` via `assert_true()`
- All assertions decorated with `@timed_assertion`
