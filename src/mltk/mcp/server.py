"""mltk MCP server -- expose ML testing tools to AI agents.

Tools (13): mltk_scan, mltk_test, mltk_list, mltk_eval,
mltk_dataset, mltk_report, mltk_suggest, mltk_experiment,
mltk_workflow, mltk_create_pr, mltk_create_issue,
mltk_container_scan, mltk_import.

Usage: ``python -m mltk.mcp``
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mltk import __version__

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

__all__ = ["create_server", "run_server"]

_SCANNER_NAMES = (
    "data, drift, bias, overfit, calibration, "
    "robustness, leakage, slice"
)


def _log(msg: str) -> None:
    """Write a diagnostic message to stderr."""
    print(f"[mltk-mcp] {msg}", file=sys.stderr)


def _ok(payload: dict[str, Any]) -> str:
    """Wrap a successful response with status=ok."""
    payload["status"] = "ok"
    return json.dumps(payload, indent=2, default=str)


def _error(
    msg: str, *,
    recoverable: bool = True,
    suggested_action: str = "",
    fallback_parameters: dict[str, Any] | None = None,
) -> str:
    """Build a JSON error response."""
    payload: dict[str, Any] = {
        "status": "error",
        "error": msg,
        "recoverable": recoverable,
        "suggested_action": (
            suggested_action
            or "Check the error message and retry."
        ),
    }
    if fallback_parameters is not None:
        payload["fallback_parameters"] = fallback_parameters
    return json.dumps(payload, indent=2)


def _resolve_eval_model(
    model_mode: str = "passthrough",
    model_ref: str = "",
) -> tuple[Any, str, str]:
    """Resolve ``mltk_eval`` model callable and honesty labels.

    Supported modes:
    - ``passthrough`` / ``identity`` (default): echo the prompt.
    - ``module``: import ``model_ref`` as ``package.module:callable``.

    ``module`` mode is gated by the ``MLTK_MCP_MODEL_MODULES`` environment
    variable — a comma-separated list of allowed module prefixes. It is
    empty by default, so the mode is disabled unless an operator opts in.
    The gate is checked *before* import, because importing runs the target
    module's top-level code.

    Returns:
        ``(model_fn, normalized_mode, model_label)``

    Raises:
        ValueError: unknown mode, missing/invalid ``model_ref``, a module
            outside the allowlist, or import/attribute failures (honest
            refuse — no silent facade).
        TypeError: resolved object is not callable.
    """
    import importlib
    from collections.abc import Callable

    mode = (model_mode or "passthrough").strip().lower()
    ref = (model_ref or "").strip()

    if mode in ("passthrough", "identity", ""):
        if ref:
            raise ValueError(
                "model_ref is only valid with model_mode='module'; "
                f"got model_mode={mode!r} with model_ref={ref!r}"
            )

        def _passthrough(prompt: str) -> str:
            return prompt

        return _passthrough, "passthrough", "identity_passthrough"

    if mode == "module":
        if not ref:
            raise ValueError(
                "model_mode='module' requires model_ref as "
                "'package.module:callable_name'"
            )
        if ":" not in ref:
            raise ValueError(
                "model_ref must be 'package.module:callable_name' "
                f"(got {ref!r})"
            )
        mod_path, fn_name = ref.rsplit(":", 1)
        if not mod_path or not fn_name:
            raise ValueError(
                "model_ref must be 'package.module:callable_name' "
                f"(got {ref!r})"
            )
        # Gate before importing: `importlib.import_module` executes the
        # target module's top-level code, so an un-allowlisted ref must be
        # refused here rather than after resolution. Without this, any
        # importable dotted path resolves (`os:system`), and GenerateSolver
        # feeds each dataset `input` row to it as the argument — so dataset
        # content, which can arrive from `mltk_import`, reaches a call sink.
        allowed = tuple(
            p.strip()
            for p in os.environ.get("MLTK_MCP_MODEL_MODULES", "").split(",")
            if p.strip()
        )
        if not allowed:
            raise ValueError(
                "model_mode='module' is disabled. Set MLTK_MCP_MODEL_MODULES "
                "to a comma-separated list of allowed module prefixes "
                "(e.g. 'myorg.models') to enable trusted model injection."
            )
        if not any(mod_path == p or mod_path.startswith(p + ".") for p in allowed):
            raise ValueError(
                f"Module {mod_path!r} is not in the MLTK_MCP_MODEL_MODULES "
                f"allowlist {list(allowed)}."
            )
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as exc:
            raise ValueError(
                f"Cannot import model module {mod_path!r}: {exc}"
            ) from exc
        try:
            fn = getattr(mod, fn_name)
        except AttributeError as exc:
            raise ValueError(
                f"Module {mod_path!r} has no attribute {fn_name!r}"
            ) from exc
        if not callable(fn):
            raise TypeError(
                f"model_ref {ref!r} resolved to non-callable "
                f"{type(fn).__name__}"
            )
        model_fn: Callable[[str], str] = fn  # type: ignore[assignment]
        return model_fn, "module", ref

    raise ValueError(
        f"Unknown model_mode={model_mode!r}. "
        "Supported: 'passthrough' (default), 'module' "
        "(with model_ref='package.module:callable')."
    )


# Components that exist in mltk.eval but cannot be built from the MCP
# surface, mapped to why. Every MCP parameter is a string, so anything
# needing a caller-supplied callable or structured value has no way to
# receive one. These refuse with their own reason rather than the
# generic unknown-value error, because the name is real and documented
# — the caller has not made a typo.
_MCP_UNAVAILABLE_SCORERS: dict[str, str] = {
    "llm_judge": (
        "'llm_judge' requires a judge_fn callable, which cannot be "
        "passed over MCP (every tool parameter is a string). Use the "
        "Python API instead: EvalTask(scorers=LLMJudgeScorer(judge_fn=...))."
    ),
}

_MCP_UNAVAILABLE_SOLVERS: dict[str, str] = {
    "few_shot": (
        "'few_shot' requires an examples list of (input, output) pairs, "
        "which cannot be passed over MCP (every tool parameter is a "
        "string). Use the Python API instead: "
        "EvalTask(solver=FewShotSolver(examples=[...]))."
    ),
}


def _resolve_eval_components(
    solver: str, scorer: str,
) -> tuple[Any, Any, str, str]:
    """Resolve solver/scorer names to classes, refusing unknown values.

    Returns ``(solver_cls, scorer_cls, normalized_solver,
    normalized_scorer)``. The normalized names are what the caller
    should echo back, and after this function they are guaranteed to
    describe what actually ran.

    Raises:
        ValueError: unknown solver or scorer name, or a component that
            exists but is unusable over MCP (honest refuse — never a
            silent fallback to a different solver or scorer).
    """
    from mltk.eval import (
        ChainOfThoughtSolver,
        ExactMatchScorer,
        GenerateSolver,
        IncludesScorer,
        PatternScorer,
    )

    solver_map = {
        "generate": GenerateSolver,
        "chain_of_thought": ChainOfThoughtSolver,
    }
    scorer_map = {
        "exact_match": ExactMatchScorer,
        "includes": IncludesScorer,
        "pattern": PatternScorer,
    }

    # An omitted/blank value means "use the documented default", which
    # matches how model_mode treats "". An unrecognized value does not.
    sk = (solver or "").strip().lower() or "generate"
    rk = (scorer or "").strip().lower() or "exact_match"

    if sk in _MCP_UNAVAILABLE_SOLVERS:
        raise ValueError(_MCP_UNAVAILABLE_SOLVERS[sk])
    if sk not in solver_map:
        raise ValueError(
            f"Unknown solver={solver!r}. Supported: "
            f"{', '.join(sorted(solver_map))}."
        )
    if rk in _MCP_UNAVAILABLE_SCORERS:
        raise ValueError(_MCP_UNAVAILABLE_SCORERS[rk])
    if rk not in scorer_map:
        raise ValueError(
            f"Unknown scorer={scorer!r}. Supported: "
            f"{', '.join(sorted(scorer_map))}."
        )

    return solver_map[sk], scorer_map[rk], sk, rk


# ----- Workflow hints for agent orchestration -----

_WORKFLOW_HINTS: dict[str, dict[str, Any]] = {
    "mltk_scan": {
        "position": "start",
        "next_tools": [
            "mltk_suggest", "mltk_test", "mltk_report",
        ],
    },
    "mltk_test": {
        "position": "middle",
        "next_tools": ["mltk_report"],
    },
    "mltk_list": {
        "position": "start",
        "next_tools": ["mltk_eval"],
    },
    "mltk_eval": {
        "position": "middle",
        "next_tools": ["mltk_report"],
    },
    "mltk_dataset": {
        "position": "info",
        "next_tools": ["mltk_eval"],
    },
    "mltk_report": {
        "position": "end",
        "next_tools": [],
    },
    "mltk_suggest": {
        "position": "middle",
        "next_tools": [
            "mltk_experiment", "mltk_create_issue",
        ],
    },
    "mltk_experiment": {
        "position": "middle",
        "next_tools": ["mltk_create_pr"],
    },
    "mltk_create_pr": {
        "position": "late",
        "next_tools": ["mltk_create_issue"],
    },
    "mltk_create_issue": {
        "position": "end",
        "next_tools": [],
    },
    "mltk_workflow": {
        "position": "info",
        "next_tools": ["mltk_scan", "mltk_list"],
    },
    "mltk_container_scan": {
        "position": "start",
        "next_tools": ["mltk_report", "mltk_create_issue"],
    },
    "mltk_import": {
        "position": "start",
        "next_tools": ["mltk_dataset", "mltk_eval", "mltk_test"],
    },
}


def _with_hint(
    tool_name: str, payload: dict[str, Any],
) -> dict[str, Any]:
    """Inject workflow_hint into a payload dict."""
    hint = _WORKFLOW_HINTS.get(tool_name)
    if hint:
        payload["workflow_hint"] = hint
    return payload


def _scan_next_step(findings: list[dict[str, Any]]) -> str:
    """Return severity-conditional suggested_next_step."""
    severities = {
        f.get("severity", "").lower() for f in findings
    }
    if "critical" in severities:
        return (
            "Critical findings detected. Run mltk_suggest "
            "on each critical finding, then mltk_experiment "
            "to rank fixes, and mltk_create_pr to apply."
        )
    if "warning" in severities:
        return (
            "Warnings found. Run mltk_suggest for fix ideas, "
            "then mltk_create_issue to track remediation."
        )
    return (
        "Only informational findings. Run mltk_report "
        "to document results."
    )


def create_server() -> FastMCP:
    """Create and configure the mltk MCP server."""
    if FastMCP is None:
        raise ImportError(
            "The 'mcp' package is required. "
            "Install with: pip install mlspec[mcp]"
        )
    mcp = FastMCP(
        "mltk", version=__version__,
        description=(
            "ML Test Kit -- pytest for ML. "
            "Scan, test, evaluate, and report."
        ),
    )
    _register_tools(mcp)
    return mcp


def _register_tools(mcp: FastMCP) -> None:  # noqa: C901
    """Register all mltk tools on the server."""

    @mcp.tool()
    def mltk_scan(path: str, scanners: str = "all") -> str:
        """Return findings from a JSON scan report or a static Python file listing.

        Python files and directories return a static file listing only;
        no ML/data scan is performed because scans require model + data
        inputs via the CLI.

        Args:
            path: Path to Python file, directory, or
                JSON scan report to load.
            scanners: Comma-separated names or 'all'.
        """
        try:
            target = Path(path).resolve()
            if not target.exists():
                return _error(
                    f"Path not found: {path}",
                    suggested_action="Provide a valid path.",
                )
            enabled = None
            if scanners.strip().lower() != "all":
                enabled = [
                    s.strip() for s in scanners.split(",")
                    if s.strip()
                ]
            if target.suffix == ".json":
                raw = json.loads(
                    target.read_text(encoding="utf-8")
                )
                findings = raw.get("findings", [])
                return _ok(_with_hint("mltk_scan", {
                    "scan_performed": True,
                    "findings": findings,
                    "scanners_run": raw.get("scanners_run", []),
                    "duration_ms": raw.get("duration_ms", 0),
                    "suggested_next_step": (
                        _scan_next_step(findings)
                    ),
                }))
            files: list[str] = []
            if target.is_dir():
                files = [
                    str(f.relative_to(target))
                    for f in sorted(
                        target.rglob("*.py")
                    )
                ][:50]
            else:
                files = [target.name]
            return _ok(_with_hint("mltk_scan", {
                "path": str(target),
                "scan_performed": False,
                "message": (
                    "No ML/data scan was performed; returned a "
                    "static file listing only. A real scan requires "
                    "the mltk CLI with model + data."
                ),
                "scanners_available": _SCANNER_NAMES,
                "enabled": enabled or "all",
                "python_files": files,
                "file_count": len(files),
                "suggested_next_step": (
                    "Run mltk scan CLI with model + "
                    "data for full scan: mltk scan -h"
                ),
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_test(
        suite_path: str, verbose: bool = False,
    ) -> str:
        """Run .py tests via pytest, or parse a YAML suite (no execution).

        YAML/.yml paths are parse-only: tests are loaded and returned with
        status ``parsed``, ``passed=0``, ``failed=0``. They are not executed.
        ``.py`` paths run ``python -m pytest`` as a subprocess and return
        real pass/fail counts from pytest output.

        Args:
            suite_path: Path to .yaml suite (parse only) or .py file (pytest).
            verbose: Include detailed per-test output / full pytest log.
        """
        try:
            target = Path(suite_path).resolve()
            if not target.exists():
                return _error(
                    f"Suite not found: {suite_path}",
                    suggested_action=(
                        "Provide a .yaml or .py path."
                    ),
                )
            suffix = target.suffix.lower()
            if suffix in (".yaml", ".yml"):
                import yaml
                raw = yaml.safe_load(
                    target.read_text(encoding="utf-8")
                )
                if not isinstance(raw, dict):
                    return _error(
                        "YAML must be a mapping.",
                    )
                tests = raw.get("tests", [])
                results = [
                    {"name": t.get("name", f"test_{i}"),
                     "definition": t, "status": "parsed"}
                    for i, t in enumerate(tests)
                ]
                return _ok(_with_hint("mltk_test", {
                    "suite": raw.get("name", target.stem),
                    "total": len(tests),
                    "passed": 0, "failed": 0,
                    "mode": "parse_only",
                    "results": results if verbose else [],
                    "suggested_next_step": (
                        "Run with pytest: pytest "
                        + str(target)
                    ),
                }))
            if suffix == ".py":
                args = [
                    sys.executable, "-m", "pytest",
                    str(target), "--tb=short", "-q",
                ]
                if verbose:
                    args.append("-v")
                proc = subprocess.run(
                    args, capture_output=True,
                    text=True, timeout=120,
                )
                output = proc.stdout + proc.stderr
                lines = output.strip().splitlines()
                passed = failed = 0
                for ln in reversed(lines):
                    lo = ln.lower()
                    if "passed" in lo or "failed" in lo:
                        pm = re.search(r"(\d+)\s+passed", lo)
                        fm = re.search(r"(\d+)\s+failed", lo)
                        if pm:
                            passed = int(pm.group(1))
                        if fm:
                            failed = int(fm.group(1))
                        break
                return _ok(_with_hint("mltk_test", {
                    "total": passed + failed,
                    "passed": passed, "failed": failed,
                    "mode": "pytest",
                    "exit_code": proc.returncode,
                    "output": (
                        output if verbose else lines[-5:]
                    ),
                    "suggested_next_step": (
                        "Fix failures and re-run."
                        if failed
                        else "All passed. Run mltk_scan "
                        "for more coverage."
                    ),
                }))
            return _error(
                f"Unsupported type: {suffix}",
                suggested_action="Use .yaml or .py.",
            )
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_list(
        filter_text: str = "", domain: str = "",
    ) -> str:
        """List available mltk assertions for ML testing.

        Args:
            filter_text: Keyword filter (e.g. 'drift').
            domain: Domain filter: data, model, llm,
                training, monitor, inference, compliance.
        """
        try:
            from mltk.cli._discovery import (
                discover_assertions,
            )
            kw = filter_text.strip()
            if domain.strip() and not kw:
                kw = domain.strip()
            elif domain.strip():
                kw = f"{kw} {domain.strip()}"
            entries = discover_assertions(kw)
            total = sum(len(v) for v in entries.values())
            assertions: list[dict[str, str]] = []
            domains_found: list[str] = []
            for cat, items in sorted(entries.items()):
                domains_found.append(cat)
                for e in items:
                    assertions.append({
                        "name": e["name"],
                        "domain": cat,
                        "description": e["doc"],
                    })
            return _ok(_with_hint("mltk_list", {
                "total": total,
                "assertions": assertions,
                "domains": domains_found,
                "suggested_next_step": (
                    "Pick an assertion for your tests, "
                    "or run mltk_scan to auto-generate."
                ),
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_eval(
        dataset_path: str,
        scorer: str = "exact_match",
        solver: str = "generate",
        model_mode: str = "passthrough",
        model_ref: str = "",
    ) -> str:
        """Run a scorer-pipeline eval with an explicit model mode.

        Default ``model_mode='passthrough'`` uses an identity model
        (prompt echoed unchanged). Metrics then reflect scorer behavior
        against passthrough only — not real model quality.

        For a real model without baking a vendor SDK into MCP, pass
        ``model_mode='module'`` and ``model_ref='package.module:callable'``
        where the callable is ``(prompt: str) -> str``. Only load trusted
        callables (local import). Unknown modes refuse honestly.

        Unknown ``solver``/``scorer`` values refuse with a recoverable
        error listing the supported names — they are never silently
        replaced by a default, so the ``solver``/``scorer`` echoed in the
        response always names what actually ran.

        Args:
            dataset_path: Path to CSV/JSON with 'input'
                and 'target' columns.
            scorer: exact_match, includes, or pattern. ``llm_judge``
                is not available here — it needs a judge callable.
            solver: generate or chain_of_thought. ``few_shot`` is not
                available here — it needs an examples list.
            model_mode: ``passthrough`` (default) or ``module``.
            model_ref: Required for ``module`` —
                ``package.module:callable_name``.
        """
        try:
            from mltk.eval.task import (
                EvalTask,
                load_dataset,
            )
            target = Path(dataset_path).resolve()
            if not target.exists():
                return _error(
                    f"Not found: {dataset_path}",
                    suggested_action="Provide .csv/.json.",
                )
            samples = load_dataset(str(target))
            if not samples:
                return _error("Dataset is empty.")

            try:
                solver_cls, scorer_cls, sk, rk = (
                    _resolve_eval_components(
                        solver=solver, scorer=scorer,
                    )
                )
            except ValueError as exc:
                return _error(
                    str(exc),
                    suggested_action=(
                        "Pass a supported solver/scorer name, or use "
                        "the Python EvalTask API for components that "
                        "need a callable or an examples list."
                    ),
                )

            try:
                model_fn, norm_mode, model_label = (
                    _resolve_eval_model(
                        model_mode=model_mode,
                        model_ref=model_ref,
                    )
                )
            except (ValueError, TypeError) as exc:
                return _error(
                    str(exc),
                    suggested_action=(
                        "Use model_mode='passthrough' or "
                        "model_mode='module' with "
                        "model_ref='package.module:callable'."
                    ),
                )

            task = EvalTask(
                name="mcp-eval",
                solver=solver_cls(),
                scorers=scorer_cls(),
                dataset=samples,
            )
            result = task.run(model_fn)
            next_step = (
                "Review metrics. For a non-passthrough model, "
                "re-run with model_mode='module' and "
                "model_ref='package.module:callable', or use "
                "the Python EvalTask API."
                if norm_mode == "passthrough"
                else (
                    "Review metrics from the injected model. "
                    "Compare against passthrough baseline if needed."
                )
            )
            return _ok(_with_hint("mltk_eval", {
                "metrics": result.metrics,
                "sample_count": result.total_samples,
                "duration_ms": result.duration_ms,
                "solver": sk, "scorer": rk,
                "model_mode": norm_mode,
                "model": model_label,
                "model_ref": (
                    model_ref.strip() if norm_mode == "module" else ""
                ),
                "suggested_next_step": next_step,
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_dataset(
        name: str, version: str = "",
    ) -> str:
        """Get info about a registered evaluation dataset with quality metrics.

        Args:
            name: Dataset name in the registry.
            version: Version to inspect (empty=latest).
        """
        try:
            from mltk.eval.dataset import DatasetRegistry
            registry = DatasetRegistry()
            ver = version.strip() or None
            if not registry.exists(name, ver):
                avail = registry.list()
                names = [i.name for i in avail]
                return _error(
                    f"Dataset '{name}' not found"
                    + (f" v{ver}" if ver else "")
                    + ".",
                    suggested_action=(
                        "Available: "
                        + (", ".join(names) or "none")
                        + ". Use DatasetRegistry.save()."
                    ),
                )
            ds = registry.load(name, ver)
            inp = [s.input for s in ds.samples]
            n, u = len(inp), len(set(inp))
            dup = round(1.0 - u / n, 4) if n else 0.0
            return _ok(_with_hint("mltk_dataset", {
                "info": {
                    "name": ds.name, "version": ds.version,
                    "card": ds.card.to_dict(),
                },
                "quality": {
                    "sample_count": ds.sample_count,
                    "target_coverage": round(
                        ds.target_coverage, 4),
                    "duplicate_rate": dup,
                    "categories": ds.categories,
                    "fingerprint": ds.fingerprint[:16] + "...",
                },
                "versions": registry.versions(name),
                "suggested_next_step": (
                    "Run mltk_eval with this dataset."
                ),
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_report(
        title: str,
        description: str = "",
        results_json: str = "",
    ) -> str:
        """Generate a formatted ML test report from scan or test results.

        Args:
            title: Report title.
            description: What was tested.
            results_json: JSON string of results.
        """
        try:
            results: list[dict[str, Any]] = []
            if results_json.strip():
                parsed = json.loads(results_json)
                if isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = [parsed]
            now = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            lines = [f"# {title}", "", f"Generated: {now}"]
            if description:
                lines += ["", f"**Description:** {description}"]
            lines.append("")
            def _is_pass(r: dict[str, Any]) -> bool:
                return bool(
                    r.get("passed") or r.get("status") == "ok"
                )
            if not results:
                lines.append("*No results yet.*")
            else:
                t, p = len(results), sum(1 for r in results if _is_pass(r))
                lines += [
                    "## Summary", "",
                    "| Metric | Value |", "|--------|-------|",
                    f"| Total  | {t} |", f"| Passed | {p} |",
                    f"| Failed | {t - p} |", "", "## Results", "",
                ]
                for i, r in enumerate(results):
                    nm = r.get("name", f"Result {i+1}")
                    st = "PASS" if _is_pass(r) else "FAIL"
                    msg = r.get("message", r.get("error", ""))
                    tag = f": {msg}" if msg else ""
                    lines.append(f"- **{nm}** [{st}]{tag}")
            lines += ["", "---", f"*mltk v{__version__}*"]
            report_text = "\n".join(lines)
            n = len(results)
            p = sum(1 for r in results if _is_pass(r))
            summary = (
                f"{n} results: {p} passed, {n-p} failed"
                if n else "No results provided."
            )
            return _ok(_with_hint("mltk_report", {
                "report_text": report_text,
                "summary": summary,
                "suggested_next_step": (
                    "Share in PR or CI. Run mltk_scan "
                    "for fresh results."
                ),
            }))
        except json.JSONDecodeError as exc:
            return _error(
                f"Invalid results_json: {exc}",
                suggested_action="Pass valid JSON.",
            )
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_suggest(
        finding_json: str,
        category: str = "",
        max_results: int = 5,
    ) -> str:
        """Get fix suggestions already attached to a finding JSON.

        Re-exports and optionally filters the ``suggested_fixes`` list on the
        input finding. Does not generate new fixes — if the finding has none,
        the result is empty. Fixes originate from scanners or prior reports.

        Args:
            finding_json: JSON string of a single scan finding
                (as produced by mltk_scan or ScanReport.to_json()).
            category: Filter by category: code, config, data,
                process. Empty = all categories.
            max_results: Maximum number of suggestions to return.
        """
        try:
            if not finding_json.strip():
                return _error(
                    "Empty finding_json.",
                    suggested_action=(
                        "Provide a JSON object from mltk_scan."
                    ),
                )
            try:
                parsed = json.loads(finding_json)
            except json.JSONDecodeError as exc:
                return _error(
                    f"Invalid finding_json: {exc}",
                    suggested_action="Pass valid JSON.",
                )
            if isinstance(parsed, list):
                return _error(
                    "finding_json must be a single object, "
                    "not an array.",
                    suggested_action=(
                        "Pass one finding at a time."
                    ),
                )
            fixes = parsed.get("suggested_fixes", [])
            if not fixes:
                return _ok(_with_hint("mltk_suggest", {
                    "suggestions": [],
                    "total": 0,
                    "filtered_by": (
                        category.strip().lower() or "none"
                    ),
                    "suggested_next_step": (
                        "No suggestions available for this "
                        "finding. Run mltk_scan with more "
                        "scanners for deeper analysis."
                    ),
                }))
            cat = category.strip().lower()
            if cat:
                fixes = [
                    f for f in fixes
                    if f.get("category", "").lower() == cat
                ]
            limit = max(1, min(max_results, 50))
            fixes = fixes[:limit]
            suggestions = [
                {
                    "category": f.get("category", ""),
                    "title": f.get("title", ""),
                    "description": f.get("description", ""),
                    "confidence": f.get("confidence", ""),
                    "code_snippet": f.get("code_snippet", ""),
                }
                for f in fixes
            ]
            return _ok(_with_hint("mltk_suggest", {
                "suggestions": suggestions,
                "total": len(suggestions),
                "filtered_by": cat or "none",
                "suggested_next_step": (
                    "Apply the highest-confidence fix first, "
                    "then re-scan to verify."
                ),
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_experiment(
        finding_json: str,
        rank_by: str = "passed",
        max_results: int = 5,
        sandbox: bool = False,
    ) -> str:
        """Rank fix suggestions for a finding using heuristic scoring.

        Scores each fix based on category actionability, confidence level,
        and code snippet availability. Use after mltk_suggest to prioritize
        which fix to try first.

        When ``sandbox=True``, runs each fix hypothesis in an isolated
        git worktree via :class:`SandboxedExperimentRunner` instead of
        pure heuristic scoring.  Requires ``git`` CLI and a git repo.

        Args:
            finding_json: JSON string of a single scan finding
                (from mltk_scan or ScanReport.to_json()).
            rank_by: Strategy: "passed" (confidence-first),
                "delta" (actionability-first), "composite" (balanced).
            max_results: Maximum results to return (1-50).
            sandbox: If True, run fixes in isolated git worktrees
                instead of heuristic ranking.
        """
        try:
            if not finding_json.strip():
                return _error(
                    "Empty finding_json.",
                    suggested_action=(
                        "Provide a JSON object from mltk_scan."
                    ),
                )
            try:
                parsed = json.loads(finding_json)
            except json.JSONDecodeError as exc:
                return _error(
                    f"Invalid finding_json: {exc}",
                    suggested_action="Pass valid JSON.",
                )
            if isinstance(parsed, list):
                return _error(
                    "finding_json must be a single object, "
                    "not an array.",
                    suggested_action=(
                        "Pass one finding at a time."
                    ),
                )

            if sandbox:
                return _experiment_sandbox(
                    parsed, rank_by, max_results,
                )

            fixes = parsed.get("suggested_fixes", [])
            if not fixes:
                return _ok(_with_hint("mltk_experiment", {
                    "ranked_fixes": [],
                    "total": 0,
                    "strategy": rank_by,
                    "suggested_next_step": (
                        "No fixes available for this finding. "
                        "Run mltk_suggest first to generate "
                        "fix suggestions."
                    ),
                }))
            confidence_map = {
                "high": 3, "medium": 2, "low": 1,
            }
            category_map = {
                "code": 4, "config": 3, "data": 2,
                "process": 1,
            }

            scored: list[dict[str, Any]] = []
            for f in fixes:
                conf = f.get("confidence", "").lower()
                cat = f.get("category", "").lower()
                snippet = f.get("code_snippet", "")
                conf_score = confidence_map.get(conf, 1)
                cat_score = category_map.get(cat, 1)
                snip_score = 1 if snippet else 0
                scored.append({
                    **f,
                    "_conf": conf_score,
                    "_cat": cat_score,
                    "_snip": snip_score,
                })

            strategy = rank_by.strip().lower()
            if strategy == "delta":
                scored.sort(
                    key=lambda x: (
                        x["_cat"], x["_conf"], x["_snip"],
                    ),
                    reverse=True,
                )
            elif strategy == "composite":
                scored.sort(
                    key=lambda x: (
                        x["_conf"] * 0.4
                        + x["_cat"] * 0.3
                        + x["_snip"] * 0.3
                    ),
                    reverse=True,
                )
            else:
                # "passed" or any unrecognised strategy
                scored.sort(
                    key=lambda x: (
                        x["_conf"], x["_cat"], x["_snip"],
                    ),
                    reverse=True,
                )
                strategy = "passed"

            limit = max(1, min(max_results, 50))
            scored = scored[:limit]

            ranked: list[dict[str, Any]] = []
            for rank, entry in enumerate(scored, start=1):
                score = round(
                    entry["_conf"] * 0.4
                    + entry["_cat"] * 0.3
                    + entry["_snip"] * 0.3,
                    2,
                )
                clean = {
                    k: v for k, v in entry.items()
                    if not k.startswith("_")
                }
                clean["score"] = score
                clean["rank"] = rank
                ranked.append(clean)

            return _ok(_with_hint("mltk_experiment", {
                "ranked_fixes": ranked,
                "total": len(ranked),
                "strategy": strategy,
                "suggested_next_step": (
                    "Apply the top-ranked fix and re-scan "
                    "to verify improvement."
                ),
            }))
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_workflow() -> str:
        """Return the canonical mltk agent workflow.

        Call this first to understand the available tools and
        recommended execution order before starting an ML
        testing workflow.  Returns pipeline paths for different
        severity levels and a decision tree for routing.
        """
        return _ok(_with_hint("mltk_workflow", {
            "pipeline": {
                "critical_path": [
                    "mltk_scan", "mltk_suggest",
                    "mltk_experiment", "mltk_create_pr",
                    "mltk_create_issue",
                ],
                "medium_path": [
                    "mltk_scan", "mltk_suggest",
                    "mltk_create_issue",
                ],
                "low_path": [
                    "mltk_scan", "mltk_report",
                ],
                "eval_path": [
                    "mltk_list", "mltk_eval",
                    "mltk_report",
                ],
                "test_path": [
                    "mltk_scan", "mltk_test",
                    "mltk_report",
                ],
            },
            "decision_tree": (
                "1. Call mltk_scan on the target path.\n"
                "2. If findings have severity=critical: "
                "mltk_suggest -> mltk_experiment -> "
                "mltk_create_pr -> mltk_create_issue.\n"
                "3. If findings have severity=warning: "
                "mltk_suggest -> mltk_create_issue.\n"
                "4. If findings are informational only: "
                "mltk_report.\n"
                "5. For evaluation workflows: "
                "mltk_list -> mltk_eval -> mltk_report."
            ),
            "tool_count": 13,
            "suggested_next_step": (
                "Call mltk_scan to begin."
            ),
        }))

    @mcp.tool()
    def mltk_create_pr(
        finding_json: str,
        fix_json: str,
        repo: str,
        base_branch: str = "main",
        draft: bool = True,
    ) -> str:
        """Create a GitHub PR with a fix for a scan finding.

        Takes a scan finding and a fix suggestion, creates an isolated
        git branch with the fix applied, pushes to remote, and opens
        a pull request via the GitHub REST API.

        Args:
            finding_json: JSON string of a scan finding
                (from mltk_scan output).
            fix_json: JSON string of a fix suggestion
                (from mltk_suggest or mltk_experiment output).
            repo: GitHub repository in ``owner/name`` format.
            base_branch: Target branch for the PR (default ``"main"``).
            draft: Create as draft PR when ``True`` (default).
        """
        try:
            return _create_pr_impl(
                finding_json, fix_json, repo,
                base_branch, draft,
            )
        except RuntimeError as exc:
            _log(traceback.format_exc())
            return _error(
                str(exc),
                suggested_action=(
                    "Create an issue instead of a PR."
                ),
                fallback_parameters={
                    "tool": "mltk_create_issue",
                },
            )
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_create_issue(
        finding_json: str,
        tracker: str = "github",
        project: str = "",
        config_json: str = "{}",
        pr_url: str = "",
    ) -> str:
        """Create an issue ticket from a scan finding.

        Supports GitHub Issues, Jira, Asana, and Linear backends.
        Applies deduplication to avoid creating duplicate tickets
        for the same finding.
        Optionally links an existing PR to the created issue.

        Args:
            finding_json: JSON string of a scan finding
                (from mltk_scan output).
            tracker: Backend tracker: ``"github"``, ``"jira"``,
                ``"asana"``, or ``"linear"``.
            project: Project key or ID (required for Jira and Asana;
                used by Linear when ``team_id`` is omitted; ignored
                for GitHub).
            config_json: JSON with adapter credentials.
                GitHub: ``{"repo": "owner/name", "token": "ghp_..."}``
                Jira: ``{"url": "https://...", "email": "...", "token": "..."}``
                Asana: ``{"token": "...", "workspace_gid": "..."}``
                Linear: ``{"api_key": "...", "team_id": "..."}``
            pr_url: Optional PR URL to link to the created issue.
        """
        try:
            return _create_issue_impl(
                finding_json, tracker, project,
                config_json, pr_url,
            )
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_container_scan(
        image: str,
        max_critical: int = 0,
        max_high: int = 0,
    ) -> str:
        """Scan a container image for vulnerabilities and secrets using Trivy.

        Args:
            image: Container image reference (e.g. ``"alpine:3.18"``).
            max_critical: Maximum allowed CRITICAL severity CVEs.
            max_high: Maximum allowed HIGH severity CVEs.
        """
        try:
            from mltk.container.assertions import (  # noqa: PLC0415
                assert_container_vulnerabilities,
                assert_no_secrets_in_image,
            )
            from mltk.core.assertion import MltkAssertionError  # noqa: PLC0415
        except ImportError as exc:
            return _error(
                str(exc),
                recoverable=True,
                suggested_action="Install the container extra: pip install mlspec[container]",
            )
        try:
            # Run both assertions.  MltkAssertionError = policy threshold exceeded
            # (scan ran fine, image just failed).  Any other exception = infra error.
            try:
                vuln = assert_container_vulnerabilities(
                    image, max_critical=max_critical, max_high=max_high,
                )
            except MltkAssertionError as exc:
                vuln = exc.result
            try:
                secret = assert_no_secrets_in_image(image)
            except MltkAssertionError as exc:
                secret = exc.result
            return _ok(_with_hint("mltk_container_scan", {
                "image": image,
                "passed": vuln.passed and secret.passed,
                "vulnerabilities": {
                    "passed": vuln.passed,
                    "message": vuln.message,
                    "details": vuln.details,
                },
                "secrets": {
                    "passed": secret.passed,
                    "message": secret.message,
                    "details": secret.details,
                },
                "suggested_next_step": (
                    "Run mltk_report to export results to HTML, "
                    "or add assert_container_vulnerabilities to your pytest suite."
                ),
            }))
        except Exception as exc:  # noqa: BLE001
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_import(
        source: str,
        split: str = "",
        input_column: str = "",
        target_column: str = "",
        name: str = "",
        golden_path: str = "",
        golden_target_column: str = "",
        golden_key: str = "",
        golden_key_column: str = "",
        judge: bool = False,
        register: bool = False,
        output_path: str = "",
    ) -> str:
        """Import a dataset into an mltk pytest suite and eval dataset.

        Maps columns, classifies the task, builds a suite, and generates a
        committable pytest scaffold.  Returns the mapping preview, task type,
        and generated code as a string; writes the file to disk only when
        ``output_path`` is set.  Optionally binds a golden reference file and
        registers the dataset behind a blocking quality gate.

        Args:
            source: Local file path or HuggingFace dataset id.
            split: Dataset split for HuggingFace sources.
            input_column: Force-map this column as the eval input.
            target_column: Force-map this column as the eval target.
            name: Dataset name for the generated eval dataset.
            golden_path: Golden/reference file to bind (CSV/TSV/JSON/JSONL).
            golden_target_column: Golden column holding the reference answer
                (required with ``golden_path``).
            golden_key: Sample-side join key -- ``"input"`` or a metadata
                field name; empty binds by row order.
            golden_key_column: Golden-side key column if it differs from
                ``golden_key``.
            judge: Emit an LLM-judge fallback test for unmatched samples
                (requires ``golden_path``).
            register: Save the dataset to the local registry after a blocking
                quality gate.
            output_path: If set, write the generated pytest file to this path.
        """
        try:
            from mltk.importer.classify import classify_task
            from mltk.importer.codegen import generate_pytest
            from mltk.importer.golden import (
                GoldenSpec,
                bind_golden,
                load_golden,
            )
            from mltk.importer.loader import DatasetImporter
            from mltk.importer.registry import register_dataset
            from mltk.importer.schema import ColumnRole
            from mltk.importer.suite_gen import build_suite
        except ImportError as exc:
            return _error(
                str(exc),
                recoverable=True,
                suggested_action=(
                    "Install the importer extra: pip install mlspec[importer]"
                ),
            )

        try:
            load_kwargs = {
                k: v
                for k, v in {
                    "split": split or None,
                    "input_column": input_column or None,
                    "target_column": target_column or None,
                }.items()
                if v is not None
            }
            result = DatasetImporter.load(source, **load_kwargs)
            if not result.mapping.columns_with_role(ColumnRole.INPUT):
                return _error(
                    "No column could be mapped to the eval INPUT role.",
                    suggested_action=(
                        "Pass input_column=... to force a mapping."
                    ),
                )

            task_type = classify_task(result.mapping)
            task_value = getattr(task_type, "value", str(task_type))
            dataset_name = name or (Path(source.rstrip("/\\")).stem or "dataset")
            eval_dataset = result.to_eval_dataset(name=dataset_name)

            golden_spec = None
            golden_summary = None
            if golden_path:
                if not golden_target_column:
                    return _error(
                        "golden_path requires golden_target_column.",
                    )
                golden_rows = load_golden(golden_path)
                eval_dataset, golden_report = bind_golden(
                    eval_dataset,
                    golden_rows,
                    target_column=golden_target_column,
                    key=golden_key or None,
                    golden_key=golden_key_column or None,
                )
                golden_summary = golden_report.summary()
                golden_spec = GoldenSpec(
                    path=golden_path,
                    target_column=golden_target_column,
                    key=golden_key or None,
                    golden_key=golden_key_column or None,
                    judge=judge,
                )

            suite = build_suite(eval_dataset, result.mapping, task_type)
            code = generate_pytest(
                result,
                task_type,
                dataset_name=dataset_name,
                output_path=output_path or None,
                load_kwargs=load_kwargs or None,
                golden_spec=golden_spec,
            )

            payload: dict[str, Any] = {
                "mapping_preview": result.mapping.preview(),
                "task_type": task_value,
                "dataset_name": dataset_name,
                "sample_count": eval_dataset.sample_count,
                "assertion_count": len(suite),
                "generated_code": code,
                "file_written": output_path or None,
                "suggested_next_step": (
                    "Review the generated pytest. Set MLTK_PREDICT_FN to "
                    "un-skip Tier 2 tests, or set register=true to save."
                ),
            }
            if golden_summary is not None:
                payload["golden_binding"] = golden_summary
            if register:
                reg = register_dataset(eval_dataset)
                payload["registration"] = {
                    "saved": reg.saved,
                    "quality_passed": reg.quality_passed,
                    "reason": reg.reason,
                    "path": reg.path,
                }

            return _ok(_with_hint("mltk_import", payload))
        except FileNotFoundError as exc:
            return _error(
                str(exc),
                suggested_action="Check the source and golden file paths.",
            )
        except (ValueError, NotImplementedError) as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            _log(traceback.format_exc())
            return _error(str(exc))


def _experiment_sandbox(
    parsed: dict[str, Any],
    rank_by: str,
    max_results: int,
) -> str:
    """Run sandbox experiment using git worktrees.

    Lazily imports sandbox dependencies, validates git
    availability, constructs domain objects from the parsed
    JSON, and delegates to
    :class:`SandboxedExperimentRunner`.

    Args:
        parsed: Parsed finding JSON dict.
        rank_by: Ranking strategy name.
        max_results: Maximum results to return.

    Returns:
        JSON response string via ``_ok()`` or ``_error()``.
    """
    try:
        from mltk.experiment.worktree import (
            find_git_root,
            git_available,
        )

        if not git_available():
            return _error(
                "Git CLI not found; sandbox mode requires git.",
                suggested_action=(
                    "Install git or use sandbox=False."
                ),
            )

        try:
            repo_root = find_git_root()
        except FileNotFoundError:
            return _error(
                "Not in a git repository; sandbox mode "
                "requires a git repo.",
                suggested_action=(
                    "Run from inside a git repository "
                    "or use sandbox=False."
                ),
            )

        from mltk.experiment.hypothesis import Hypothesis
        from mltk.experiment.sandbox import (
            SandboxedExperimentRunner,
        )
        from mltk.scan.finding import (
            FixSuggestion,
            ScanFinding,
        )

        fixes_raw = parsed.get("suggested_fixes", [])
        if not fixes_raw:
            return _ok(_with_hint("mltk_experiment", {
                "ranked_fixes": [],
                "total": 0,
                "strategy": rank_by,
                "sandbox": True,
                "suggested_next_step": (
                    "No fixes available for this finding. "
                    "Run mltk_suggest first to generate "
                    "fix suggestions."
                ),
            }))

        fix_objs: list[FixSuggestion] = []
        for f in fixes_raw:
            fix_objs.append(FixSuggestion(
                category=f.get("category", "code"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                confidence=f.get("confidence", "low"),
                code_snippet=f.get("code_snippet", ""),
            ))

        from mltk.core.result import Severity, TestResult

        _baseline = TestResult(
            name="sandbox.baseline",
            passed=False,
            severity=Severity.WARNING,
            message="Baseline from MCP sandbox request",
        )
        finding_obj = ScanFinding(
            result=_baseline,
            assertion_fn=lambda: _baseline,
            assertion_args=(),
            assertion_kwargs={},
            scanner_name=parsed.get("scanner_name", ""),
            suggested_fixes=fix_objs,
        )

        hypotheses = [
            Hypothesis(
                fix=fix,
                apply_fn=lambda _f=fix: _baseline,  # noqa: ARG005
                description=fix.title,
            )
            for fix in fix_objs
        ]

        strategy = rank_by.strip().lower() or "passed"
        runner = SandboxedExperimentRunner(
            repo_root=repo_root,
            strategy=strategy,
        )
        result = runner.run(
            finding_obj, hypotheses=hypotheses,
        )

        limit = max(1, min(max_results, 50))
        ranked: list[dict[str, Any]] = []
        for hr in result.hypothesis_results[:limit]:
            ranked.append({
                "category": hr.hypothesis.fix.category,
                "title": hr.hypothesis.fix.title,
                "description": (
                    hr.hypothesis.fix.description
                ),
                "confidence": (
                    hr.hypothesis.fix.confidence
                ),
                "code_snippet": (
                    hr.hypothesis.fix.code_snippet
                ),
                "rank": hr.rank,
                "improvement": hr.improvement,
                "passed": hr.fixed_result.passed,
            })

        payload: dict[str, Any] = {
            "ranked_fixes": ranked,
            "total": len(ranked),
            "strategy": strategy or "passed",
            "sandbox": True,
            "duration_ms": result.duration_ms,
        }

        if result.selected_fix is not None:
            payload["selected_fix"] = (
                result.selected_fix.title
            )

        payload["suggested_next_step"] = (
            "Apply the selected fix from the sandbox "
            "experiment and re-scan to verify."
            if result.any_fix_works
            else "No fix resolved the finding in sandbox. "
            "Try different fixes or manual investigation."
        )

        return _ok(_with_hint("mltk_experiment", payload))
    except Exception as exc:
        _log(traceback.format_exc())
        return _error(str(exc))


def _finding_from_json(
    finding_data: dict[str, Any],
    fixes: list[Any] | None = None,
) -> Any:
    """Build a ScanFinding from parsed JSON.

    Shared by ``_create_pr_impl`` and ``_create_issue_impl``
    to avoid duplicating the Severity-map + TestResult
    construction.
    """
    from mltk.core.result import Severity, TestResult
    from mltk.scan.finding import ScanFinding

    result_data = finding_data.get("result")
    if not isinstance(result_data, dict):
        result_data = finding_data
    _sev_map = {
        "critical": Severity.CRITICAL,
        "warning": Severity.WARNING,
        "info": Severity.INFO,
    }
    details = result_data.get("details", {})
    if not isinstance(details, dict):
        details = {}
    try:
        duration_ms = float(result_data.get("duration_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration_ms = 0.0
    _baseline = TestResult(
        name=result_data.get("name", ""),
        passed=bool(result_data.get("passed", False)),
        severity=_sev_map.get(
            str(result_data.get("severity", "warning")).lower(),
            Severity.WARNING,
        ),
        message=str(result_data.get("message", "")),
        details=details,
        duration_ms=duration_ms,
    )
    return ScanFinding(
        result=_baseline,
        assertion_fn=lambda: _baseline,
        assertion_args=(),
        assertion_kwargs={},
        scanner_name=finding_data.get("scanner_name", ""),
        suggested_fixes=fixes or [],
    )


def _create_pr_impl(
    finding_json: str,
    fix_json: str,
    repo: str,
    base_branch: str,
    draft: bool,
) -> str:
    """Implementation for mltk_create_pr tool.

    Validates inputs, checks git availability, constructs domain
    objects, and delegates to :class:`PullRequestGenerator`.
    """
    if not finding_json.strip():
        return _error(
            "Empty finding_json.",
            suggested_action="Provide a JSON object from mltk_scan.",
        )
    if not fix_json.strip():
        return _error(
            "Empty fix_json.",
            suggested_action="Provide a JSON object from mltk_suggest.",
        )

    try:
        finding_data = json.loads(finding_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"Invalid finding_json: {exc}",
            suggested_action="Pass valid JSON.",
        )

    try:
        fix_data = json.loads(fix_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"Invalid fix_json: {exc}",
            suggested_action="Pass valid JSON.",
        )

    from mltk.experiment.worktree import (
        find_git_root,
        git_available,
    )

    if not git_available():
        return _error(
            "Git CLI not found; PR creation requires git.",
            suggested_action="Install git or create the PR manually.",
        )

    try:
        repo_root = find_git_root()
    except FileNotFoundError:
        return _error(
            "Not in a git repository; PR creation requires a git repo.",
            suggested_action="Run from inside a git repository.",
        )

    from mltk.integrations.github_adapter import GitHubIssuesAdapter
    from mltk.integrations.pr_generator import PullRequestGenerator
    from mltk.scan.finding import FixSuggestion

    fix_obj = FixSuggestion(
        category=fix_data.get("category", "code"),
        title=fix_data.get("title", ""),
        description=fix_data.get("description", ""),
        confidence=fix_data.get("confidence", "low"),
        code_snippet=fix_data.get("code_snippet", ""),
    )

    finding_obj = _finding_from_json(finding_data, fixes=[fix_obj])

    github = GitHubIssuesAdapter(repo)
    if not github.token:
        return _error(
            "No GitHub token found. Set GITHUB_TOKEN env var "
            "or pass a token.",
            suggested_action=(
                "Export GITHUB_TOKEN=ghp_... before running."
            ),
        )
    generator = PullRequestGenerator(
        github=github, repo_root=repo_root,
    )
    pr_result = generator.create_pr(
        finding=finding_obj,
        fix=fix_obj,
        base_branch=base_branch,
        draft=draft,
    )

    return _ok(_with_hint("mltk_create_pr", {
        "url": pr_result.url,
        "branch": pr_result.branch,
        "number": pr_result.number,
        "draft": pr_result.draft,
        "suggested_next_step": (
            "PR created. Link it to an issue with "
            "mltk_create_issue(pr_url=...) or review manually."
        ),
    }))


def _create_issue_impl(
    finding_json: str,
    tracker: str,
    project: str,
    config_json: str,
    pr_url: str,
) -> str:
    """Implementation for mltk_create_issue tool.

    Validates inputs, constructs the appropriate adapter, and
    delegates to :class:`IssueLinker`.
    """
    if not finding_json.strip():
        return _error(
            "Empty finding_json.",
            suggested_action="Provide a JSON object from mltk_scan.",
        )

    try:
        finding_data = json.loads(finding_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"Invalid finding_json: {exc}",
            suggested_action="Pass valid JSON.",
        )

    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return _error(
            f"Invalid config_json: {exc}",
            suggested_action="Pass valid JSON with adapter credentials.",
        )

    tracker_lower = tracker.strip().lower()
    supported_trackers = ("github", "jira", "asana", "linear")
    if tracker_lower not in supported_trackers:
        return _error(
            f"Unsupported tracker: {tracker!r}. "
            "Use 'github', 'jira', 'asana', or 'linear'.",
            suggested_action=(
                "Set tracker to 'github', 'jira', 'asana', or 'linear'."
            ),
        )

    from mltk.integrations.issue_linker import IssueLinker

    # Build adapter
    if tracker_lower == "github":
        from mltk.integrations.github_adapter import (
            GitHubIssuesAdapter,
        )
        adapter = GitHubIssuesAdapter(
            repo=config.get("repo", ""),
            token=config.get("token"),
        )
    elif tracker_lower == "jira":
        from mltk.integrations.jira_adapter import JiraAdapter
        adapter = JiraAdapter(
            instance_url=config.get("url", ""),
            email=config.get("email", ""),
            api_token=config.get("token", ""),
        )
    elif tracker_lower == "asana":
        from mltk.integrations.asana_adapter import AsanaAdapter
        adapter = AsanaAdapter(
            token=config.get("token"),
            workspace_gid=config.get("workspace_gid"),
        )
    else:
        from mltk.integrations.linear_adapter import LinearAdapter
        adapter = LinearAdapter(
            api_key=config.get("api_key"),
            team_id=config.get("team_id"),
        )

    finding_obj = _finding_from_json(finding_data)

    linker = IssueLinker(adapter)
    issue_key = linker.create_from_finding(
        finding_obj, project or "",
    )

    linked = ""
    if pr_url and issue_key:
        linker.link_pr(str(issue_key), pr_url)
        linked = pr_url

    return _ok(_with_hint("mltk_create_issue", {
        "issue_key": str(issue_key) if issue_key else None,
        "issue_url": str(issue_key) if issue_key else None,
        "linked_pr": linked,
        "suggested_next_step": (
            "Issue created. Review and assign priority."
            if issue_key
            else "Issue skipped by dedup. A similar ticket "
            "already exists — search your tracker."
        ),
    }))


def run_server() -> None:
    """Run the MCP server on stdio transport."""
    server = create_server()
    _log("mltk MCP server starting (stdio)")
    server.run(transport="stdio")
