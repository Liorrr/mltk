"""mltk MCP server -- expose ML testing tools to AI agents.

Tools: mltk_scan, mltk_test, mltk_list, mltk_eval,
mltk_dataset, mltk_report, mltk_suggest, mltk_experiment.

Usage: ``mltk serve --mcp`` or ``python -m mltk.mcp``
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
) -> str:
    """Build a JSON error response."""
    return json.dumps({
        "status": "error",
        "error": msg,
        "recoverable": recoverable,
        "suggested_action": (
            suggested_action
            or "Check the error message and retry."
        ),
    }, indent=2)


def create_server() -> FastMCP:
    """Create and configure the mltk MCP server."""
    if FastMCP is None:
        raise ImportError(
            "The 'mcp' package is required. "
            "Install with: pip install mltk[mcp]"
        )
    mcp = FastMCP(
        "mltk", version="0.9.0",
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
        """Scan an ML project for quality issues, drift, bias, and security vulnerabilities.

        Args:
            path: Path to Python file, directory, or
                JSON scan report to analyze.
            scanners: Comma-separated names or 'all'.
        """
        try:
            from mltk.scan import ScanConfig, ScanEngine
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
            ScanEngine(config=ScanConfig(
                enabled_scanners=enabled,
            ))
            if target.suffix == ".json":
                raw = json.loads(
                    target.read_text(encoding="utf-8")
                )
                return _ok({
                    "findings": raw.get("findings", []),
                    "scanners_run": raw.get("scanners_run", []),
                    "duration_ms": raw.get("duration_ms", 0),
                    "suggested_next_step": (
                        "Review findings and run "
                        "mltk_test on suggested tests."
                    ),
                })
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
            return _ok({
                "path": str(target),
                "scanners_available": _SCANNER_NAMES,
                "enabled": enabled or "all",
                "python_files": files,
                "file_count": len(files),
                "suggested_next_step": (
                    "Run mltk scan CLI with model + "
                    "data for full scan: mltk scan -h"
                ),
            })
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_test(
        suite_path: str, verbose: bool = False,
    ) -> str:
        """Run an mltk test suite and return pass/fail results.

        Args:
            suite_path: Path to .yaml suite or .py file.
            verbose: Include detailed per-test output.
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
                return _ok({
                    "suite": raw.get("name", target.stem),
                    "total": len(tests),
                    "passed": 0, "failed": 0,
                    "results": results if verbose else [],
                    "suggested_next_step": (
                        "Run with pytest: pytest "
                        + str(target)
                    ),
                })
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
                return _ok({
                    "total": passed + failed,
                    "passed": passed, "failed": failed,
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
                })
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
            return _ok({
                "total": total,
                "assertions": assertions,
                "domains": domains_found,
                "suggested_next_step": (
                    "Pick an assertion for your tests, "
                    "or run mltk_scan to auto-generate."
                ),
            })
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_eval(
        dataset_path: str,
        scorer: str = "exact_match",
        solver: str = "generate",
    ) -> str:
        """Run an evaluation pipeline on a dataset with configurable solvers and scorers.

        Args:
            dataset_path: Path to CSV/JSON with 'input'
                and 'target' columns.
            scorer: exact_match, includes, or pattern.
            solver: generate, chain_of_thought, few_shot.
        """
        try:
            from mltk.eval import (
                ChainOfThoughtSolver,
                ExactMatchScorer,
                FewShotSolver,
                GenerateSolver,
                IncludesScorer,
                PatternScorer,
            )
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
            solver_map = {
                "generate": GenerateSolver,
                "chain_of_thought": ChainOfThoughtSolver,
                "few_shot": FewShotSolver,
            }
            scorer_map = {
                "exact_match": ExactMatchScorer,
                "includes": IncludesScorer,
                "pattern": PatternScorer,
            }
            sk = solver.strip().lower()
            rk = scorer.strip().lower()
            solver_cls = solver_map.get(
                sk, GenerateSolver
            )
            scorer_cls = scorer_map.get(
                rk, ExactMatchScorer
            )

            def _passthrough(prompt: str) -> str:
                return prompt

            task = EvalTask(
                name="mcp-eval",
                solver=solver_cls(),
                scorers=scorer_cls(),
                dataset=samples,
            )
            result = task.run(_passthrough)
            return _ok({
                "metrics": result.metrics,
                "sample_count": result.total_samples,
                "duration_ms": result.duration_ms,
                "solver": sk, "scorer": rk,
                "suggested_next_step": (
                    "Review metrics. Integrate a real "
                    "model via the Python API."
                ),
            })
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
            return _ok({
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
            })
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
            lines += ["", "---", "*mltk v0.9.0*"]
            report_text = "\n".join(lines)
            n = len(results)
            p = sum(1 for r in results if _is_pass(r))
            summary = (
                f"{n} results: {p} passed, {n-p} failed"
                if n else "No results provided."
            )
            return _ok({
                "report_text": report_text,
                "summary": summary,
                "suggested_next_step": (
                    "Share in PR or CI. Run mltk_scan "
                    "for fresh results."
                ),
            })
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
        """Get fix suggestions for a scan finding.

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
                return _ok({
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
                })
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
            return _ok({
                "suggestions": suggestions,
                "total": len(suggestions),
                "filtered_by": cat or "none",
                "suggested_next_step": (
                    "Apply the highest-confidence fix first, "
                    "then re-scan to verify."
                ),
            })
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))

    @mcp.tool()
    def mltk_experiment(
        finding_json: str,
        rank_by: str = "passed",
        max_results: int = 5,
    ) -> str:
        """Rank fix suggestions for a finding using heuristic scoring.

        Scores each fix based on category actionability, confidence level,
        and code snippet availability. Use after mltk_suggest to prioritize
        which fix to try first.

        Args:
            finding_json: JSON string of a single scan finding
                (from mltk_scan or ScanReport.to_json()).
            rank_by: Strategy: "passed" (confidence-first),
                "delta" (actionability-first), "composite" (balanced).
            max_results: Maximum results to return (1-50).
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
                return _ok({
                    "ranked_fixes": [],
                    "total": 0,
                    "strategy": rank_by,
                    "suggested_next_step": (
                        "No fixes available for this finding. "
                        "Run mltk_suggest first to generate "
                        "fix suggestions."
                    ),
                })
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

            return _ok({
                "ranked_fixes": ranked,
                "total": len(ranked),
                "strategy": strategy,
                "suggested_next_step": (
                    "Apply the top-ranked fix and re-scan "
                    "to verify improvement."
                ),
            })
        except Exception as exc:
            _log(traceback.format_exc())
            return _error(str(exc))


def run_server() -> None:
    """Run the MCP server on stdio transport."""
    server = create_server()
    _log("mltk MCP server starting (stdio)")
    server.run(transport="stdio")
