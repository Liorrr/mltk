"""Generate mltk Claude Code skill index from source code.

Parses the mltk source tree and produces:
1. Compact skill (~200 lines) -> ~/.claude/skills/mltk-index.md
2. Detailed reference (~600-900 lines) -> docs/reference/full-api-index.md

Usage: python scripts/generate_skill_index.py
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# -------------------------------------------------------------------
# Data types
# -------------------------------------------------------------------


@dataclass
class AssertionInfo:
    name: str
    file: str       # relative path from src/mltk/
    line: int


@dataclass
class FileAssertions:
    file: str       # just filename like "drift.py"
    rel_path: str   # relative path like "data/drift.py"
    names: list[str]


@dataclass
class CliCommand:
    name: str       # e.g. "model-card", "contract init"
    line: int
    docstring: str  # first line only


@dataclass
class McpTool:
    name: str
    line: int
    docstring: str
    params: list[tuple[str, str, str]]  # (name, type, default)


@dataclass
class ScannerEntry:
    name: str       # e.g. "BiasScanner"
    file: str       # relative path
    line: int


@dataclass
class KeyClass:
    name: str
    file: str       # relative like "scan/finding.py"
    line: int
    purpose: str    # first line of docstring


@dataclass
class AllData:
    assertions: dict[str, list[FileAssertions]]
    assertion_count: int
    cli_cmds: list[CliCommand]
    cli_groups: list[str]
    mcp_tools: list[McpTool]
    scanners: list[ScannerEntry]
    key_classes: list[KeyClass]
    test_dirs: list[tuple[str, str]]  # (test_dir, src_dir)

    @property
    def cli_summary(self) -> str:
        """Honest CLI header, e.g. ``19 top-level + 5 groups``."""
        return format_cli_summary(self.cli_cmds, self.cli_groups)


# -------------------------------------------------------------------
# Domain grouping (ordered longest-prefix-first)
# -------------------------------------------------------------------

DOMAIN_PREFIXES: list[tuple[str, str]] = [
    ("domains/llm/behavioral", "llm.behavioral"),
    ("domains/llm/red_team", "llm.red_team"),
    ("domains/llm/synthetic", "llm.synthetic"),
    ("domains/llm", "llm"),
    ("domains/cv", "cv"),
    ("domains/nlp", "nlp"),
    ("domains/multimodal", "multimodal"),
    ("domains/speech", "speech"),
    ("domains/tabular", "tabular"),
    ("domains", "domains"),
    ("data", "data"),
    ("model", "model"),
    ("monitor", "monitor"),
    ("pipeline", "pipeline"),
    ("training", "training"),
    ("inference", "inference"),
    ("compliance", "compliance"),
    ("eval", "eval"),
    ("testing", "testing"),
    ("integrations", "integrations"),
    ("core", "core"),
    ("server", "server"),
    ("scan", "scan"),
]


def path_to_domain(rel_path: str) -> str:
    """Map a relative path (posix) to a domain label."""
    for prefix, domain in DOMAIN_PREFIXES:
        if rel_path.startswith(prefix + "/") or rel_path == prefix:
            return domain
    return "other"


# -------------------------------------------------------------------
# Allowlisted key-class files
# -------------------------------------------------------------------

KEY_CLASS_FILES: list[str] = [
    "scan/config.py",
    "scan/finding.py",
    "scan/engine.py",
    "experiment/hypothesis.py",
    "experiment/result.py",
    "experiment/worktree.py",
    "core/result.py",
    "core/suite.py",
    "eval/dataset.py",
    "eval/task.py",
    "domains/llm/trace.py",
    "domains/llm/mcp.py",
    "domains/llm/behavioral/paraphrase.py",
    "contracts/schema.py",
]

REGULAR_KEY_CLASSES: set[str] = {
    "GitWorktree",
    "ExperimentRunner",
    "MltkSuite",
    "EvalTask",
    "ParaphraseGenerator",
    "ScanEngine",
}


# -------------------------------------------------------------------
# Collectors
# -------------------------------------------------------------------

_ASSERT_RE = re.compile(r"^def (assert_\w+)\(", re.MULTILINE)


def collect_assertions(
    src_root: Path,
) -> tuple[dict[str, list[FileAssertions]], int]:
    """Walk src_root for assert_* functions, grouped by domain."""
    grouped: dict[str, list[FileAssertions]] = {}
    total = 0

    for py in sorted(src_root.rglob("*.py")):
        rel = py.relative_to(src_root)
        rel_posix = rel.as_posix()
        if "__pycache__" in rel_posix or rel.name == "__init__.py":
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        names = _ASSERT_RE.findall(text)
        if not names:
            continue
        domain = path_to_domain(rel_posix)
        fa = FileAssertions(
            file=rel.name,
            rel_path=rel_posix,
            names=sorted(names),
        )
        grouped.setdefault(domain, []).append(fa)
        total += len(names)

    # Sort domains and file lists
    sorted_grouped: dict[str, list[FileAssertions]] = {}
    for domain in sorted(grouped):
        sorted_grouped[domain] = sorted(
            grouped[domain], key=lambda fa: fa.rel_path
        )
    return sorted_grouped, total


def _decorator_attr(node: ast.FunctionDef, attr_name: str) -> str | None:
    """If node has a decorator like ``X.command(...)`` return X's id."""
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == attr_name
        ):
            if isinstance(func.value, ast.Name):
                return func.value.id
    return None


_APP_PREFIX_MAP: dict[str, str] = {
    "app": "",
    "contract_app": "contract ",
    "docs_app": "docs ",
    "registry_app": "registry ",
    "notify_app": "notify ",
}


def _imported_function_docstring(
    app_py: Path, module: str | None, fn_name: str
) -> str:
    """First docstring line of *fn_name* defined in dotted *module*."""
    if not module or not module.startswith("mltk."):
        return ""
    rel = Path(*module.split(".")[1:]).with_suffix(".py")
    candidate = app_py.parent.parent / rel
    try:
        tree = ast.parse(candidate.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            doc = ast.get_docstring(node) or ""
            return doc.split("\n")[0].strip() if doc else ""
    return ""


def _import_alias_map(tree: ast.Module) -> dict[str, str]:
    """Map local import aliases to fully-qualified module paths."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module
    return imports


def _collect_call_registrations(
    tree: ast.Module, app_py: Path
) -> list[CliCommand]:
    """Find call-style registrations like ``app.command(name="import")(fn)``.

    Commands whose implementation lives in another module (e.g.
    ``mltk.cli.importer``) are registered as an expression, not a
    decorator, so the decorator scan cannot see them.
    """
    imports = _import_alias_map(tree)

    cmds: list[CliCommand] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Call)
        ):
            continue
        inner = call.func
        if not (
            isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "command"
            and isinstance(inner.func.value, ast.Name)
        ):
            continue
        if not (call.args and isinstance(call.args[0], ast.Name)):
            continue
        fn_name = call.args[0].id

        cmd_name: str | None = None
        for kw in inner.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                cmd_name = str(kw.value.value)
        if (
            cmd_name is None
            and inner.args
            and isinstance(inner.args[0], ast.Constant)
        ):
            cmd_name = str(inner.args[0].value)
        if cmd_name is None:
            cmd_name = fn_name.replace("_", "-")

        prefix = _APP_PREFIX_MAP.get(inner.func.value.id, "")
        cmds.append(CliCommand(
            name=prefix + cmd_name,
            line=node.lineno,
            docstring=_imported_function_docstring(
                app_py, imports.get(fn_name), fn_name
            ),
        ))
    return cmds


def _typer_app_name(tree: ast.Module) -> str | None:
    """Return Typer ``name=`` for the first ``X = typer.Typer(...)`` assign."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_typer = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "typer"
            and func.attr == "Typer"
        ) or (isinstance(func, ast.Name) and func.id == "Typer")
        if not is_typer:
            continue
        for kw in node.value.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
        # Fall back to assigned variable name
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    return None


def _local_typer_var_names(tree: ast.AST) -> dict[str, str]:
    """Map local ``X = typer.Typer(name=...)`` vars to group names.

    Walks nested scopes (CLI registers groups inside ``main()``).
    """
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_typer = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "typer"
            and func.attr == "Typer"
        )
        if not is_typer:
            continue
        group: str | None = None
        for kw in node.value.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                group = str(kw.value.value)
                break
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Skip the root app (name="mltk")
                if target.id == "app":
                    continue
                result[target.id] = (
                    group or target.id.removesuffix("_app")
                )
    return result


def _collect_external_typer_commands(
    tree: ast.Module, app_py: Path
) -> tuple[list[CliCommand], list[str]]:
    """Collect groups + subcommands from ``app.add_typer(...)``.

    In-file groups (``contract_app = typer.Typer(...)``) contribute group
    names only — their subcommands are already covered by the decorator
    scan. External apps such as ``from mltk.cli.container import app as
    container_app`` also contribute leaf command paths.
    """
    imports = _import_alias_map(tree)
    # Nested imports inside main() (e.g. container_app)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module

    local_typers = _local_typer_var_names(tree)

    group_names: list[str] = []
    cmds: list[CliCommand] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(
            node.value, ast.Call
        ):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_typer"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "app"
            and call.args
            and isinstance(call.args[0], ast.Name)
        ):
            continue
        alias = call.args[0].id
        # Explicit name= on add_typer takes precedence
        explicit_group: str | None = None
        for kw in call.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                explicit_group = str(kw.value.value)

        if alias in local_typers:
            group = explicit_group or local_typers[alias]
            if group and group not in group_names:
                group_names.append(group)
            continue

        module = imports.get(alias)
        if not module or not module.startswith("mltk."):
            continue
        rel = Path(*module.split(".")[1:]).with_suffix(".py")
        candidate = app_py.parent.parent / rel
        try:
            ext_tree = ast.parse(candidate.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        group = explicit_group or _typer_app_name(ext_tree) or alias
        if group not in group_names:
            group_names.append(group)
        prefix = f"{group} "
        for ext_node in ast.walk(ext_tree):
            if not isinstance(ext_node, ast.FunctionDef):
                continue
            caller_id = _decorator_attr(ext_node, "command")
            if caller_id is None:
                continue
            cmd_name: str | None = None
            for dec in ext_node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                if (
                    isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "command"
                ):
                    if dec.args and isinstance(dec.args[0], ast.Constant):
                        cmd_name = str(dec.args[0].value)
                    for kw in dec.keywords:
                        if (
                            kw.arg == "name"
                            and isinstance(kw.value, ast.Constant)
                        ):
                            cmd_name = str(kw.value.value)
                    break
            if cmd_name is None:
                cmd_name = ext_node.name.replace("_", "-")
            doc = ast.get_docstring(ext_node) or ""
            first_line = doc.split("\n")[0].strip() if doc else ""
            cmds.append(CliCommand(
                name=prefix + cmd_name,
                line=ext_node.lineno,
                docstring=first_line,
            ))
    return cmds, group_names


def collect_cli_commands(app_py: Path) -> list[CliCommand]:
    """Parse CLI leaf commands from app.py (+ external add_typer modules).

    Returns every invocable leaf path (e.g. ``scan``, ``contract init``,
    ``container scan``). Prefer :func:`format_cli_summary` for public counts
    — do not report ``len(result)`` as "N CLI commands".
    """
    try:
        tree = ast.parse(app_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    cmds: list[CliCommand] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        caller_id = _decorator_attr(node, "command")
        if caller_id is None:
            continue

        prefix = _APP_PREFIX_MAP.get(caller_id, "")

        # Extract explicit command name from decorator arg
        cmd_name: str | None = None
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            if (
                isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "command"
            ):
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    cmd_name = str(dec.args[0].value)
                break

        if cmd_name is None:
            cmd_name = node.name.replace("_", "-")

        full_name = prefix + cmd_name
        doc = ast.get_docstring(node) or ""
        first_line = doc.split("\n")[0].strip() if doc else ""
        cmds.append(CliCommand(
            name=full_name,
            line=node.lineno,
            docstring=first_line,
        ))

    cmds.extend(_collect_call_registrations(tree, app_py))
    ext_cmds, _groups = _collect_external_typer_commands(tree, app_py)
    cmds.extend(ext_cmds)
    return sorted(cmds, key=lambda c: (c.name, c.line))


def collect_cli_groups(app_py: Path) -> list[str]:
    """Return Typer group names registered via ``app.add_typer(...)``."""
    try:
        tree = ast.parse(app_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    _ext_cmds, groups = _collect_external_typer_commands(tree, app_py)
    return groups


def cli_top_level_count(cmds: list[CliCommand]) -> int:
    """Count top-level leaf commands (no group prefix / space in name)."""
    return sum(1 for c in cmds if " " not in c.name)


def format_cli_summary(
    cmds: list[CliCommand], groups: list[str] | None = None
) -> str:
    """Honest CLI wording for headers: ``19 top-level + 5 groups``."""
    top = cli_top_level_count(cmds)
    if groups is None:
        groups = sorted({
            c.name.split(" ", 1)[0]
            for c in cmds
            if " " in c.name
        })
    n_groups = len(groups)
    return f"{top} top-level + {n_groups} groups"


def collect_mcp_tools(server_py: Path) -> list[McpTool]:
    """Parse MCP tool registrations from server.py using AST."""
    try:
        tree = ast.parse(server_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    tools: list[McpTool] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        caller_id = _decorator_attr(node, "tool")
        if caller_id is None:
            continue

        doc = ast.get_docstring(node) or ""
        first_line = doc.split("\n")[0].strip() if doc else ""

        # Collect parameters
        params: list[tuple[str, str, str]] = []
        args_node = node.args
        defaults_offset = (
            len(args_node.args) - len(args_node.defaults)
        )
        for i, arg in enumerate(args_node.args):
            name = arg.arg
            # Skip context-like first params (self, ctx)
            if i == 0 and name in ("self", "ctx", "context"):
                continue
            ann = ast.unparse(arg.annotation) if arg.annotation else ""
            default_idx = i - defaults_offset
            if default_idx >= 0:
                default = ast.unparse(
                    args_node.defaults[default_idx]
                )
            else:
                default = ""
            params.append((name, ann, default))

        tools.append(McpTool(
            name=node.name,
            line=node.lineno,
            docstring=first_line,
            params=params,
        ))

    return sorted(tools, key=lambda t: t.line)


def collect_scanners(scanners_dir: Path) -> list[ScannerEntry]:
    """Find Scanner subclasses in the scanners directory."""
    entries: list[ScannerEntry] = []
    for py in sorted(scanners_dir.glob("*.py")):
        if py.name in ("base.py", "__init__.py"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        rel = py.relative_to(scanners_dir.parent.parent)
        rel_posix = rel.as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = [
                b.id for b in node.bases if isinstance(b, ast.Name)
            ]
            if "Scanner" in base_names:
                entries.append(ScannerEntry(
                    name=node.name,
                    file=rel_posix,
                    line=node.lineno,
                ))
    return sorted(entries, key=lambda e: e.name)


def _has_dataclass_decorator(node: ast.ClassDef) -> bool:
    """Check whether a ClassDef has a @dataclass decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "dataclass"
        ):
            return True
    return False


def collect_key_classes(src_root: Path) -> list[KeyClass]:
    """Collect important dataclasses and named classes."""
    results: list[KeyClass] = []
    seen: set[tuple[str, str]] = set()

    for rel_file in KEY_CLASS_FILES:
        full = src_root / rel_file
        if not full.exists():
            print(
                f"[warn] key-class file not found: {rel_file}",
                file=sys.stderr,
            )
            continue
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue

            is_dc = _has_dataclass_decorator(node)
            is_named = node.name in REGULAR_KEY_CLASSES

            if not (is_dc or is_named):
                continue

            key = (node.name, rel_file)
            if key in seen:
                continue
            seen.add(key)

            doc = ast.get_docstring(node) or ""
            first_sentence = doc.split("\n")[0].strip() if doc else ""
            # Trim to first sentence (period-terminated)
            dot_idx = first_sentence.find(".")
            if dot_idx > 0:
                first_sentence = first_sentence[: dot_idx + 1]

            results.append(KeyClass(
                name=node.name,
                file=rel_file,
                line=node.lineno,
                purpose=first_sentence,
            ))

    return sorted(results, key=lambda kc: (kc.file, kc.line))


def collect_test_layout(tests_dir: Path) -> list[tuple[str, str]]:
    """Map test_* directories to their source counterparts."""
    if not tests_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for child in sorted(tests_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("test_"):
            continue
        src_module = name.removeprefix("test_")
        pairs.append((name, f"src/mltk/{src_module}/"))
    return pairs


# -------------------------------------------------------------------
# Detailed assertion signatures (second AST pass)
# -------------------------------------------------------------------

@dataclass
class AssertionSignature:
    name: str
    signature: str
    rel_path: str
    line: int
    docstring_first: str


def _collect_assertion_signatures(
    src_root: Path,
    grouped: dict[str, list[FileAssertions]],
) -> dict[str, list[AssertionSignature]]:
    """Second AST pass: full signatures for assertions."""
    result: dict[str, list[AssertionSignature]] = {}

    for domain, file_assertions_list in sorted(grouped.items()):
        sigs: list[AssertionSignature] = []
        for fa in file_assertions_list:
            full = src_root / fa.rel_path
            if not full.exists():
                continue
            try:
                tree = ast.parse(full.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            name_set = set(fa.names)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name not in name_set:
                    continue
                try:
                    sig = ast.unparse(node.args)
                except Exception:  # noqa: BLE001
                    sig = "..."
                doc = ast.get_docstring(node) or ""
                first = doc.split("\n")[0].strip() if doc else ""
                sigs.append(AssertionSignature(
                    name=node.name,
                    signature=sig,
                    rel_path=fa.rel_path,
                    line=node.lineno,
                    docstring_first=first,
                ))
        if sigs:
            result[domain] = sorted(sigs, key=lambda s: s.name)

    return result


# -------------------------------------------------------------------
# Detailed dataclass fields
# -------------------------------------------------------------------

@dataclass
class ClassField:
    name: str
    annotation: str
    default: str


def _collect_dataclass_fields(
    src_root: Path,
    key_classes: list[KeyClass],
) -> dict[str, list[ClassField]]:
    """Extract fields from dataclass definitions."""
    result: dict[str, list[ClassField]] = {}

    for kc in key_classes:
        full = src_root / kc.file
        if not full.exists():
            continue
        try:
            tree = ast.parse(full.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != kc.name:
                continue
            if not _has_dataclass_decorator(node):
                continue
            fields: list[ClassField] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.target:
                    fname = ast.unparse(stmt.target)
                    fann = (
                        ast.unparse(stmt.annotation)
                        if stmt.annotation
                        else ""
                    )
                    fdef = (
                        ast.unparse(stmt.value)
                        if stmt.value
                        else ""
                    )
                    fields.append(ClassField(
                        name=fname,
                        annotation=fann,
                        default=fdef,
                    ))
            if fields:
                result[kc.name] = fields
    return result


# -------------------------------------------------------------------
# Formatting: compact skill
# -------------------------------------------------------------------

def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return 0


def _fmt_assertion_line(fa: FileAssertions) -> str:
    """Format one file's assertions as a compact line."""
    prefix = f"- {fa.file}: "
    names_str = ", ".join(fa.names)
    line = prefix + names_str
    if len(line) <= 120:
        return line
    # Wrap naturally
    return prefix + names_str


def fmt_compact(data: AllData) -> str:
    """Generate ~200 line compact skill markdown."""
    today = date.today().isoformat()
    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append("description: >")
    lines.append(
        f"  mltk codebase index -- {data.assertion_count} assertions,"
        f" {len(data.mcp_tools)} MCP tools,"
        f" {data.cli_summary} CLI,"
        f" {len(data.scanners)} scanners."
        " For full signatures: docs/reference/full-api-index.md"
    )
    lines.append("---")
    lines.append("")
    lines.append("# mltk Codebase Index")
    lines.append(f"> Generated {today} by scripts/generate_skill_index.py")
    lines.append("")

    # Assertions by domain
    lines.append(f"## Assertions by Domain ({data.assertion_count})")
    lines.append("")
    for domain, file_list in sorted(data.assertions.items()):
        count = sum(len(fa.names) for fa in file_list)
        lines.append(f"### {domain} ({count})")
        for fa in file_list:
            lines.append(_fmt_assertion_line(fa))
        lines.append("")

    # MCP Tools
    lines.append(f"## MCP Tools ({len(data.mcp_tools)})")
    lines.append("")
    lines.append("| Tool | File:Line | Purpose |")
    lines.append("|------|-----------|---------|")
    for tool in data.mcp_tools:
        lines.append(
            f"| {tool.name} | server.py:{tool.line}"
            f" | {tool.docstring} |"
        )
    lines.append("")

    # CLI Commands (honest: top-level + groups, not raw leaf-path count)
    lines.append(f"## CLI Commands ({data.cli_summary})")
    lines.append("")
    lines.append(
        f"Groups: {', '.join(data.cli_groups) if data.cli_groups else 'none'}"
    )
    lines.append("")
    lines.append("| Command | File:Line |")
    lines.append("|---------|-----------|")
    for cmd in data.cli_cmds:
        # External typer modules keep their own filename in path via name
        file_hint = "app.py"
        if " " in cmd.name:
            group = cmd.name.split(" ", 1)[0]
            if group not in {
                "contract", "docs", "registry", "notify",
            }:
                file_hint = f"{group}.py"
        lines.append(f"| {cmd.name} | {file_hint}:{cmd.line} |")
    lines.append("")

    # Scanners
    lines.append(f"## Scanners ({len(data.scanners)})")
    lines.append("")
    lines.append("| Scanner | File |")
    lines.append("|---------|------|")
    for sc in data.scanners:
        lines.append(f"| {sc.name} | {sc.file} |")
    lines.append("")

    # Key Classes
    lines.append("## Key Classes")
    lines.append("")
    lines.append("| Class | File:Line | Purpose |")
    lines.append("|-------|-----------|---------|")
    for kc in data.key_classes:
        lines.append(
            f"| {kc.name} | {kc.file}:{kc.line}"
            f" | {kc.purpose} |"
        )
    lines.append("")

    # Test Layout
    lines.append("## Test Layout")
    lines.append("")
    for test_dir, src_dir in data.test_dirs:
        lines.append(f"- {test_dir}/ -> {src_dir}")
    lines.append("")

    lines.append(
        "> Full signatures: docs/reference/full-api-index.md"
    )
    lines.append("")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Formatting: detailed reference
# -------------------------------------------------------------------


def fmt_detailed(data: AllData) -> str:
    """Generate ~600-900 line detailed reference markdown."""
    today = date.today().isoformat()
    src_root = (
        Path(__file__).resolve().parent.parent / "src" / "mltk"
    )

    lines: list[str] = []
    lines.append("# mltk Full API Index")
    lines.append(f"> Generated {today} by scripts/generate_skill_index.py")
    lines.append("")
    lines.append(
        f"**{data.assertion_count}** assertions | "
        f"**{len(data.mcp_tools)}** MCP tools | "
        f"**{data.cli_summary}** CLI | "
        f"**{len(data.scanners)}** scanners"
    )
    lines.append("")

    # --- Full assertion signatures ---
    lines.append("---")
    lines.append("")
    lines.append(
        f"## Assertion Signatures ({data.assertion_count})"
    )
    lines.append("")

    sig_map = _collect_assertion_signatures(src_root, data.assertions)
    for domain in sorted(sig_map):
        sigs = sig_map[domain]
        lines.append(f"### {domain}")
        lines.append("")
        for sig in sigs:
            lines.append(
                f"**`{sig.name}`** "
                f"({sig.rel_path}:{sig.line})"
            )
            lines.append("```python")
            lines.append(f"def {sig.name}({sig.signature})")
            lines.append("```")
            if sig.docstring_first:
                lines.append(f"> {sig.docstring_first}")
            lines.append("")
    lines.append("")

    # --- MCP tool parameter schemas ---
    lines.append("---")
    lines.append("")
    lines.append(f"## MCP Tools ({len(data.mcp_tools)})")
    lines.append("")
    for tool in data.mcp_tools:
        lines.append(f"### `{tool.name}` (server.py:{tool.line})")
        lines.append("")
        lines.append(f"> {tool.docstring}")
        lines.append("")
        if tool.params:
            lines.append("| Param | Type | Default |")
            lines.append("|-------|------|---------|")
            for pname, ptype, pdefault in tool.params:
                default_str = (
                    f"`{pdefault}`" if pdefault else "*required*"
                )
                type_str = f"`{ptype}`" if ptype else ""
                lines.append(
                    f"| {pname} | {type_str} | {default_str} |"
                )
        else:
            lines.append("*No parameters.*")
        lines.append("")
    lines.append("")

    # --- CLI command details ---
    lines.append("---")
    lines.append("")
    lines.append(f"## CLI Commands ({data.cli_summary})")
    lines.append("")
    lines.append(
        f"Typer groups: "
        f"{', '.join(f'`{g}`' for g in data.cli_groups) if data.cli_groups else 'none'}"
    )
    lines.append("")
    lines.append(
        "Listed rows are invocable leaf paths (top-level commands and "
        "group subcommands). Prefer the summary count above over the "
        "raw row count."
    )
    lines.append("")
    lines.append("| # | Command | Line | Description |")
    lines.append("|---|---------|------|-------------|")
    for i, cmd in enumerate(data.cli_cmds, 1):
        lines.append(
            f"| {i} | `mltk {cmd.name}` | {cmd.line}"
            f" | {cmd.docstring} |"
        )
    lines.append("")

    # --- Scanners ---
    lines.append("---")
    lines.append("")
    lines.append(f"## Scanners ({len(data.scanners)})")
    lines.append("")
    lines.append("| Scanner | File | Line |")
    lines.append("|---------|------|------|")
    for sc in data.scanners:
        lines.append(f"| {sc.name} | {sc.file} | {sc.line} |")
    lines.append("")

    # --- Key classes with dataclass fields ---
    lines.append("---")
    lines.append("")
    lines.append(f"## Key Classes ({len(data.key_classes)})")
    lines.append("")

    dc_fields = _collect_dataclass_fields(src_root, data.key_classes)
    for kc in data.key_classes:
        lines.append(
            f"### `{kc.name}` ({kc.file}:{kc.line})"
        )
        if kc.purpose:
            lines.append(f"> {kc.purpose}")
        lines.append("")
        fields = dc_fields.get(kc.name)
        if fields:
            lines.append("| Field | Type | Default |")
            lines.append("|-------|------|---------|")
            for fld in fields:
                default_str = (
                    f"`{fld.default}`" if fld.default else ""
                )
                lines.append(
                    f"| {fld.name} | `{fld.annotation}`"
                    f" | {default_str} |"
                )
            lines.append("")
        else:
            lines.append("*(no dataclass fields)*")
            lines.append("")

    # --- Test layout ---
    lines.append("---")
    lines.append("")
    lines.append("## Test Layout")
    lines.append("")
    lines.append("| Test Directory | Source Module |")
    lines.append("|----------------|---------------|")
    for test_dir, src_dir in data.test_dirs:
        lines.append(f"| {test_dir}/ | {src_dir} |")
    lines.append("")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src" / "mltk"

    assertions, count = collect_assertions(src_root)
    app_py = src_root / "cli" / "app.py"
    cli_cmds = collect_cli_commands(app_py)
    cli_groups = collect_cli_groups(app_py)
    mcp_tools = collect_mcp_tools(src_root / "mcp" / "server.py")
    scanners = collect_scanners(src_root / "scan" / "scanners")
    key_classes = collect_key_classes(src_root)
    test_dirs = collect_test_layout(repo_root / "tests")

    data = AllData(
        assertions, count, cli_cmds, cli_groups, mcp_tools,
        scanners, key_classes, test_dirs,
    )

    # Write compact skill
    skill_path = Path.home() / ".claude" / "skills" / "mltk-index.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(fmt_compact(data), encoding="utf-8")

    # Write detailed reference
    detail_path = (
        repo_root / "docs" / "reference" / "full-api-index.md"
    )
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(fmt_detailed(data), encoding="utf-8")

    # Install repo skills to ~/.claude/skills/
    import shutil
    skills_dir = repo_root / "skills"
    installed_skills: list[Path] = []
    if skills_dir.exists():
        for src in sorted(skills_dir.glob("mltk-*.md")):
            dst = skill_path.parent / src.name
            shutil.copy2(src, dst)
            installed_skills.append(dst)

    print(  # noqa: T201
        f"Compact skill: {skill_path}"
        f" ({_line_count(skill_path)} lines)"
    )
    print(  # noqa: T201
        f"Detailed ref:  {detail_path}"
        f" ({_line_count(detail_path)} lines)"
    )
    for dst in installed_skills:
        print(  # noqa: T201
            f"Skill:         {dst}"
            f" ({_line_count(dst)} lines)"
        )
    print(  # noqa: T201
        f"Assertions: {count}"
        f" | CLI: {data.cli_summary}"
        f" (leaves={len(cli_cmds)})"
        f" | MCP: {len(mcp_tools)}"
        f" | Scanners: {len(scanners)}"
    )


if __name__ == "__main__":
    main()
