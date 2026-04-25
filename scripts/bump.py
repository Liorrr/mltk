"""mltk version bump and doc-count refresh tool.

Subcommands:
  refresh              Recompute counts, rewrite stale docs in-place, git-add changes.
  verify               Check for drift, exit non-zero if found (no writes). Used by CI.
  release <version>    Bump version + roll CHANGELOG + run refresh.
  release --dry-run <version>   Preview diff without writing.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "mltk"
CLI_APP = SRC_ROOT / "cli" / "app.py"
MCP_SERVER = SRC_ROOT / "mcp" / "server.py"
SCANNERS_DIR = SRC_ROOT / "scan" / "scanners"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CARGO_TOML = REPO_ROOT / "rust" / "Cargo.toml"
INIT_PY = SRC_ROOT / "__init__.py"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_skill_index import (  # noqa: E402
    collect_assertions,
    collect_cli_commands,
    collect_mcp_tools,
    collect_scanners,
)

# Files where the prose "v0.9.0" release claim should be updated.
# Deliberately excludes BACKLOG.md (has historical sprint refs like "S74: v0.9.0 release").
PROSE_VERSION_TARGETS: list[str] = [
    "README.md",
    "CLAUDE.md",
    "docs/index.md",
    "docs/guides/demo-script.md",
    "docs/guides/container-deployment.md",
]

COUNT_TARGETS: list[str] = [
    "README.md",
    "CLAUDE.md",
    "BACKLOG.md",
    "docs/index.md",
    "docs/roadmap.md",
    "docs/guides/demo-script.md",
    "docs/api/assertion-index.md",
    "docs/api/domain-overview.md",
    "docs/api/test-index.md",
    "docs/api/mcp-server.md",
    "docs/api/llm-judge.md",
    "docs/api/multimodal.md",
    "docs/api/multimodal-rl.md",
]

VERSION_TARGETS: list[str] = [
    "pyproject.toml",
    "rust/Cargo.toml",
    "src/mltk/__init__.py",
]

_VER_RE = re.compile(r'version\s*=\s*"(\d+\.\d+\.\d+)"')
_INIT_VER_RE = re.compile(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"')
_VALID_VER = re.compile(r"^\d+\.\d+\.\d+$")

_CHANGELOG_UNRELEASED = re.compile(
    r"^(##\s*\[Unreleased\]\s*)$", re.MULTILINE
)

_CHANGELOG_STUB = """\
## [Unreleased]

### Added

### Changed

### Fixed

"""


@dataclass
class LiveCounts:
    assertions: int
    cli: int
    mcp: int
    scanners: int
    tests: int
    version: str


def get_test_count() -> int:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(REPO_ROOT / "tests")],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        for line in reversed(output.splitlines()):
            m = re.search(r"(\d[\d,]*)\s+(?:tests?\s+collected|selected)", line)
            if m:
                return int(m.group(1).replace(",", ""))
    except Exception:  # noqa: BLE001
        pass
    # Fallback: count def test_ lines
    count = 0
    for path in (REPO_ROOT / "tests").rglob("*.py"):
        try:
            count += len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), re.MULTILINE))
        except OSError:
            pass
    return count


def get_current_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = _VER_RE.search(text)
    if not m:
        raise ValueError("Could not find version in pyproject.toml")
    return m.group(1)


_cached_test_count: int | None = None


def get_live_counts() -> LiveCounts:
    global _cached_test_count
    grouped, total_assertions = collect_assertions(SRC_ROOT)
    cli_cmds = collect_cli_commands(CLI_APP)
    mcp_tools = collect_mcp_tools(MCP_SERVER)
    scanners = collect_scanners(SCANNERS_DIR)
    if _cached_test_count is None:
        _cached_test_count = get_test_count()
    return LiveCounts(
        assertions=total_assertions,
        cli=len(cli_cmds),
        mcp=len(mcp_tools),
        scanners=len(scanners),
        tests=_cached_test_count,
        version=get_current_version(),
    )


def _has_since_context(text: str, match_start: int) -> bool:
    """Return True if the 30 chars before match_start contain 'Since'."""
    prefix = text[max(0, match_start - 30): match_start]
    return "Since" in prefix


def _replace_counts_in_text(text: str, counts: LiveCounts) -> str:
    """Apply all count replacements to text, skipping Since: context."""

    # assertions: e.g. "224 assertions" or "224+ assertions"
    assertion_re = re.compile(r"\b(\d{3,4}\+?)\s+(assertions?)\b")
    def _replace_assertion(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"{counts.assertions} {m.group(2)}"
    text = assertion_re.sub(_replace_assertion, text)

    # tests badge: tests-3388%2B%20passed-
    badge_re = re.compile(r"tests-(\d[\d,]*)%2B%20passed-")
    def _replace_badge(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"tests-{counts.tests}%2B%20passed-"
    text = badge_re.sub(_replace_badge, text)

    # tests: e.g. "4225+ tests" or "3,388+ tests"
    # Use negative lookbehind for digit/comma to avoid matching inside comma-formatted numbers
    test_re = re.compile(r"(?<![,\d])(\d{1,5}(?:,\d{3})*\+?)\s+(tests?)\b")
    def _replace_tests(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        raw = m.group(1).replace(",", "").rstrip("+")
        try:
            n = int(raw)
        except ValueError:
            return m.group(0)
        if n < 100:
            return m.group(0)
        return f"{counts.tests}+ {m.group(2)}"
    text = test_re.sub(_replace_tests, text)

    # scanners: e.g. "8 scanners"
    scanner_re = re.compile(r"\b(\d)\s+(scanners?)\b")
    def _replace_scanners(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"{counts.scanners} {m.group(2)}"
    text = scanner_re.sub(_replace_scanners, text)

    # MCP tools: e.g. "11 MCP tools"
    mcp_re = re.compile(r"\b(\d{1,2})\s+(MCP\s+tools?)\b")
    def _replace_mcp(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"{counts.mcp} {m.group(2)}"
    text = mcp_re.sub(_replace_mcp, text)

    # CLI commands: e.g. "24 CLI commands" or "24+ CLI commands"
    cli_re = re.compile(r"\b(\d{1,3}\+?)\s+(CLI\s+commands?)\b")
    def _replace_cli(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"{counts.cli} {m.group(2)}"
    text = cli_re.sub(_replace_cli, text)

    return text


def _replace_version_in_text(
    text: str, old_ver: str, new_ver: str, file_rel: str
) -> str:
    """Replace version strings, skipping Since: context."""
    if file_rel == "src/mltk/__init__.py":
        def _sub_init(m: re.Match[str]) -> str:
            if _has_since_context(text, m.start()):
                return m.group(0)
            return m.group(0).replace(m.group(1), new_ver)
        return _INIT_VER_RE.sub(_sub_init, text)

    def _sub_ver(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return m.group(0).replace(m.group(1), new_ver)
    return _VER_RE.sub(_sub_ver, text)


_BACKLOG_HEADER_RE = re.compile(
    r"(## DONE \(S0-S\d+:.*?\))\s*—\s*v[\d.]+(?: / v[\d.]+ pending)?"
)


def _update_backlog_header(text: str, new_ver: str) -> str:
    """Replace the DONE header's version suffix with the new release version."""
    return _BACKLOG_HEADER_RE.sub(rf"\1 — v{new_ver}", text)


def _roll_changelog(text: str, new_ver: str) -> str:
    today = date.today().isoformat()
    new_header = f"## [{new_ver}] — {today}"
    replacement = f"{_CHANGELOG_STUB}{new_header}"
    result, n = _CHANGELOG_UNRELEASED.subn(replacement, text, count=1)
    if n == 0:
        raise ValueError("Could not find '## [Unreleased]' in CHANGELOG.md")
    return result


def _replace_prose_version_in_text(text: str, old_ver: str, new_ver: str) -> str:
    """Replace prose 'v0.9.0' references, skipping Since: context."""
    pattern = re.compile(r"v" + re.escape(old_ver))

    def _sub(m: re.Match[str]) -> str:
        if _has_since_context(text, m.start()):
            return m.group(0)
        return f"v{new_ver}"

    return pattern.sub(_sub, text)


def _unified_diff(path: Path, old: str, new: str) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    lines = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    ))
    return "".join(lines)


def _print_utf8(text: str) -> None:
    """Print text to stdout, forcing UTF-8 on Windows to avoid cp1252 errors."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _git_add(files: list[Path]) -> None:
    try:
        subprocess.run(
            ["git", "add"] + [str(f) for f in files],
            cwd=str(REPO_ROOT),
            capture_output=True,
        )
    except FileNotFoundError:
        pass


def cmd_refresh(dry_run: bool = False, counts: LiveCounts | None = None) -> list[Path]:
    """Rewrite stale count docs. Returns list of modified file paths."""
    if counts is None:
        counts = get_live_counts()
    modified: list[Path] = []

    for rel in COUNT_TARGETS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_counts_in_text(original, counts)
        if updated != original:
            modified.append(path)
            if not dry_run:
                path.write_text(updated, encoding="utf-8")

    return modified


def cmd_verify() -> int:
    counts = get_live_counts()
    drift: list[tuple[str, str, str]] = []

    for rel in COUNT_TARGETS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_counts_in_text(original, counts)
        if updated != original:
            diff = _unified_diff(path, original, updated)
            drift.append((rel, "", diff))

    if drift:
        for rel, _, diff in drift:
            _print_utf8(f"DRIFT: {rel}")
            _print_utf8(diff)
        return 1
    print("OK — all counts current.")
    return 0


def cmd_release(new_ver: str, dry_run: bool) -> int:
    if not _VALID_VER.match(new_ver):
        print(f"ERROR: invalid version format '{new_ver}'. Expected X.Y.Z")
        return 1

    old_ver = get_current_version()
    counts = get_live_counts()
    counts.version = new_ver

    diffs: list[str] = []
    writes: list[tuple[Path, str]] = []

    # Version file updates
    for rel in VERSION_TARGETS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_version_in_text(original, old_ver, new_ver, rel)
        if updated != original:
            diffs.append(_unified_diff(path, original, updated))
            writes.append((path, updated))

    # CHANGELOG roll
    changelog_orig = CHANGELOG.read_text(encoding="utf-8")
    try:
        changelog_new = _roll_changelog(changelog_orig, new_ver)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if changelog_new != changelog_orig:
        diffs.append(_unified_diff(CHANGELOG, changelog_orig, changelog_new))
        writes.append((CHANGELOG, changelog_new))

    # BACKLOG DONE header version
    backlog_path = REPO_ROOT / "BACKLOG.md"
    if backlog_path.exists():
        backlog_orig = backlog_path.read_text(encoding="utf-8")
        backlog_new = _update_backlog_header(backlog_orig, new_ver)
        if backlog_new != backlog_orig:
            diffs.append(_unified_diff(backlog_path, backlog_orig, backlog_new))
            writes.append((backlog_path, backlog_new))

    # Count refresh + prose version bump for doc targets
    count_modified_paths: list[Path] = []
    prose_ver_set = set(PROSE_VERSION_TARGETS)
    for rel in COUNT_TARGETS:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _replace_counts_in_text(original, counts)
        if rel in prose_ver_set:
            updated = _replace_prose_version_in_text(updated, old_ver, new_ver)
        if updated != original:
            diffs.append(_unified_diff(path, original, updated))
            writes.append((path, updated))
            count_modified_paths.append(path)

    if dry_run:
        for diff in diffs:
            _print_utf8(diff)
        return 0

    # Apply all writes
    all_changed: list[Path] = []
    for path, content in writes:
        path.write_text(content, encoding="utf-8")
        all_changed.append(path)

    _git_add(all_changed)

    changed_names = " ".join(p.relative_to(REPO_ROOT).as_posix() for p in all_changed)
    print(f"Refreshed: {changed_names}" if all_changed else "No changes needed.")
    print(f"Remember: git commit && git tag v{new_ver}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh", help="Recompute counts and rewrite stale docs.")
    sub.add_parser("verify", help="Check for drift without writing (CI mode).")

    rel_p = sub.add_parser("release", help="Bump version, roll CHANGELOG, refresh counts.")
    rel_p.add_argument("version", help="New version e.g. 1.0.0")
    rel_p.add_argument("--dry-run", action="store_true", help="Preview diff without writing.")

    args = parser.parse_args()

    if args.cmd == "refresh":
        modified = cmd_refresh()
        if modified:
            _git_add(modified)
            names = " ".join(p.relative_to(REPO_ROOT).as_posix() for p in modified)
            print(f"Refreshed: {names}")
        else:
            print("No changes needed.")
        sys.exit(0)

    elif args.cmd == "verify":
        sys.exit(cmd_verify())

    elif args.cmd == "release":
        sys.exit(cmd_release(args.version, args.dry_run))


if __name__ == "__main__":
    main()
