"""Run CI checks locally and require green branch CI before opening a PR.

Local checks are not a substitute for the OS/Python matrix. The ready command
checks the pushed commit, current master ancestry, and the latest branch CI run.
This script never commits, pushes, or modifies Git configuration. Only the
explicit open command creates a PR, after the ready gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_INSTALL = [
    "-e", ".[dev,scipy,sklearn,importer,cli,server,report,multimodal,onnx]", "onnx",
]
INSTALLS = {
    "lint": ["ruff"],
    "claim-regression": ["-e", ".[dev]"],
    "test": CI_INSTALL,
    "docs-freshness": ["-e", ".[dev]"],
}
COMMANDS = {
    "lint": [
        ["python", "-m", "ruff", "check", "src/", "tests/"],
        ["python", "-m", "ruff", "check", "scripts/preflight.py",
         "scripts/tests/test_preflight.py"],
    ],
    "claim-regression": [["pytest", "tests/regression", "-q"]],
    "test": [
        # Deliberately use the console entry point, just as in CI. `python -m`
        # adds cwd to sys.path and has hidden collection failures in the past.
        ["pytest", "--collect-only", "-q"],
        ["pytest", "scripts/tests", "-q"],
        ["pytest", "--cov=mltk", "--cov-report=xml", "--cov-fail-under=60", "-q"],
    ],
    "docs-freshness": [["python", "scripts/bump.py", "verify", "--skip-test-count"]],
    "rust": [
        ["cargo", "fmt", "--check"],
        ["cargo", "clippy", "--", "-D", "warnings"],
        ["cargo", "test"],
    ],
}
REQUIRED_JOBS = {"lint", "claim-regression", "docs-freshness", "security"} | {
    f"test ({platform}, {version})"
    for platform in ("ubuntu-latest", "macos-latest", "windows-latest")
    for version in ("3.10", "3.12", "3.13")
} | {f"rust ({platform})" for platform in ("ubuntu-latest", "macos-latest", "windows-latest")}


def clean_env(python: Path) -> dict[str, str]:
    """Prevent local import and pytest overrides from concealing CI failures."""
    env = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE",
        "PYTHONOPTIMIZE", "PYTHONUTF8", "PYTHONIOENCODING", "PYTHONWARNINGS",
        "PIP_CONSTRAINT",
    ):
        env.pop(name, None)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(args: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+ " + subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8").strip()


def job(name: str, python: Path, *, install: bool = False) -> None:
    env = clean_env(python)
    if install and name in INSTALLS:
        run([str(python), "-m", "pip", "install", *INSTALLS[name]], env=env)
    if name in {"claim-regression", "test", "docs-freshness"}:
        # A stale editable install can silently test a different worktree.
        probe = (
            "from pathlib import Path; import mltk; "
            f"expected = Path({str(ROOT / 'src/mltk/__init__.py')!r}).resolve(); "
            "actual = Path(mltk.__file__).resolve(); "
            "assert actual == expected, f'Wrong checkout: {actual} != {expected}'; "
            "print('Testing', actual, mltk.__version__)"
        )
        run([str(python), "-c", probe], env=env)
    for command in COMMANDS[name]:
        args = list(command)
        if args[0] == "python":
            args[0] = str(python)
        elif args[0] == "pytest":
            executable = scripts_directory(python) / (
                "pytest.exe" if os.name == "nt" else "pytest"
            )
            if not executable.is_file():
                raise RuntimeError(f"Missing {executable}; install this job's dependencies first")
            args[0] = str(executable)
        run(args, cwd=ROOT / "rust" if name == "rust" else ROOT, env=env)


def scripts_directory(python: Path) -> Path:
    # setup-python's Windows interpreter is outside Scripts/, unlike a venv.
    if python == Path(sys.executable):
        return Path(sysconfig.get_path("scripts"))
    # local() supplies newly created venv interpreters only.
    return python.parent


def local() -> None:
    """Use fresh minimal and full environments, never the developer's extras."""
    if sys.version_info[:2] not in {(3, 10), (3, 12), (3, 13)}:
        raise RuntimeError("Use a CI interpreter: Python 3.10, 3.12 (recommended), or 3.13")
    if not shutil.which("cargo"):
        raise RuntimeError("Install Rust with rustfmt and clippy before running preflight")
    run(["cargo", "fmt", "--version"])
    run(["cargo", "clippy", "--version"])
    if sys.version_info[:2] != (3, 12):
        print("Local checks use this Python; CI also runs its fixed Python 3.12 jobs.")
    print("Local checks include working-tree files; remote CI validates the pushed commit.")
    with tempfile.TemporaryDirectory(
        prefix="mltk-preflight-", ignore_cleanup_errors=True,
    ) as temporary:
        base = Path(temporary)
        for label, jobs in (
            ("lint", ["lint"]),
            ("minimal", ["claim-regression", "docs-freshness"]),
            ("full", ["test"]),
        ):
            directory = base / label
            run([sys.executable, "-m", "venv", str(directory)])
            python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            for index, name in enumerate(jobs):
                job(name, python, install=index == 0)
        job("rust", Path(sys.executable))
    print("Local preflight passed. The remote matrix and security jobs are still required.")


def select_run(runs: list[dict], sha: str, branch: str) -> dict:
    """Require the newest CI attempt for this branch and exact commit."""
    candidates = [
        entry for entry in runs
        if entry.get("headSha") == sha and entry.get("headBranch") == branch
        and entry.get("event") == "push" and entry.get("workflowName") == "CI"
    ]
    if not candidates:
        raise RuntimeError("No branch CI run for this commit. Push and wait for CI first.")
    latest = max(candidates, key=lambda entry: int(entry["databaseId"]))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise RuntimeError(
            f"Latest branch CI is {latest.get('status')}/{latest.get('conclusion')}: "
            f"{latest.get('url', '')}"
        )
    return latest


def verify_jobs(jobs: list[dict]) -> None:
    missing = REQUIRED_JOBS - {entry["name"] for entry in jobs}
    failed = [
        entry["name"] for entry in jobs
        if entry.get("status") != "completed" or entry.get("conclusion") != "success"
    ]
    if missing or failed:
        raise RuntimeError(f"CI jobs missing: {sorted(missing)}; not successful: {failed}")


def ready() -> dict[str, str]:
    """Read-only PR gate; no old green SHA or untested local changes qualify."""
    root = Path(capture(["git", "rev-parse", "--show-toplevel"])).resolve()
    if root != ROOT.resolve():
        raise RuntimeError("Run ready from a real checkout, not a source export")
    if capture(["git", "status", "--porcelain", "--untracked-files=normal"]):
        raise RuntimeError("Working tree is dirty. Commit and push the final changes first.")
    branch = capture(["git", "symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch == "master":
        raise RuntimeError("Open PRs from a feature branch, not master")
    sha = capture(["git", "rev-parse", "HEAD"])
    origin = capture(["git", "remote", "get-url", "origin"])
    info = json.loads(capture([
        "gh", "repo", "view", origin, "--json", "nameWithOwner,isFork,parent",
    ]))
    if info.get("isFork"):
        # A fork's master is not the PR base, and gh pr create would target the fork.
        parent = info.get("parent") or {}
        upstream = "/".join(
            part for part in ((parent.get("owner") or {}).get("login"), parent.get("name"))
            if part
        ) or "the upstream repository"
        raise RuntimeError(
            f"origin is a fork of {upstream}. ready and open only verify branches pushed "
            f"to {upstream} itself. From a fork, run local, push to your fork, and open "
            f"the PR on {upstream} directly; its PR CI checks the merge result.",
        )
    repository = info["nameWithOwner"]
    # Query the actual remote, not potentially stale origin/* tracking refs.
    refs = capture([
        "git", "ls-remote", "--heads", "origin", "refs/heads/master", f"refs/heads/{branch}",
    ])
    remote = dict(line.split()[::-1] for line in refs.splitlines())
    if remote.get(f"refs/heads/{branch}") != sha:
        raise RuntimeError("Remote branch differs from HEAD. Push the final commit first.")
    master = remote.get("refs/heads/master")
    if not master:
        raise RuntimeError("Cannot resolve origin/master")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", master, sha], cwd=ROOT, check=False,
    )
    if ancestry.returncode:
        raise RuntimeError("Incorporate current origin/master, then rerun preflight and branch CI.")
    runs = json.loads(capture([
        "gh", "run", "list", "--repo", repository, "--workflow", "ci.yml", "--branch", branch,
        "--commit", sha, "--event", "push", "--limit", "100", "--json",
        "databaseId,headSha,headBranch,event,workflowName,status,conclusion,url",
    ]))
    result = select_run(runs, sha, branch)
    details = json.loads(capture([
        "gh", "run", "view", str(result["databaseId"]), "--repo", repository,
        "--json", "jobs,headSha",
    ]))
    if details.get("headSha") != sha:
        raise RuntimeError("CI evidence changed during verification. Rerun ready.")
    verify_jobs(details["jobs"])
    # Refuse to reuse evidence if HEAD or the worktree changed while querying CI.
    if capture(["git", "rev-parse", "HEAD"]) != sha or capture([
        "git", "status", "--porcelain", "--untracked-files=normal",
    ]):
        raise RuntimeError("Checkout changed during verification. Rerun ready.")
    if capture([
        "git", "ls-remote", "--heads", "origin", "refs/heads/master", f"refs/heads/{branch}",
    ]) != refs:
        raise RuntimeError("Remote branch or master changed during verification. Rerun ready.")
    print(f"PR ready: {sha}\nCI: {result['url']}\nBase verified: {master}")
    print("You may now open the PR. PR merge-result CI remains required.")
    return {"repository": repository, "branch": branch, "sha": sha,
            "url": result["url"], "base": master}


def open_pr(title: str, body_file: Path) -> None:
    if not body_file.is_file():
        raise RuntimeError(f"PR body file does not exist: {body_file}")
    body = body_file.read_text(encoding="utf-8")
    evidence = ready()
    body += (
        "\n\n## Verified preflight evidence\n\n"
        f"- Commit: `{evidence['sha']}`\n"
        f"- Branch CI: {evidence['url']}\n"
        f"- Base master: `{evidence['base']}`\n"
    )
    with tempfile.TemporaryDirectory(prefix="mltk-pr-body-") as temporary:
        prepared = Path(temporary) / "body.md"
        prepared.write_text(body, encoding="utf-8")
        run([
            "gh", "pr", "create", "--repo", evidence["repository"], "--base", "master",
            "--head", evidence["branch"], "--title", title, "--body-file", str(prepared),
        ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("local", help="Fresh environments; CI commands on this OS/Python")
    checks = sub.add_parser("job", help="Run a shared CI job in the current interpreter")
    checks.add_argument("name", choices=COMMANDS)
    checks.add_argument("--install", action="store_true")
    sub.add_parser("ready", help="Require a clean, pushed HEAD with green branch CI")
    opening = sub.add_parser("open", help="Verify readiness, then create a PR in origin")
    opening.add_argument("--title", required=True)
    opening.add_argument("--body-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "local":
            local()
        elif args.command == "job":
            if (args.install and sys.prefix == sys.base_prefix
                    and os.getenv("GITHUB_ACTIONS") != "true"):
                raise RuntimeError(
                    "Use local or a virtualenv; job --install outside a venv is CI-only",
                )
            job(args.name, Path(sys.executable), install=args.install)
        elif args.command == "ready":
            ready()
        else:
            open_pr(args.title, args.body_file)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Preflight FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
