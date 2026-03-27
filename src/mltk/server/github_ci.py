"""GitHub CI/CD integration — PR comments and check runs from test results."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from importlib.metadata import version as _pkg_version

# Single-source version — reads from installed package metadata (pyproject.toml)
_VERSION = _pkg_version("mltk")

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _github_api(
    method: str,
    url: str,
    token: str,
    body: dict | None = None,  # type: ignore[type-arg]
) -> dict:  # type: ignore[type-arg]
    """Make an authenticated GitHub API request.

    Args:
        method: HTTP verb ("GET", "POST", etc.).
        url: Full GitHub API URL.
        token: GitHub personal access token or Actions GITHUB_TOKEN.
        body: Optional JSON-serialisable payload.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        urllib.error.HTTPError: On 4xx/5xx responses.
        ValueError: If the response body is not valid JSON.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": f"mltk-github-ci/{_VERSION}",
    }
    req = urllib.request.Request(url, method=method, headers=headers)
    if body is not None:
        req.data = json.dumps(body).encode()
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_pr_comment(results: dict) -> str:  # type: ignore[type-arg]
    """Format test results as a GitHub-flavored Markdown comment.

    Args:
        results: Dict with keys: total, passed, failed, score, and optionally
                 a 'results' list of per-test dicts (name, passed, message).

    Returns:
        GitHub-flavored Markdown string ready to POST as a PR comment body.
    """
    total: int = results.get("total", 0)
    passed: int = results.get("passed", 0)
    failed: int = results.get("failed", 0)
    score: float = results.get("score", 0.0)

    # Header + summary line
    if failed == 0:
        status_icon = "✅"
        headline = "All tests passed"
    else:
        status_icon = "❌"
        headline = f"{failed} test(s) failed"

    lines: list[str] = [
        f"## {status_icon} mltk Test Results",
        "",
        f"**{passed}/{total} passed** | score: **{score:.1f}%** | {headline}",
    ]

    # Failed-tests table (only when there are failures)
    failed_tests = [
        r for r in results.get("results", []) if not r.get("passed", True)
    ]
    if failed_tests:
        lines += [
            "",
            "### Failed Tests",
            "",
            "| Test | Severity | Message |",
            "| ---- | -------- | ------- |",
        ]
        for r in failed_tests:
            name = str(r.get("name", "unknown"))
            severity = str(r.get("severity", "error"))
            message = str(r.get("message", "")).replace("|", "\\|")
            lines.append(f"| `{name}` | {severity} | {message} |")

    # Footer
    lines += [
        "",
        "---",
        f"_Posted by [mltk](https://github.com/Liorrr/mltk) v{_VERSION}_",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def post_pr_comment(
    repo: str,
    pr_number: int,
    results: dict,  # type: ignore[type-arg]
    token: str,
) -> bool:
    """Post a test summary as a PR comment on GitHub.

    Args:
        repo: Repository in ``"owner/repo"`` format.
        pr_number: Pull request number.
        results: Dict with keys ``total``, ``passed``, ``failed``, ``score``,
                 and an optional ``results`` list of per-test dicts.
        token: GitHub personal access token (needs ``repo`` scope) or
               ``GITHUB_TOKEN`` from a GitHub Actions workflow.

    Returns:
        ``True`` if the comment was created successfully, ``False`` otherwise.
    """
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    body = {"body": format_pr_comment(results)}
    try:
        _github_api("POST", url, token, body)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return False


def create_check_run(
    repo: str,
    sha: str,
    results: dict,  # type: ignore[type-arg]
    token: str,
    name: str = "mltk",
) -> bool:
    """Create a GitHub Check Run with test results.

    Args:
        repo: Repository in ``"owner/repo"`` format.
        sha: Full commit SHA that the check run targets.
        results: Dict with keys ``total``, ``passed``, ``failed``, ``score``,
                 and an optional ``results`` list of per-test dicts.
        token: GitHub personal access token (needs ``repo`` scope) or
               ``GITHUB_TOKEN`` from a GitHub Actions workflow.
        name: Display name for the check run (default: ``"mltk"``).

    Returns:
        ``True`` if the check run was created successfully, ``False`` otherwise.

    Notes:
        Sets ``conclusion`` to ``"success"`` when all tests pass, ``"failure"``
        when any test fails.  Adds per-test annotations for failing tests.
    """
    total: int = results.get("total", 0)
    passed: int = results.get("passed", 0)
    failed: int = results.get("failed", 0)
    score: float = results.get("score", 0.0)

    conclusion = "success" if failed == 0 else "failure"

    summary = (
        f"{passed}/{total} tests passed — score {score:.1f}%\n\n"
        + (f"{failed} test(s) failed." if failed else "All tests passed.")
    )

    # Build annotations for failing tests (GitHub caps at 50 per request)
    annotations: list[dict] = []  # type: ignore[type-arg]
    for r in results.get("results", []):
        if r.get("passed", True):
            continue
        annotation: dict = {  # type: ignore[type-arg]
            "path": ".",          # file path required by API; use repo root
            "start_line": 1,
            "end_line": 1,
            "annotation_level": "failure",
            "title": str(r.get("name", "unknown")),
            "message": str(r.get("message", "")),
        }
        annotations.append(annotation)
        if len(annotations) >= 50:
            break

    output: dict = {  # type: ignore[type-arg]
        "title": f"mltk — {passed}/{total} passed",
        "summary": summary,
        "annotations": annotations,
    }

    url = f"https://api.github.com/repos/{repo}/check-runs"
    body: dict = {  # type: ignore[type-arg]
        "name": name,
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": output,
    }

    try:
        _github_api("POST", url, token, body)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return False
