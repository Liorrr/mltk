"""GitHub App integration — webhook verification, check runs, and App auth.

Enables mltk to receive GitHub webhook events on pull requests, run ML tests
automatically, and post results back as GitHub Check Runs (the structured
results that appear in a PR's "Checks" tab).

Uses urllib (stdlib) — no external dependencies required.
Auth uses a simplified HMAC-based JWT for local dev/testing. For production
deployments, install ``PyJWT`` and ``cryptography`` for proper RS256 signing.

Architecture overview::

    GitHub                        Your server
    ------                        -----------
    PR opened/updated
      |
      +--> POST /webhook  -----> verify_webhook_signature()
                                    |
                                    v
                                  parse event, run mltk tests
                                    |
                                    v
                                  format_check_run_output()
                                    |
                                    v
                                  create_check_run() -----> GitHub Checks API
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from typing import Any

_GITHUB_API = "https://api.github.com"

# GitHub truncates check run summaries beyond this limit.
_MAX_SUMMARY_LENGTH = 65535

# GitHub limits annotations to 50 per API call.
_MAX_ANNOTATIONS_PER_REQUEST = 50


# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------


class GitHubAppAuth:
    """Authenticate as a GitHub App installation.

    GitHub Apps use a two-step auth process:

    1. **Sign a JWT** with the App's private key (RS256, 10-minute expiry).
       The JWT identifies *which* App is making the request.
    2. **Exchange the JWT** for an installation access token (1-hour expiry,
       scoped to the specific repos the App is installed on).

    The installation token is what you actually use for API calls (creating
    check runs, reading PR data, etc.). This class handles token caching
    and automatic renewal when the token expires.

    Why two steps? Security. The private key never leaves your server.
    Installation tokens are short-lived and scoped to only the repos the
    user has granted access to — principle of least privilege.

    Args:
        app_id: The GitHub App's numeric ID (found in App settings).
        private_key: The PEM-encoded private key downloaded from the App
            settings page. In production, load this from a secret manager
            — never hard-code it.
        installation_id: The numeric ID of the App installation (specific
            to the org/user that installed your App). Found via the
            ``GET /app/installations`` API endpoint.

    Example::

        auth = GitHubAppAuth(
            app_id="12345",
            private_key=open("my-app.pem").read(),
            installation_id="67890",
        )
        token = auth.get_installation_token()
        # Use token for API calls...

    Note:
        This implementation uses HMAC-SHA256 for JWT signing, which is
        suitable for **testing and local development only**. GitHub's
        production API requires RS256 (RSA + SHA-256). For production use,
        install ``PyJWT`` and ``cryptography``::

            pip install PyJWT cryptography
            import jwt
            token = jwt.encode(payload, private_key, algorithm="RS256")
    """

    def __init__(
        self,
        app_id: str,
        private_key: str,
        installation_id: str,
    ) -> None:
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id

        # Cached installation token and its expiry time (epoch seconds).
        # We cache because requesting a new token on every API call would
        # be slow and hit rate limits.
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _base64url_encode(self, data: bytes) -> str:
        """Encode bytes to base64url (URL-safe base64 without padding).

        Standard base64 uses ``+`` and ``/`` which are not URL-safe, and
        adds ``=`` padding. Base64url (RFC 4648 Section 5) replaces these
        with ``-`` and ``_``, and strips padding. JWTs require this format.

        Args:
            data: Raw bytes to encode.

        Returns:
            Base64url-encoded string with no padding.
        """
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _generate_jwt(self) -> str:
        """Generate a JWT signed with the App's private key.

        A JWT (JSON Web Token) has three parts separated by dots::

            header.payload.signature

        - **Header**: Identifies the signing algorithm and token type.
        - **Payload**: Contains claims — who issued it (``iss``), when
          (``iat``), and when it expires (``exp``).
        - **Signature**: Proves the token wasn't tampered with. Created by
          signing ``header.payload`` with the private key.

        GitHub requires:

        - ``iat``: Issued-at time (current time minus 60s for clock drift).
        - ``exp``: Expiry (max 10 minutes from ``iat``).
        - ``iss``: The App ID.

        Note:
            This uses HMAC-SHA256 (symmetric signing with the private key
            as the secret). Production GitHub Apps require RS256 (asymmetric
            RSA signing). The HMAC approach works for testing the JWT
            structure and flow, but GitHub will reject it in production.
            Use ``PyJWT`` for real deployments::

                import jwt
                token = jwt.encode(
                    {"iat": now - 60, "exp": now + 540, "iss": app_id},
                    private_key,
                    algorithm="RS256",
                )

        Returns:
            A JWT string in ``header.payload.signature`` format.
        """
        now = int(time.time())

        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iat": now - 60,        # 60s in the past for clock skew
            "exp": now + (10 * 60),  # 10-minute expiry (GitHub's max)
            "iss": self.app_id,
        }

        # Encode header and payload as base64url JSON
        header_b64 = self._base64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload, separators=(",", ":")).encode())

        # Sign with HMAC-SHA256 using the private key as the secret
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.private_key.encode(),
            signing_input.encode(),
            hashlib.sha256,
        ).digest()
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def _request(
        self,
        method: str,
        path: str,
        token: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        """Execute an HTTP request against the GitHub API.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.).
            path: API path (e.g., ``/app/installations/123/access_tokens``).
            token: Bearer token (JWT or installation token).
            body: Optional JSON request body.

        Returns:
            Tuple of (HTTP status code, parsed JSON response).
        """
        url = f"{_GITHUB_API}{path}"
        data = json.dumps(body).encode() if body is not None else None

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read())
            except Exception:
                err_body = {}
            return exc.code, err_body

    def get_installation_token(self) -> str:
        """Get or refresh the installation access token.

        Installation tokens are scoped to the specific repos where your
        App is installed and expire after 1 hour. This method caches the
        token and only requests a new one when the current token has
        expired (with a 60-second buffer to avoid edge-case failures).

        The flow:

        1. Generate a JWT (identifies the App).
        2. ``POST /app/installations/{id}/access_tokens`` with the JWT.
        3. GitHub returns a token and its expiry time.
        4. Cache the token until it expires.

        Returns:
            A valid installation access token string.

        Raises:
            RuntimeError: If the GitHub API rejects the token request
                (e.g., bad private key, wrong installation ID, or App
                has been uninstalled).
        """
        # Return cached token if still valid (with 60s buffer)
        if self._token and time.time() < (self._token_expires_at - 60):
            return self._token

        jwt_token = self._generate_jwt()
        status, data = self._request(
            "POST",
            f"/app/installations/{self.installation_id}/access_tokens",
            jwt_token,
        )

        if status not in (200, 201):
            message = data.get("message", "unknown error") if isinstance(data, dict) else str(data)
            raise RuntimeError(
                f"Failed to get installation token (HTTP {status}): {message}"
            )

        self._token = data["token"]

        # Parse the expiry time. GitHub returns ISO 8601 format like
        # "2024-01-15T12:00:00Z". We convert to epoch seconds for easy
        # comparison with time.time().
        expires_at = data.get("expires_at", "")
        if expires_at:
            # Parse ISO 8601 datetime string to epoch seconds.
            # Format: "2024-01-15T12:00:00Z"
            import calendar

            struct_time = time.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ")
            self._token_expires_at = float(calendar.timegm(struct_time))
        else:
            # Fallback: assume 1-hour expiry (GitHub's default).
            self._token_expires_at = time.time() + 3600

        return self._token  # type: ignore[return-value]


# ------------------------------------------------------------------
# Webhook signature verification
# ------------------------------------------------------------------


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify a GitHub webhook payload signature (HMAC-SHA256).

    **Why this matters:** Without signature verification, anyone who
    discovers your webhook URL can send fake payloads to trigger
    unwanted actions — running tests on malicious code, creating fake
    check runs, or even deploying code. GitHub signs every webhook
    payload with your webhook secret using HMAC-SHA256, and sends the
    signature in the ``X-Hub-Signature-256`` header.

    **How it works:**

    1. You configure a webhook secret in your GitHub App settings.
    2. GitHub computes ``HMAC-SHA256(secret, payload_body)`` and sends
       it as ``X-Hub-Signature-256: sha256=<hex_digest>``.
    3. Your server computes the same HMAC and compares. If they match,
       the payload is authentic and unmodified.

    We use ``hmac.compare_digest()`` instead of ``==`` to prevent
    timing attacks. A naive string comparison leaks information about
    how many characters matched via response timing, which an attacker
    could exploit to guess the signature byte-by-byte.

    Args:
        payload: The raw request body bytes (do NOT parse/re-encode —
            use the exact bytes GitHub sent).
        signature: The ``X-Hub-Signature-256`` header value, in the
            format ``"sha256=<hex_digest>"``.
        secret: Your webhook secret string (configured in the GitHub
            App's webhook settings).

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.

    Example::

        # In your webhook handler (e.g., Flask, FastAPI, etc.):
        payload = request.body
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_webhook_signature(payload, signature, WEBHOOK_SECRET):
            return Response("Forbidden", status=403)
    """
    if not signature.startswith("sha256="):
        return False

    expected_signature = signature[len("sha256="):]

    computed = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison prevents timing attacks.
    return hmac.compare_digest(computed, expected_signature)


# ------------------------------------------------------------------
# Check Runs
# ------------------------------------------------------------------


def create_check_run(
    repo: str,
    head_sha: str,
    name: str,
    title: str,
    summary: str,
    conclusion: str,
    token: str,
    details_url: str | None = None,
    annotations: list[dict] | None = None,
) -> dict:
    """Create a GitHub Check Run with test results.

    **Why Check Runs instead of PR comments?** Check Runs are a first-class
    GitHub concept with significant advantages:

    - They appear in a dedicated **Checks tab** on the PR, not buried in
      the comment thread.
    - They can **block merging** via branch protection rules (e.g., require
      the "mltk" check to pass before merge).
    - They support **structured annotations** — per-file, per-line markers
      that appear inline in the PR diff (like "this test failed on line 42").
    - They have a defined lifecycle (``queued`` -> ``in_progress`` ->
      ``completed``) with conclusion states (``success``, ``failure``,
      ``neutral``, etc.).

    PR comments are just unstructured text — they can't block merges or
    point to specific code locations.

    Args:
        repo: Repository in ``owner/name`` format (e.g., ``"acme/ml-service"``).
        head_sha: The commit SHA to attach the check run to. Must match
            the PR's head commit.
        name: The name of the check (appears in the Checks tab, e.g.,
            ``"mltk"``). Must be unique per App + commit.
        title: Short title displayed in the check run header (e.g.,
            ``"3 tests passed, 1 failed"``).
        summary: Markdown-formatted summary body. Can include tables,
            charts, detailed results. Truncated to 65,535 characters.
        conclusion: The final status. Must be one of: ``"success"``,
            ``"failure"``, ``"neutral"``, ``"cancelled"``, ``"timed_out"``,
            ``"action_required"``, ``"skipped"``, or ``"stale"``.
        token: A valid installation access token (from ``GitHubAppAuth``).
        details_url: Optional URL linking to a full test results page.
        annotations: Optional list of annotation dicts, each with:
            - ``path`` (str): File path relative to repo root.
            - ``start_line`` (int): Starting line number.
            - ``end_line`` (int): Ending line number.
            - ``annotation_level`` (str): ``"notice"``, ``"warning"``, or ``"failure"``.
            - ``message`` (str): Annotation text shown inline.

    Returns:
        The parsed JSON response from the GitHub API (contains the check
        run ID, URL, and other metadata).

    Raises:
        RuntimeError: If the API returns a non-2xx status (e.g., bad token,
            repo not found, App not installed on that repo).

    Example::

        result = create_check_run(
            repo="myorg/ml-service",
            head_sha="abc123",
            name="mltk",
            title="2/3 tests passed",
            summary="## Results\\n| Test | Status |\\n...",
            conclusion="failure",
            token=auth.get_installation_token(),
            annotations=[{
                "path": "models/predict.py",
                "start_line": 42,
                "end_line": 42,
                "annotation_level": "failure",
                "message": "Drift detected: PSI = 0.35 (threshold: 0.2)",
            }],
        )
    """
    body: dict[str, Any] = {
        "name": name,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": title,
            "summary": summary[:_MAX_SUMMARY_LENGTH],
        },
    }

    if details_url:
        body["details_url"] = details_url

    if annotations:
        # GitHub limits to 50 annotations per request. We include up to
        # the limit and note the truncation in the summary if needed.
        body["output"]["annotations"] = annotations[:_MAX_ANNOTATIONS_PER_REQUEST]

    url = f"{_GITHUB_API}/repos/{repo}/check-runs"
    data = json.dumps(body).encode()

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read())
        except Exception:
            err_body = {}
        message = (
            err_body.get("message", "unknown error")
            if isinstance(err_body, dict)
            else str(err_body)
        )
        raise RuntimeError(
            f"GitHub Check Run API error {exc.code}: {message}"
        ) from exc


def format_check_run_output(
    results: list[dict],
    title: str = "mltk Test Results",
) -> dict:
    """Format mltk test results into a GitHub Check Run output.

    Takes a list of test result dicts (as produced by mltk's test runners)
    and converts them into the structure expected by ``create_check_run()``:
    a title, Markdown summary, conclusion, and per-test annotations.

    The summary includes:

    - **Pass/fail ratio** with percentage (e.g., "7/10 passed (70.0%)")
    - **Total duration** in seconds
    - **Markdown table** of all test results (name, status, duration, message)
    - **Failed test details** as annotations (appear inline in PR diffs)

    Args:
        results: List of test result dicts. Each dict should have:
            - ``name`` (str): Test name (e.g., ``"drift_psi_age"``).
            - ``passed`` (bool): Whether the test passed.
            - ``message`` (str, optional): Details about the result.
            - ``duration`` (float, optional): Duration in seconds.
            - ``file`` (str, optional): File path for annotation placement.
            - ``line`` (int, optional): Line number for annotation placement.
        title: Override the default title string.

    Returns:
        Dict with keys ``title``, ``summary``, ``conclusion``, and
        ``annotations`` — ready to be unpacked into ``create_check_run()``.

    Example::

        results = [
            {"name": "drift_psi", "passed": True, "duration": 0.5},
            {"name": "bias_dpd", "passed": False, "message": "DPD=0.15", "duration": 1.2,
             "file": "models/predict.py", "line": 42},
        ]
        output = format_check_run_output(results)
        create_check_run(
            repo="myorg/ml-service",
            head_sha="abc123",
            name="mltk",
            token=token,
            **output,
        )
    """
    if not results:
        return {
            "title": title,
            "summary": "No test results to report.",
            "conclusion": "neutral",
            "annotations": [],
        }

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed
    total_duration = sum(r.get("duration", 0.0) for r in results)
    pass_rate = (passed / total) * 100 if total > 0 else 0.0

    # Determine conclusion: success only if ALL tests pass.
    # "neutral" would mean "informational only, don't fail the check",
    # but failing tests SHOULD block merges.
    conclusion = "success" if failed == 0 else "failure"

    # Build Markdown summary table
    lines = [
        f"## {title}",
        "",
        f"**{passed}/{total}** tests passed ({pass_rate:.1f}%) "
        f"| Duration: **{total_duration:.2f}s**",
        "",
        "| Test | Status | Duration | Message |",
        "|------|--------|----------|---------|",
    ]

    for r in results:
        name = r.get("name", "unnamed")
        status = "PASS" if r.get("passed", False) else "FAIL"
        status_icon = "+" if r.get("passed", False) else "-"
        duration = f"{r.get('duration', 0.0):.3f}s"
        message = r.get("message", "")
        # Escape pipe characters in message so they don't break the table
        message = message.replace("|", "\\|")
        lines.append(f"| {name} | {status_icon} {status} | {duration} | {message} |")

    summary = "\n".join(lines)

    # Truncate summary if it exceeds GitHub's limit
    if len(summary) > _MAX_SUMMARY_LENGTH:
        truncation_notice = "\n\n*... summary truncated (exceeded 65,535 characters) ...*"
        summary = summary[: _MAX_SUMMARY_LENGTH - len(truncation_notice)] + truncation_notice

    # Build annotations for failed tests (these appear inline in PR diffs)
    annotations: list[dict] = []
    for r in results:
        if r.get("passed", False):
            continue
        if "file" not in r:
            continue
        annotations.append({
            "path": r["file"],
            "start_line": r.get("line", 1),
            "end_line": r.get("line", 1),
            "annotation_level": "failure",
            "message": r.get("message", f"Test '{r.get('name', 'unnamed')}' failed"),
        })

    return {
        "title": title,
        "summary": summary,
        "conclusion": conclusion,
        "annotations": annotations,
    }
