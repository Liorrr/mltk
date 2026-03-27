# GitHub App Integration

Automate mltk test runs on pull requests with a GitHub App -- webhook receiver, check runs, and branch protection.

**Module:** `mltk.integrations.github_app`

---

## Why a GitHub App (vs GitHub Actions)?

mltk already supports GitHub Actions workflows (see [CI/CD Integration](pytest-plugin.md#cicd-integration)). A GitHub Actions integration runs *inside* the CI pipeline: you add a step to your workflow YAML, it calls `pytest --mltk-report`, and the job passes or fails based on the exit code. This is the right choice for most teams starting out.

A GitHub App is a different architecture. Instead of running inside CI, it runs as an independent service that *receives webhooks* from GitHub. When a pull request is opened or updated, GitHub sends an HTTP POST to your app's webhook endpoint. The app then runs mltk tests on its own infrastructure and posts the results back to GitHub as a **check run** -- the same green/red status checks you see from CI, but produced by your own service.

### When to use which

| Approach | Architecture | Best for |
|----------|-------------|----------|
| **GitHub Actions** | Runs inside `.github/workflows/*.yml` | Single repo, standard CI, quick setup |
| **GitHub App** | Standalone webhook receiver, runs independently | Multi-repo fleet, custom dashboards, aggregated results, branch protection gates, self-hosted runners |

The key differences that make a GitHub App worthwhile at scale:

**Multi-repo aggregation.** A GitHub Actions workflow lives in a single repository. If you have 15 ML services, you maintain 15 workflow files and 15 separate dashboards. A GitHub App receives webhooks from all repositories it is installed on, running tests centrally and feeding results into a single mltk server dashboard.

**Custom check runs.** GitHub Actions produces pass/fail at the job level. A GitHub App creates check runs with rich output: per-assertion annotations inline on the PR diff, structured summaries, and severity-based conclusions. A PR reviewer sees exactly which assertions failed and on which lines, without opening a separate CI log.

**Branch protection gates.** GitHub branch protection rules can require specific check runs to pass before merging. With a GitHub App, the check run name is stable and predictable (e.g., `mltk`), making it straightforward to enforce "all ML tests must pass before merge" as a repository policy.

**Decoupled infrastructure.** The app runs on your own hardware or cloud. You control the Python environment, GPU availability, dataset access, and secrets -- without configuring runners or caching strategies in YAML.

---

## Design Decisions

### Why pure stdlib (urllib, hmac, hashlib)?

mltk's core principle is **minimal dependencies**. The GitHub App module uses only Python's standard library for HTTP calls (`urllib.request`), signature verification (`hmac` + `hashlib`), and JWT encoding (`base64` + `json`). This means:

- **Zero install friction** — no `requests`, no `PyJWT`, no `cryptography` to install
- **No supply chain risk** — stdlib is audited with Python itself
- **Predictable behavior** — no version conflicts with user's existing dependencies

For production deployment with RS256 JWT signing (required by GitHub for real App authentication), install `PyJWT` and `cryptography`. The module's docstrings document exactly where to swap the HMAC-based dev JWT for RS256.

### Why Check Runs (not PR comments)?

Check runs are a first-class GitHub API — they appear in the PR's Checks tab, can block merging via branch protection rules, and support structured annotations (per-file, per-line failure markers). PR comments are unstructured text that can be buried in conversation threads. For CI/CD gating, check runs are the correct primitive.

---

## Setup

### 1. Create the GitHub App

Navigate to **github.com/settings/apps/new** (or your organization's settings if you want an org-level app) and fill in:

| Field | Value |
|-------|-------|
| **App name** | `mltk-tests` (or any name you prefer) |
| **Homepage URL** | Your mltk server URL (e.g., `https://mltk.internal.example.com`) |
| **Webhook URL** | `https://your-server.example.com/api/github/webhook` |
| **Webhook secret** | Generate a strong random string (save it -- you will need it for configuration) |

#### Required permissions

Under **Repository permissions**, set:

| Permission | Access | Why |
|------------|--------|-----|
| **Checks** | Read & Write | Create check runs with test results and annotations |
| **Pull requests** | Read | Read PR metadata (head SHA, branch, author) to determine what to test |
| **Contents** | Read | Clone the repository to run tests against the PR code |

Under **Subscribe to events**, check:

| Event | Why |
|-------|-----|
| **Pull request** | Trigger test runs when PRs are opened, updated, or synchronized |
| **Check suite** | Respond to re-run requests when a user clicks "Re-run" on a check |

After creating the app, you will see the **App ID** on the app's settings page. Save this value.

#### Generate a private key

On the app's settings page, scroll to **Private keys** and click **Generate a private key**. This downloads a `.pem` file. Store it securely -- this key is used to authenticate API calls as the app.

### 2. Install on Repository

Go to the app's public page (linked from the settings page) and click **Install**. Select the repositories you want the app to monitor, or choose "All repositories" for org-wide coverage.

After installation, GitHub redirects you. Note the **installation ID** from the URL:

```
https://github.com/settings/installations/12345678
                                           ^^^^^^^^
                                           This is the installation_id
```

Alternatively, you can retrieve installation IDs programmatically through the GitHub API once the app is configured.

### 3. Configure mltk

Set the following environment variables on the machine or container running your mltk server:

| Variable | Example | Description |
|----------|---------|-------------|
| `MLTK_GITHUB_APP_ID` | `123456` | App ID from the app settings page |
| `MLTK_GITHUB_PRIVATE_KEY_PATH` | `/etc/mltk/github-app.pem` | Path to the downloaded `.pem` private key file |
| `MLTK_GITHUB_INSTALLATION_ID` | `12345678` | Installation ID from step 2 |
| `MLTK_GITHUB_WEBHOOK_SECRET` | `whsec_abc123...` | The webhook secret you chose in step 1 |

```bash
export MLTK_GITHUB_APP_ID=123456
export MLTK_GITHUB_PRIVATE_KEY_PATH=/etc/mltk/github-app.pem
export MLTK_GITHUB_INSTALLATION_ID=12345678
export MLTK_GITHUB_WEBHOOK_SECRET=whsec_your_secret_here
```

In a Docker deployment, pass these as environment variables or mount the key file as a secret volume.

---

## API Reference

### verify_webhook_signature

```python
def verify_webhook_signature(
    payload_body: bytes,
    signature_header: str,
    secret: str,
) -> bool
```

#### What it does

Verifies that an incoming webhook request was genuinely sent by GitHub, not by an attacker who discovered your webhook URL. GitHub signs every webhook payload using HMAC-SHA256 with the secret you configured during setup. This function recomputes the signature from the raw request body and compares it against the `X-Hub-Signature-256` header.

#### Why it matters

Without signature verification, anyone who knows your webhook URL can forge requests. An attacker could send a crafted `pull_request` event with a malicious `head.sha`, causing your app to create check runs against arbitrary commits -- or trigger test runs that waste compute resources.

HMAC-SHA256 verification ensures that only GitHub (which knows the shared secret) can produce a valid signature for a given payload. The comparison uses a constant-time algorithm (`hmac.compare_digest`) to prevent timing side-channel attacks.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `payload_body` | `bytes` | Raw HTTP request body (before JSON parsing) |
| `signature_header` | `str` | Value of the `X-Hub-Signature-256` header from the request |
| `secret` | `str` | The webhook secret configured in the GitHub App settings |

#### Returns

`bool` -- `True` if the signature is valid, `False` otherwise.

#### Example

```python
from mltk.integrations.github_app import verify_webhook_signature

# In your webhook endpoint handler (e.g., FastAPI, Flask, etc.)
async def handle_webhook(request):
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    secret = os.environ["MLTK_GITHUB_WEBHOOK_SECRET"]

    if not verify_webhook_signature(body, signature, secret):
        return {"error": "Invalid signature"}, 401

    event = request.headers.get("X-GitHub-Event")
    payload = json.loads(body)
    # ... handle event
```

#### How the signature works

1. GitHub computes `HMAC-SHA256(webhook_secret, request_body)` and sends the result as `sha256=<hex_digest>` in the `X-Hub-Signature-256` header.
2. Your app computes the same HMAC using the shared secret and the raw body bytes.
3. If the two digests match (constant-time comparison), the payload is authentic.

```python
# Equivalent manual verification (for understanding -- use the function above in practice)
import hashlib, hmac

expected = hmac.new(
    secret.encode(),
    msg=payload_body,
    digestmod=hashlib.sha256,
).hexdigest()

# signature_header is "sha256=abc123..."
provided = signature_header.split("=", 1)[1]
is_valid = hmac.compare_digest(expected, provided)
```

---

### create_check_run

```python
def create_check_run(
    repo: str,
    sha: str,
    results: dict,
    token: str,
    name: str = "mltk",
) -> bool
```

#### What it does

Creates a GitHub Check Run on a specific commit. Check runs are the green checkmark or red X that appear on pull requests and commits. Unlike PR comments (which are just text), check runs are first-class GitHub objects: they have a structured conclusion (`success`, `failure`, `neutral`), a summary, and per-line annotations that appear inline in the PR diff.

#### Check runs vs PR comments

| Feature | PR Comment | Check Run |
|---------|-----------|-----------|
| Visibility | Appears in PR conversation thread | Appears in the Checks tab and PR status area |
| Structure | Free-form Markdown | Structured: title, summary, annotations, conclusion |
| Branch protection | Cannot be required | Can be required before merge |
| Re-run | Cannot be re-run | Users can click "Re-run" to trigger again |
| Annotations | Not supported | Inline annotations on specific files/lines |

mltk's server module already includes `post_pr_comment` (in `mltk.server.github_ci`) for posting Markdown summaries as PR comments. `create_check_run` is the richer alternative used by the GitHub App integration.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `repo` | `str` | *(required)* | Repository in `owner/name` format (e.g., `"myorg/ml-service"`) |
| `sha` | `str` | *(required)* | Full 40-character commit SHA to attach the check run to |
| `results` | `dict` | *(required)* | Test results dict with keys: `total`, `passed`, `failed`, `score`, and an optional `results` list of per-test dicts |
| `token` | `str` | *(required)* | GitHub installation access token (obtained via `GitHubAppAuth`) |
| `name` | `str` | `"mltk"` | Display name for the check run (visible in the Checks tab) |

#### Returns

`bool` -- `True` if the check run was created successfully, `False` on any API error.

#### Conclusion logic

| Condition | `conclusion` value |
|-----------|-------------------|
| `failed == 0` | `"success"` |
| `failed > 0` | `"failure"` |

#### Annotations

For each failed test result, the check run includes an annotation. Annotations appear as inline comments in the GitHub PR diff view, making it immediately clear which assertions failed without opening a separate log.

Each annotation includes:

| Field | Value |
|-------|-------|
| `path` | `.` (repository root -- mltk assertions are not tied to specific source lines) |
| `start_line` | `1` |
| `annotation_level` | `"failure"` |
| `title` | Test name (e.g., `"data.drift[age]"`) |
| `message` | Test failure message (e.g., `"PSI 0.35 exceeds threshold 0.2"`) |

GitHub caps annotations at 50 per check run request. If more than 50 tests fail, only the first 50 are annotated.

#### Example

```python
from mltk.integrations.github_app import create_check_run

results = {
    "total": 12,
    "passed": 10,
    "failed": 2,
    "score": 83.3,
    "results": [
        {"name": "data.schema", "passed": True, "message": "Schema valid"},
        {"name": "data.drift[age]", "passed": False, "message": "PSI 0.35 > 0.2",
         "severity": "critical"},
        {"name": "data.drift[income]", "passed": False, "message": "PSI 0.28 > 0.2",
         "severity": "critical"},
        {"name": "model.accuracy", "passed": True, "message": "0.94 >= 0.90"},
        # ... remaining results
    ],
}

success = create_check_run(
    repo="myorg/ml-service",
    sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    results=results,
    token=installation_token,  # obtained from GitHubAppAuth
    name="mltk",
)
```

The resulting check run appears in the PR as:

```
mltk -- 10/12 passed

Summary: 10/12 tests passed -- score 83.3%
2 test(s) failed.

Annotations:
  data.drift[age]    -- PSI 0.35 > 0.2
  data.drift[income] -- PSI 0.28 > 0.2
```

---

### format_check_run_output

```python
def format_check_run_output(results: dict) -> dict
```

#### What it does

Converts an mltk test results dictionary into the `output` object expected by the GitHub Check Runs API. This is the formatting layer -- it builds the title, summary Markdown, and annotations list without making any API calls. Use it when you need to inspect or customize the check run payload before sending it.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `results` | `dict` | Test results dict with keys: `total`, `passed`, `failed`, `score`, and an optional `results` list |

#### Returns

A dictionary matching the GitHub Check Runs API `output` schema:

```python
{
    "title": "mltk -- 10/12 passed",
    "summary": "10/12 tests passed -- score 83.3%\n\n2 test(s) failed.",
    "annotations": [
        {
            "path": ".",
            "start_line": 1,
            "end_line": 1,
            "annotation_level": "failure",
            "title": "data.drift[age]",
            "message": "PSI 0.35 exceeds threshold 0.2",
        },
        # ... one per failed test, up to 50
    ],
}
```

#### Example

```python
from mltk.integrations.github_app import format_check_run_output
import json

results = {
    "total": 5,
    "passed": 5,
    "failed": 0,
    "score": 100.0,
    "results": [
        {"name": "data.schema", "passed": True, "message": "ok"},
        {"name": "data.range[age]", "passed": True, "message": "0-120 within bounds"},
    ],
}

output = format_check_run_output(results)
print(json.dumps(output, indent=2))
# {
#   "title": "mltk -- 5/5 passed",
#   "summary": "5/5 tests passed -- score 100.0%\n\nAll tests passed.",
#   "annotations": []
# }
```

---

### GitHubAppAuth

```python
class GitHubAppAuth:
    def __init__(
        self,
        app_id: str,
        private_key_path: str,
        installation_id: str,
    ) -> None: ...

    def get_installation_token(self) -> str: ...
```

#### What it does

Handles the two-step authentication flow required by GitHub Apps. This is more involved than personal access tokens (PATs) because GitHub Apps use asymmetric cryptography rather than static secrets.

#### The two-step flow

**Step 1: Generate a JWT.** The app signs a short-lived JSON Web Token using its private key. This JWT identifies the app itself (not any specific installation or repository). The JWT is valid for up to 10 minutes.

```
Private Key (.pem) + App ID --> JWT (valid ~10 min)
```

**Step 2: Exchange JWT for an installation token.** The JWT is sent to GitHub's API to request an installation access token. This token is scoped to the specific repositories where the app is installed and expires after 1 hour. All subsequent API calls (creating check runs, reading PR data) use this installation token.

```
JWT + Installation ID --> Installation Token (valid ~1 hour)
```

This design means the private key never leaves your server, and the installation token has limited scope and lifetime. If a token is leaked, it expires in an hour and only grants access to the repositories where the app is installed.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `app_id` | `str` | GitHub App ID (from the app's settings page) |
| `private_key_path` | `str` | Filesystem path to the `.pem` private key file |
| `installation_id` | `str` | Installation ID for the target organization or user account |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_installation_token()` | `str` | Returns a fresh installation access token (valid ~1 hour). Caches the token and refreshes automatically when it nears expiration. |

#### Example

```python
from mltk.integrations.github_app import GitHubAppAuth, create_check_run

auth = GitHubAppAuth(
    app_id=os.environ["MLTK_GITHUB_APP_ID"],
    private_key_path=os.environ["MLTK_GITHUB_PRIVATE_KEY_PATH"],
    installation_id=os.environ["MLTK_GITHUB_INSTALLATION_ID"],
)

# Get a fresh token (cached internally, auto-refreshed)
token = auth.get_installation_token()

# Use the token for API calls
create_check_run(
    repo="myorg/ml-service",
    sha="a1b2c3d4...",
    results=my_results,
    token=token,
)
```

#### Token lifecycle

```
t=0min   get_installation_token() --> API call, returns token A (expires t=60min)
t=10min  get_installation_token() --> returns cached token A (still valid)
t=55min  get_installation_token() --> API call, returns token B (token A near expiry)
t=70min  get_installation_token() --> returns cached token B (still valid)
```

---

## Integration with pytest

Combining the pytest plugin with the GitHub App gives you automated check runs on every pull request. The flow:

1. Developer opens a PR.
2. GitHub sends a `pull_request` webhook to your mltk server.
3. The server verifies the webhook signature, clones the repo at the PR's head SHA, and runs `pytest --mltk-report`.
4. Results are posted back as a check run with per-assertion annotations.
5. Branch protection rules enforce that the `mltk` check must pass before merging.

### Webhook handler (conceptual)

```python
from mltk.integrations.github_app import (
    GitHubAppAuth,
    verify_webhook_signature,
    create_check_run,
)

auth = GitHubAppAuth(
    app_id=os.environ["MLTK_GITHUB_APP_ID"],
    private_key_path=os.environ["MLTK_GITHUB_PRIVATE_KEY_PATH"],
    installation_id=os.environ["MLTK_GITHUB_INSTALLATION_ID"],
)

async def handle_pr_webhook(request):
    # Step 1: Verify the webhook is from GitHub
    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_webhook_signature(body, sig, os.environ["MLTK_GITHUB_WEBHOOK_SECRET"]):
        return {"error": "Invalid signature"}, 401

    payload = json.loads(body)
    action = payload.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored"}

    repo = payload["repository"]["full_name"]
    sha = payload["pull_request"]["head"]["sha"]

    # Step 2: Run mltk tests (subprocess, task queue, etc.)
    results = run_mltk_tests(repo, sha)

    # Step 3: Post results as a check run
    token = auth.get_installation_token()
    create_check_run(repo=repo, sha=sha, results=results, token=token)

    return {"status": "ok"}
```

### Branch protection

After the app is working, configure branch protection in your repository settings:

1. Go to **Settings > Branches > Branch protection rules**.
2. Click **Add rule** for `main` (or your default branch).
3. Enable **Require status checks to pass before merging**.
4. Search for and select the **mltk** check.

Now every PR must have passing mltk tests before it can be merged. This enforces ML test quality at the repository level, not just through CI convention.

### GitHub Actions fallback

For repositories where you do not want to run a full webhook-based app, you can still use the existing `mltk.server.github_ci` module to post check runs from a GitHub Actions workflow:

```yaml
# .github/workflows/ml-tests.yml
jobs:
  ml-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install mltk[sklearn]
      - run: pytest --mltk-report --mltk-export-json results.json
      - name: Post check run
        if: always()
        run: |
          python -c "
          import json
          from mltk.server.github_ci import create_check_run
          results = json.load(open('results.json'))
          create_check_run(
              repo='${{ github.repository }}',
              sha='${{ github.sha }}',
              results=results,
              token='${{ secrets.GITHUB_TOKEN }}',
          )
          "
```

This uses the same `create_check_run` function from the server module, but triggered from within a GitHub Actions workflow rather than from a standalone webhook receiver.

---

## Comparison: existing modules vs GitHub App

mltk provides several layers of GitHub integration. Here is how they relate:

| Module | Purpose | Auth method |
|--------|---------|-------------|
| `mltk.integrations.github_adapter` | Create/search/update **GitHub Issues** from test failures | Personal access token (PAT) |
| `mltk.server.github_ci` | Post **PR comments** and **check runs** from CI pipelines | PAT or `GITHUB_TOKEN` (Actions) |
| `mltk.integrations.github_app` | Full **GitHub App**: webhook receiver, JWT auth, installation tokens, check runs | App private key + installation token |

The GitHub Issues adapter (documented in [GitHub, Slack & Plugin System](github-integration.md)) creates tickets from failures. The server CI module posts one-off comments and check runs. The GitHub App module provides the complete lifecycle: receiving events, authenticating as an app, and creating structured check runs with automatic token management.

---
