# DevOps Guide

The definitive guide for deploying, operating, and monitoring mltk in production environments.

---

## 1. Server Deployment

### Quick Start

```bash
# Install server extras (FastAPI + uvicorn)
pip install mltk[server]

# Generate an API key
mltk server-create-key --project my-project
# API key created for project 'my-project':
#   mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
# Store this key securely — it will not be shown again.

# Start the server
mltk server
# mltk server at http://127.0.0.1:8080
```

The server auto-creates `mltk_server.db` (SQLite) on first run. All data lives in this single file.

### Production Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` for all interfaces. |
| `--port` | `8080` | HTTP listen port |
| `--db` | `mltk_server.db` | SQLite database path |

```bash
# Bind to all interfaces on port 9000, store data in /var/lib/mltk/
mltk server --host 0.0.0.0 --port 9000 --db /var/lib/mltk/mltk_server.db
```

The database path can also be set via the `MLTK_DB_PATH` environment variable.

### Custom Deployment with uvicorn

For more control over workers, timeouts, and SSL:

```python
import uvicorn
from mltk.server import create_app

app = create_app(db_path="/data/mltk.db")
uvicorn.run(app, host="0.0.0.0", port=8080, workers=2)
```

Or from the command line:

```bash
uvicorn mltk.server:create_app --factory \
    --host 0.0.0.0 --port 8080 --workers 2 --timeout-keep-alive 30
```

### Reverse Proxy (nginx)

Put mltk behind nginx for TLS termination, rate limiting, and static caching:

```nginx
upstream mltk_backend {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl http2;
    server_name mltk.internal.example.com;

    ssl_certificate     /etc/ssl/certs/mltk.crt;
    ssl_certificate_key /etc/ssl/private/mltk.key;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Rate limiting (10 requests/sec burst 20)
    limit_req_zone $binary_remote_addr zone=mltk:10m rate=10r/s;
    limit_req zone=mltk burst=20 nodelay;

    location / {
        proxy_pass http://mltk_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check (no rate limit)
    location = /api/health {
        proxy_pass http://mltk_backend;
        limit_req off;
    }
}

# HTTP redirect
server {
    listen 80;
    server_name mltk.internal.example.com;
    return 301 https://$host$request_uri;
}
```

### Health Check

```bash
curl http://localhost:8080/api/health
# {"status": "ok", "service": "mltk-server"}
```

Use this endpoint for load balancer probes, Docker HEALTHCHECK, and Kubernetes readiness checks.

---

## 2. API Reference

All endpoints are prefixed with `/api`. Write operations (POST, DELETE) require Bearer token authentication. Read operations (GET) are public.

### `GET /api/health`

Health check. No authentication required.

```bash
curl http://localhost:8080/api/health
```

**Response:**

```json
{"status": "ok", "service": "mltk-server"}
```

---

### `POST /api/runs`

Submit test results from a completed run. **Requires Bearer API key.**

```bash
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "my-project",
    "results": [
      {
        "name": "data.schema",
        "passed": true,
        "severity": "critical",
        "message": "Schema OK",
        "details": {},
        "duration_ms": 12.4
      },
      {
        "name": "data.drift.psi",
        "passed": false,
        "severity": "critical",
        "message": "PSI 0.35 > 0.10 threshold",
        "details": {"statistic": 0.35, "threshold": 0.10},
        "duration_ms": 89.1
      }
    ]
  }'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `str` | `"default"` | Project name for grouping |
| `results` | `list[dict]` | *(required)* | Array of test result objects |

**Result object fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Test identifier (e.g., `data.drift.psi`) |
| `passed` | `bool` | Whether the test passed |
| `severity` | `str` | `critical`, `error`, `warning`, or `info` |
| `message` | `str` | Human-readable result message |
| `details` | `dict` | Arbitrary metadata (statistic values, thresholds, etc.) |
| `duration_ms` | `float` | Test execution time in milliseconds |

**Response:**

```json
{"run_id": 42, "status": "saved"}
```

---

### `GET /api/runs`

List recent test runs, ordered most-recent first.

```bash
# All runs
curl http://localhost:8080/api/runs

# Filter by project, limit to 10
curl "http://localhost:8080/api/runs?project=my-project&limit=10"
```

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project` | *(all)* | Filter by project name |
| `limit` | `50` | Maximum runs to return |

**Response:**

```json
{
  "runs": [
    {
      "id": 42,
      "project": "my-project",
      "timestamp": "2026-03-26T13:00:00+00:00",
      "total": 50,
      "passed": 48,
      "failed": 2,
      "score": 96.0,
      "duration_ms": 3420.5
    }
  ]
}
```

---

### `GET /api/runs/{id}`

Full details of a specific run, including every per-test result.

```bash
curl http://localhost:8080/api/runs/42
```

**Response:**

```json
{
  "id": 42,
  "project": "my-project",
  "timestamp": "2026-03-26T13:00:00+00:00",
  "total": 50,
  "passed": 48,
  "failed": 2,
  "score": 96.0,
  "duration_ms": 3420.5,
  "results": [
    {
      "id": 1,
      "name": "data.drift.psi",
      "passed": false,
      "severity": "critical",
      "message": "PSI 0.35 > 0.10 threshold",
      "details": {"statistic": 0.35, "threshold": 0.10},
      "duration_ms": 89.1
    }
  ]
}
```

Returns `404` if the run ID does not exist.

---

### `GET /api/trends/{project}`

Score trend over time. Returns data points in chronological order (oldest first) for chart rendering.

```bash
curl http://localhost:8080/api/trends/my-project
curl "http://localhost:8080/api/trends/my-project?limit=10"
```

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | `20` | Number of data points |

**Response:**

```json
{
  "project": "my-project",
  "trends": [
    {"id": 40, "timestamp": "2026-03-24T10:00:00+00:00", "score": 94.0, "passed": 47, "failed": 3, "total": 50},
    {"id": 41, "timestamp": "2026-03-25T10:00:00+00:00", "score": 96.0, "passed": 48, "failed": 2, "total": 50},
    {"id": 42, "timestamp": "2026-03-26T10:00:00+00:00", "score": 98.0, "passed": 49, "failed": 1, "total": 50}
  ]
}
```

---

### `GET /api/compare`

Compare two runs and get a structured diff.

```bash
curl "http://localhost:8080/api/compare?run_a=40&run_b=42"
```

**Query parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `run_a` | Yes | Baseline run ID |
| `run_b` | Yes | Comparison run ID |

**Response:**

```json
{
  "run_a": 40,
  "run_b": 42,
  "diff": {
    "new_failures": [],
    "fixed": ["data.drift.psi", "model.bias"],
    "still_failing": [],
    "still_passing": ["data.schema", "inference.latency"],
    "new_tests": [],
    "removed_tests": [],
    "score_change": 4.0
  }
}
```

Returns `404` if either run ID does not exist.

---

### `GET /api/summary/{project}`

Analyze test run history for trends, recurring failures, flaky tests, and recommendations.

```bash
curl "http://localhost:8080/api/summary/my-project?limit=20"
```

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | `20` | Number of recent runs to analyze |

**Response:**

```json
{
  "project": "my-project",
  "summary": {
    "trend": "improving",
    "avg_score": 78.5,
    "most_common_failures": [
      ["model.metric.accuracy", 4],
      ["data.drift.psi", 2]
    ],
    "flaky_tests": ["inference.latency.p99"],
    "recommendations": [
      "Found 1 flaky test(s): inference.latency.p99. Stabilize these before trusting pass/fail signals.",
      "Most frequent failure: 'model.metric.accuracy' failed 4 time(s). Prioritize fixing this test or the code it covers."
    ]
  }
}
```

---

### `POST /api/webhooks`

Register a webhook. **Requires Bearer API key.**

```bash
curl -X POST http://localhost:8080/api/webhooks \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://hooks.slack.com/services/T00/B00/xxx",
    "events": ["on_failure"],
    "project": "my-project"
  }'
```

**Response:** `{"webhook_id": 1, "status": "created"}`

---

### `GET /api/webhooks`

List registered webhooks.

```bash
curl http://localhost:8080/api/webhooks
curl "http://localhost:8080/api/webhooks?project=my-project"
```

**Response:**

```json
{
  "webhooks": [
    {"id": 1, "url": "https://hooks.slack.com/...", "events": ["on_failure"], "project": "my-project"}
  ]
}
```

---

### `DELETE /api/webhooks/{id}`

Remove a webhook. **Requires Bearer API key.**

```bash
curl -X DELETE http://localhost:8080/api/webhooks/1 \
  -H "Authorization: Bearer mltk_your_key_here"
```

**Response:** `{"status": "deleted"}`

Returns `404` if the webhook ID does not exist.

---

## 3. CI/CD Integration

### GitHub Actions

Complete workflow for ML testing with server reporting and PR comments:

```yaml
name: ML Tests
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # nightly drift check

jobs:
  ml-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install mltk[all]
          pip install -r requirements.txt

      # Smoke tests on every push (fast, <2 min)
      - name: Smoke tests
        if: github.event_name == 'push'
        run: pytest -m ml_smoke --mltk-report -q

      # Full suite on PRs
      - name: Full ML test suite
        if: github.event_name == 'pull_request'
        run: pytest --mltk-report --mltk-export-json results.json -q

      # Drift + monitoring on nightly schedule
      - name: Nightly drift check
        if: github.event_name == 'schedule'
        run: pytest -m "ml_drift or ml_model" --mltk-report --mltk-export-json results.json -q

      # Push results to mltk server
      - name: Report to mltk server
        if: always()
        run: |
          curl -sf -X POST ${{ secrets.MLTK_SERVER_URL }}/api/runs \
            -H "Authorization: Bearer ${{ secrets.MLTK_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d "{\"project\": \"${{ github.repository }}\", \"results\": $(cat results.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\"results\", d) if isinstance(d, dict) else d))')}"
        continue-on-error: true

      # PR comment with results
      - name: Post PR comment
        if: github.event_name == 'pull_request' && always()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 -c "
          from mltk.server.github_ci import post_pr_comment
          import json, os
          with open('results.json') as f:
              results = json.load(f)
          if isinstance(results, list):
              data = {'total': len(results), 'passed': sum(1 for r in results if r.get('passed')), 'failed': sum(1 for r in results if not r.get('passed')), 'score': 0, 'results': results}
              data['score'] = data['passed'] / data['total'] * 100 if data['total'] else 0
          else:
              data = results
          post_pr_comment(
              repo='${{ github.repository }}',
              pr_number=${{ github.event.pull_request.number }},
              results=data,
              token=os.environ['GITHUB_TOKEN'],
          )
          "

      - name: Upload report artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mltk-report
          path: mltk-reports/
          retention-days: 30

      # Compliance artifacts (optional)
      - name: Generate compliance report
        if: github.event_name == 'schedule'
        run: |
          mltk compliance results.json --risk-level high --system-name "My Model"
          mltk fda-audit results.json --system-name "My Model" --operator "CI Bot"

      - name: Upload compliance docs
        if: github.event_name == 'schedule'
        uses: actions/upload-artifact@v4
        with:
          name: compliance-reports
          path: |
            mltk-reports/
            fda-audit-trail.md
```

**Required secrets:**

| Secret | Description |
|--------|-------------|
| `MLTK_SERVER_URL` | Base URL of your mltk server (e.g., `https://mltk.internal.example.com`) |
| `MLTK_API_KEY` | API key from `mltk server-create-key` |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions |
| `CODECOV_TOKEN` | Token for Codecov coverage uploads (optional) |

### Coverage Enforcement

The CI pipeline enforces a minimum **80% code coverage** threshold via `--fail-under=80` on the pytest-cov command. Any PR that drops coverage below 80% will fail the test job. Coverage reports are uploaded to Codecov on the primary matrix combination (ubuntu-latest, Python 3.12).

### Cross-Platform Rust CI

Rust compilation, linting (`cargo clippy`), formatting (`cargo fmt --check`), and unit tests (`cargo test`) run on a three-OS matrix: **ubuntu-latest**, **macos-latest**, and **windows-latest**. This catches platform-specific issues in the PyO3 bridge, system-dependent FFI behavior, and C-library linking differences before they reach users.

### Release Smoke Testing

The release workflow includes a `test-wheels` job that runs **between** wheel building and PyPI publishing. For each platform (Linux, macOS, Windows), it:

1. Installs the built wheel artifact
2. Verifies `import mltk` succeeds and prints the version
3. Runs `tests/test_rust_bridge.py` to confirm the Rust extension loads and bridge functions return correct types

This gate prevents publishing broken wheels. The publish step only runs after all three platforms pass smoke tests.

---

### GitLab CI

```yaml
stages:
  - test
  - report

ml-tests:
  stage: test
  image: python:3.12
  variables:
    MLTK_DRIFT_METHOD: psi
    MLTK_DRIFT_THRESHOLD: "0.1"
  script:
    - pip install mltk[all]
    - pip install -r requirements.txt
    - pytest --mltk-report --mltk-export-json results.json -q
  artifacts:
    when: always
    paths:
      - mltk-reports/
      - results.json
    expire_in: 30 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"

nightly-drift:
  stage: test
  image: python:3.12
  script:
    - pip install mltk[all]
    - pytest -m "ml_drift or ml_model" --mltk-report --mltk-export-json results.json -q
  artifacts:
    when: always
    paths:
      - mltk-reports/
      - results.json
    expire_in: 90 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"

report-to-server:
  stage: report
  image: python:3.12
  needs: ["ml-tests"]
  script:
    - |
      curl -sf -X POST "${MLTK_SERVER_URL}/api/runs" \
        -H "Authorization: Bearer ${MLTK_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"project\": \"${CI_PROJECT_PATH}\", \"results\": $(python3 -c 'import json; d=json.load(open("results.json")); print(json.dumps(d.get("results", d) if isinstance(d, dict) else d))')}"
  allow_failure: true
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

---

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent {
        docker { image 'python:3.12' }
    }
    environment {
        MLTK_API_KEY = credentials('mltk-api-key')
        MLTK_SERVER_URL = 'https://mltk.internal.example.com'
    }
    stages {
        stage('Install') {
            steps {
                sh 'pip install mltk[all] -r requirements.txt'
            }
        }
        stage('ML Tests') {
            steps {
                sh 'pytest --mltk-report --mltk-export-json results.json -q'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'mltk-reports/**', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'results.json', allowEmptyArchive: true
                }
            }
        }
        stage('Report to Server') {
            steps {
                sh '''
                    curl -sf -X POST "${MLTK_SERVER_URL}/api/runs" \
                      -H "Authorization: Bearer ${MLTK_API_KEY}" \
                      -H "Content-Type: application/json" \
                      -d "{\\"project\\": \\"${JOB_NAME}\\", \\"results\\": $(python3 -c 'import json; d=json.load(open("results.json")); print(json.dumps(d.get("results", d) if isinstance(d, dict) else d))')}"
                '''
            }
        }
    }
    post {
        failure {
            sh 'mltk notify slack --results-json results.json'
        }
    }
}
```

---

### Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
pr:
  branches:
    include:
      - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.12'

  - script: |
      pip install mltk[all]
      pip install -r requirements.txt
    displayName: 'Install dependencies'

  - script: |
      pytest --mltk-report --mltk-export-json results.json -q
    displayName: 'Run ML tests'

  - script: |
      curl -sf -X POST "$(MLTK_SERVER_URL)/api/runs" \
        -H "Authorization: Bearer $(MLTK_API_KEY)" \
        -H "Content-Type: application/json" \
        -d "{\"project\": \"$(Build.Repository.Name)\", \"results\": $(python3 -c 'import json; d=json.load(open("results.json")); print(json.dumps(d.get("results", d) if isinstance(d, dict) else d))')}"
    displayName: 'Report to mltk server'
    condition: always()
    continueOnError: true

  - task: PublishBuildArtifacts@1
    inputs:
      pathtoPublish: 'mltk-reports'
      artifactName: 'mltk-report'
    condition: always()
```

---

### Generic CI (any system that runs Python)

```bash
#!/bin/bash
set -euo pipefail

# Install
pip install mltk[all]
pip install -r requirements.txt

# Run tests with JSON export
pytest --mltk-report --mltk-export-json results.json -q

# Push to mltk server
RESULTS=$(python3 -c 'import json; d=json.load(open("results.json")); print(json.dumps(d.get("results", d) if isinstance(d, dict) else d))')
curl -sf -X POST "${MLTK_SERVER_URL}/api/runs" \
  -H "Authorization: Bearer ${MLTK_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"project\": \"${PROJECT_NAME}\", \"results\": ${RESULTS}}"

# Send Slack notification on failure
if [ $? -ne 0 ]; then
  mltk notify slack --results-json results.json
fi
```

Or use the built-in pytest flag for automatic server reporting:

```bash
pytest --mltk-server https://mltk.internal.example.com \
       --mltk-server-key mltk_your_key_here \
       --mltk-server-project my-project \
       --mltk-report
```

---

## 4. Authentication & Security

### Generating API Keys

```bash
# Create a key for a specific project
mltk server-create-key --project my-project
# API key created for project 'my-project':
#   mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
# Store this key securely — it will not be shown again.

# Use a specific database path
mltk server-create-key --project staging --db /data/mltk_server.db
```

Keys are generated using Python's `secrets.token_urlsafe(32)`, prefixed with `mltk_`. Only the SHA-256 hash is stored in the database. The raw key is shown exactly once at creation time.

### Bearer Token Usage

Include the key as a `Bearer` token in the `Authorization` header:

```bash
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd" \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "results": []}'
```

### Protected vs. Public Endpoints

| Endpoint | Auth Required | Description |
|----------|:---:|-------------|
| `GET /api/health` | No | Health check |
| `GET /api/runs` | No | List runs |
| `GET /api/runs/{id}` | No | Run details |
| `GET /api/trends/{project}` | No | Score trends |
| `GET /api/compare` | No | Run comparison |
| `GET /api/summary/{project}` | No | History summary |
| `GET /api/webhooks` | No | List webhooks |
| `POST /api/runs` | **Yes** | Submit results |
| `POST /api/webhooks` | **Yes** | Create webhook |
| `DELETE /api/webhooks/{id}` | **Yes** | Delete webhook |

### Authentication Errors

| HTTP Status | Detail | Cause |
|-------------|--------|-------|
| `401` | `API key required` | Missing `Authorization` header |
| `401` | `Invalid API key` | Key hash not found in database |

### SSRF Protection

Webhook URLs are validated before dispatch. The server rejects:

- Non-HTTP(S) schemes (`file://`, `ftp://`, etc.)
- Empty or missing hostnames
- `localhost` and `localhost.localdomain`
- Private/loopback IP addresses: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `::1/128`, `fc00::/7`

### Network Security Recommendations

1. **Run behind a reverse proxy** with TLS termination (see nginx example in Section 1)
2. **Restrict network access** to the server port using firewall rules or security groups
3. **Use separate API keys** per project and per environment (dev, staging, prod)
4. **Rotate keys** periodically by creating a new key and updating CI secrets before deleting the old one
5. **Do not expose** the server to the public internet unless necessary. Prefer VPN or internal networking.
6. **Store API keys** in your CI system's secret management (GitHub Secrets, GitLab CI Variables, Jenkins Credentials)

---

## 5. Webhooks & Alerts

### Supported Events

| Event | Triggers When |
|-------|---------------|
| `on_failure` | Any test in the run failed |
| `on_success` | All tests in the run passed |
| `on_drift` | Reserved for future drift detection alerts |

### Webhook Payload

When a run triggers a webhook, the server POSTs this JSON:

```json
{
  "event": "on_failure",
  "run_id": 42,
  "project": "my-project",
  "passed": 48,
  "failed": 2,
  "total": 50
}
```

Dispatch is best-effort with a 10-second timeout per endpoint.

### Slack Integration

**Option A: Server webhook (automatic on every run)**

```bash
# Register a Slack incoming webhook
curl -X POST http://localhost:8080/api/webhooks \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://hooks.slack.com/services/T00/B00/xxx",
    "events": ["on_failure"],
    "project": "my-project"
  }'
```

**Option B: CLI notification (explicit, from CI)**

```bash
# From JSON results
export MLTK_SLACK_WEBHOOK=https://hooks.slack.com/services/T00/B00/xxx
mltk notify slack --results-json results.json

# Custom message
mltk notify slack --message "Nightly drift check passed for production model"
```

**Option C: Python API (custom logic)**

```python
from mltk.integrations import notify_slack

# Send results summary to Slack
notify_slack(
    webhook_url="https://hooks.slack.com/services/T00/B00/xxx",
    suite=my_suite,
)

# Or a plain message
notify_slack(
    webhook_url="https://hooks.slack.com/services/T00/B00/xxx",
    message="Drift detected: feature_age PSI 0.35 exceeds threshold",
)
```

The Slack message includes pass/fail counts, score percentage, and a list of up to 20 failed tests with a green/red color border.

### PagerDuty Integration

Register a PagerDuty Events API v2 webhook to page on-call when ML tests fail:

```bash
curl -X POST http://localhost:8080/api/webhooks \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://events.pagerduty.com/v2/enqueue",
    "events": ["on_failure"],
    "project": "production-model"
  }'
```

For PagerDuty's expected payload format, use a middleware endpoint (e.g., AWS Lambda, Cloudflare Worker) that receives the mltk webhook and transforms it:

```python
# Example Lambda handler: mltk webhook -> PagerDuty event
import json
import urllib.request

def handler(event, context):
    body = json.loads(event["body"])
    pagerduty_payload = {
        "routing_key": "YOUR_PAGERDUTY_INTEGRATION_KEY",
        "event_action": "trigger",
        "payload": {
            "summary": f"mltk: {body['failed']}/{body['total']} tests failed in {body['project']}",
            "severity": "critical",
            "source": "mltk-server",
            "custom_details": body,
        },
    }
    req = urllib.request.Request(
        "https://events.pagerduty.com/v2/enqueue",
        data=json.dumps(pagerduty_payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)
    return {"statusCode": 200}
```

### Custom Webhook Handler

Any HTTP service that accepts POST with JSON body works as a webhook target:

```python
# Flask example: receive mltk webhook events
from flask import Flask, request

app = Flask(__name__)

@app.route("/mltk-webhook", methods=["POST"])
def handle_mltk():
    data = request.json
    event = data["event"]          # "on_failure" or "on_success"
    project = data["project"]
    failed = data["failed"]
    total = data["total"]

    if event == "on_failure":
        # Custom logic: create ticket, send SMS, page on-call, etc.
        print(f"ALERT: {failed}/{total} tests failed in {project}")

    return "", 200
```

---

## 6. Cloud Monitoring

mltk can validate ML model endpoints deployed on major cloud platforms. These assertions run as regular pytest tests, making cloud health checks part of your CI/CD pipeline.

### AWS SageMaker

```bash
pip install mltk[aws]
```

```python
from mltk.monitor.aws import (
    assert_endpoint_healthy,
    assert_endpoint_latency,
    assert_endpoint_error_rate,
)

# Endpoint is InService
assert_endpoint_healthy("my-model-endpoint", region="us-east-1")

# CloudWatch ModelLatency P99 within threshold
assert_endpoint_latency("my-model-endpoint", max_p99_ms=500)

# 4XX+5XX error rate within threshold
assert_endpoint_error_rate("my-model-endpoint", max_rate=0.01)
```

### GCP Vertex AI

```bash
pip install mltk[gcp]
```

```python
from mltk.monitor.gcp import assert_endpoint_healthy, assert_prediction_latency

assert_endpoint_healthy("my-endpoint", project="my-project", location="us-central1")
assert_prediction_latency("my-endpoint", max_p99_ms=500)
```

### Azure ML

```bash
pip install mltk[azure]
```

```python
from mltk.monitor.azure import assert_endpoint_healthy, assert_endpoint_latency

assert_endpoint_healthy("my-endpoint", resource_group="my-rg")
assert_endpoint_latency("my-endpoint", max_p99_ms=500)
```

### Prometheus + Grafana

For on-prem or self-managed infrastructure:

```python
from mltk.monitor.prometheus import (
    assert_prometheus_metric,
    assert_gpu_utilization,
    assert_triton_healthy,
)

# Run a PromQL query and check result
assert_prometheus_metric(
    "http://prometheus:9090",
    'up{job="model-server"}',
    threshold=1.0,
)

# DCGM GPU utilization below limit
assert_gpu_utilization("http://prometheus:9090", max_util=0.95)

# Triton Inference Server health
assert_triton_healthy("http://triton:8000")
```

### SLA Validation

Combine with mltk's built-in SLA assertion for production monitoring:

```python
from mltk.monitor import assert_sla, assert_no_degradation

# Validate latency and error rate against SLA thresholds
assert_sla(
    latency_p99=150.0,
    error_rate=0.002,
    thresholds={"latency_p99_ms": 200.0, "error_rate": 0.01},
)

# Detect gradual performance decline
daily_accuracy = [0.95, 0.94, 0.95, 0.93, 0.94, 0.92, 0.91, 0.90]
assert_no_degradation(daily_accuracy, window=3, max_decline=0.03)
```

### Monitoring Test as a Scheduled CI Job

```yaml
# .github/workflows/ml-monitoring.yml
name: ML Production Monitoring
on:
  schedule:
    - cron: '0 */4 * * *'  # every 4 hours

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install mltk[all]
      - run: pytest tests/monitoring/ --mltk-report --mltk-export-json results.json -q
      - run: |
          curl -sf -X POST "${{ secrets.MLTK_SERVER_URL }}/api/runs" \
            -H "Authorization: Bearer ${{ secrets.MLTK_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d "{\"project\": \"monitoring\", \"results\": $(cat results.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\"results\", d) if isinstance(d, dict) else d))')}"
        if: always()
```

---

## 7. Environment Variables

### mltk Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLTK_DRIFT_METHOD` | `ks` | Drift detection method: `ks`, `psi`, `kl`, `chi2` |
| `MLTK_DRIFT_THRESHOLD` | `0.05` | Drift detection threshold |
| `MLTK_REPORT_DIR` | `./mltk-reports` | Output directory for HTML reports |
| `MLTK_PII_PATTERNS` | `email,phone,ssn,credit_card` | Comma-separated PII pattern names |
| `MLTK_SLACK_WEBHOOK` | *(none)* | Default Slack webhook URL for `mltk notify slack` |
| `MLTK_DB_PATH` | `mltk_server.db` | SQLite database path (server mode) |
| `MLTK_DOCS_PORT` | `8000` | Port for `mltk docs serve` |
| `MLTK_DOCS_HOST` | `127.0.0.1` | Host for `mltk docs serve` |

### GitHub Integration Variables

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub personal access token (auto-provided in GitHub Actions) |

### Configuration Cascade

mltk resolves configuration in this priority order (highest wins):

1. **Function arguments** -- direct values passed to assertion calls
2. **Environment variables** -- `MLTK_*` variables
3. **`mltk.yaml`** -- if present in the working directory
4. **`pyproject.toml [tool.mltk]`** -- if present in the working directory
5. **Built-in defaults** -- always available

### CI-Specific Configuration

Set environment variables in your CI pipeline to override project defaults:

```yaml
# GitHub Actions
env:
  MLTK_DRIFT_METHOD: psi
  MLTK_DRIFT_THRESHOLD: "0.1"
  MLTK_REPORT_DIR: ./mltk-reports
  MLTK_PII_PATTERNS: email,phone,ssn,credit_card

# GitLab CI
variables:
  MLTK_DRIFT_METHOD: psi
  MLTK_DRIFT_THRESHOLD: "0.1"
```

---

## 8. Docker & Kubernetes

### Dockerfile for Running Tests

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install mltk with all extras
RUN pip install --no-cache-dir mltk[all]

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["pytest", "--mltk-report", "--mltk-export-json", "results.json", "-q"]
```

```bash
docker build -t ml-tests .
docker run --rm -v $(pwd)/mltk-reports:/app/mltk-reports ml-tests
```

### Dockerfile for the Server

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mltk[server]

# Create data directory
RUN mkdir -p /app/data

ENV MLTK_DB_PATH=/app/data/mltk_server.db

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')"

CMD ["mltk", "server", "--host", "0.0.0.0", "--port", "8080"]
```

### docker-compose (Full Stack)

```yaml
# docker-compose.yml
services:
  mltk-server:
    build:
      context: .
      dockerfile: Dockerfile.server
    ports:
      - "8080:8080"
    volumes:
      - mltk-data:/app/data
    environment:
      - MLTK_DB_PATH=/app/data/mltk_server.db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Run tests against the server
  ml-tests:
    build:
      context: .
      dockerfile: Dockerfile.tests
    environment:
      - MLTK_SERVER_URL=http://mltk-server:8080
    depends_on:
      mltk-server:
        condition: service_healthy
    profiles:
      - test

volumes:
  mltk-data:
```

```bash
# Start the server
docker compose up -d mltk-server

# Create an API key
docker compose exec mltk-server \
  mltk server-create-key --project my-project --db /app/data/mltk_server.db

# Run tests (on demand)
docker compose run --rm ml-tests
```

### Kubernetes Deployment

```yaml
# mltk-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mltk-server
  labels:
    app: mltk-server
spec:
  replicas: 1  # SQLite: single writer
  selector:
    matchLabels:
      app: mltk-server
  template:
    metadata:
      labels:
        app: mltk-server
    spec:
      containers:
        - name: mltk-server
          image: mltk-server:latest
          ports:
            - containerPort: 8080
          env:
            - name: MLTK_DB_PATH
              value: /data/mltk_server.db
          volumeMounts:
            - name: mltk-data
              mountPath: /data
          readinessProbe:
            httpGet:
              path: /api/health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /api/health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
      volumes:
        - name: mltk-data
          persistentVolumeClaim:
            claimName: mltk-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mltk-server
spec:
  selector:
    app: mltk-server
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mltk-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

```bash
kubectl apply -f mltk-deployment.yaml

# Create API key in the running pod
kubectl exec deploy/mltk-server -- \
  mltk server-create-key --project my-project --db /data/mltk_server.db

# Access from within the cluster
curl http://mltk-server:8080/api/health
```

### Helm Chart Notes

There is no official Helm chart yet. To create one from the Kubernetes manifests above:

1. Template the image tag, resource limits, PVC size, and replica count
2. Use a `values.yaml` for environment-specific configuration
3. Add an Ingress resource for external access with TLS

---

## 9. Backup & Maintenance

### SQLite Database Location

By default, the database is created at `./mltk_server.db` in the working directory. Override with `--db` or `MLTK_DB_PATH`.

```bash
# Check current database path
ls -la mltk_server.db

# In Docker
docker compose exec mltk-server ls -la /app/data/mltk_server.db
```

### Database Schema

The server creates five tables automatically via the migration system:

| Table | Purpose |
|-------|---------|
| `schema_versions` | Tracks which migrations have been applied (version + timestamp) |
| `runs` | Test run summaries (project, timestamp, score, pass/fail counts) |
| `results` | Per-test results linked to runs (name, passed, severity, details) |
| `api_keys` | SHA-256 hashed API keys with project association |
| `webhooks` | Registered webhook URLs, events, and project filters |

### Storage Architecture

The SQLite storage layer includes several production-hardening features:

**WAL Journal Mode** — `PRAGMA journal_mode=WAL` is enabled at startup for concurrent-read performance. Dashboard queries do not block run submissions.

**Foreign Key Enforcement** — `PRAGMA foreign_keys = ON` prevents orphan rows in `results` that reference non-existent `runs`. Any such insert raises an `IntegrityError`.

**Performance Indexes** — Three indexes are created automatically:

```
idx_runs_project   → runs(project, id DESC)    — project-filtered listings
idx_results_run    → results(run_id)           — per-run result lookups
idx_api_keys_hash  → api_keys(key_hash)        — API key verification
```

**Singleton Connection** — A single `sqlite3.Connection` with `check_same_thread=False` is reused across all requests, avoiding per-request connection overhead. Call `storage.close()` for clean shutdown.

**Batch Inserts** — `save_run()` uses `executemany()` for result rows, which is significantly faster than per-row `execute()` for large test suites.

**Webhook URL Validation** — URLs are validated at registration time (`POST /api/webhooks`). Private IPs, localhost, and non-HTTP schemes are rejected with HTTP 422 before being persisted. Redirect following is disabled in webhook dispatch to prevent SSRF bypass.

### Backup Strategy

SQLite databases can be backed up with a simple file copy while the server is idle, or with `sqlite3 .backup` for a safe online backup:

```bash
# Simple file copy (stop server first, or accept minimal risk)
cp /data/mltk_server.db /backups/mltk_server_$(date +%Y%m%d).db

# Online backup using sqlite3 CLI (safe while server is running)
sqlite3 /data/mltk_server.db ".backup '/backups/mltk_server_$(date +%Y%m%d).db'"
```

Automate with a cron job:

```bash
# /etc/cron.d/mltk-backup
0 3 * * * root sqlite3 /data/mltk_server.db ".backup '/backups/mltk_server_$(date +\%Y\%m\%d).db'" && find /backups -name "mltk_server_*.db" -mtime +30 -delete
```

### Log Rotation

mltk server logs to stdout. Use your container runtime or systemd for log management:

```bash
# systemd service with journal
# /etc/systemd/system/mltk-server.service
[Unit]
Description=mltk Server
After=network.target

[Service]
Type=simple
User=mltk
ExecStart=/usr/local/bin/mltk server --host 0.0.0.0 --port 8080 --db /data/mltk_server.db
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# View logs
journalctl -u mltk-server -f

# Docker logs with rotation
docker run -d --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 mltk-server
```

### Schema Migration System

The server uses a versioned migration system to manage database schema changes safely. Every schema change is defined as a numbered migration in `_MIGRATIONS` (inside `storage.py`), and the server tracks which migrations have been applied in a `schema_versions` table.

**How it works:**

1. On startup, the server creates the `schema_versions` table if it does not exist.
2. It reads the highest applied version from that table.
3. Any migrations with a version number higher than the current version are executed in order.
4. Each migration is recorded with its version number and a timestamp.

**Current migrations:**

| Version | Description |
|---------|-------------|
| 1 | Initial schema: `runs`, `results`, `api_keys`, `webhooks` tables + performance indexes |

**Adding a new migration:**

To add a schema change in a future release, append a new tuple to the `_MIGRATIONS` list in `src/mltk/server/storage.py`:

```python
_MIGRATIONS = [
    (1, "Initial schema ...", [...]),
    (2, "Add tags column to runs", [
        "ALTER TABLE runs ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
    ]),
]
```

The migration engine uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` so it is safe for existing databases that already have the tables. A pre-migration database (created before the migration system was added) will have migration v1 applied on first startup with the new code, which creates the tracking table and indexes without data loss.

**Inspecting migration status:**

```bash
sqlite3 /data/mltk_server.db "SELECT version, applied_at FROM schema_versions ORDER BY version"
```

### Upgrading mltk

```bash
# Upgrade the package
pip install --upgrade mltk[server]

# Check version
mltk version

# Restart the server — migrations run automatically on startup
systemctl restart mltk-server
# or
docker compose up -d --build mltk-server
```

Database migrations are applied automatically when the server starts. Existing data is preserved. You can verify the current schema version with the `schema_versions` query shown above.

---

## 10. Troubleshooting

### `mltk doctor`

Run diagnostics on any machine to identify environment issues:

```bash
mltk doctor
```

```text
mltk doctor -- environment diagnostics
============================================================
[OK  ] Python version: 3.12.3
[OK  ] numpy installed: 1.26.4
[OK  ] pandas installed: 2.2.1
[WARN] scipy not installed -- required for KS drift test
         -> pip install scipy
[OK  ] mltk.yaml config found
[OK  ] Report directory exists: ./mltk-reports
[FAIL] Rust extension not available -- using pure-Python fallback
         -> pip install mltk[rust]
[OK  ] pytest plugin: registered
[OK  ] Config valid: no misconfigurations
============================================================
Summary: 6 OK, 1 warnings, 1 failures
```

Each check returns a **fix hint** when it fails. Exit code is `1` if any check fails, `0` otherwise.

**Checks performed:**

| # | Check | OK | WARN | FAIL |
|---|-------|-----|------|------|
| 1 | Python version | >= 3.10 | -- | < 3.10 |
| 2 | Core dependencies | numpy + pandas installed | -- | Missing |
| 3 | Optional dependencies | All installed | Some missing | -- |
| 4 | Config file | Found + parseable | Not found | Parse error |
| 5 | Report directory | Exists + writable | -- | Not writable |
| 6 | Baseline directory | Exists | Not found | -- |
| 7 | Rust extension | Loaded | Not available | -- |
| 8 | pytest plugin | Registered | -- | Not found |
| 9 | Config validation | No issues | Suspicious values | -- |

### Common Deployment Issues

#### Server won't start

```
Error: No module named 'fastapi'
```

**Fix:** Install server extras: `pip install mltk[server]`

```
Address already in use
```

**Fix:** Another process is using port 8080. Either stop it (`lsof -i :8080`) or use a different port: `mltk server --port 9000`

```
PermissionError: [Errno 13] Permission denied: 'mltk_server.db'
```

**Fix:** Ensure the user running the server has write access to the database directory. For Docker, check volume permissions.

#### Tests timeout in CI

```
FAILED: Timeout >300s
```

**Fix:** Use test tiering to run fast tests on every push and slow tests on a schedule:

```bash
# Every push: smoke tests only (<2 min)
pytest -m ml_smoke --mltk-report -q

# PRs: full suite except slow tests
pytest -m "not ml_slow" --mltk-report -q

# Nightly: everything
pytest --mltk-report -q
```

Available markers for tiering:

| Marker | Speed | Trigger |
|--------|-------|---------|
| `ml_smoke` | < 2 min | Every commit |
| `ml_data` | < 5 min | Every PR |
| `ml_model` | < 10 min | Merge to main |
| `ml_drift` | < 10 min | Nightly schedule |
| `ml_inference` | Variable | Pre-deploy |
| `ml_slow` | > 30 min | Nightly only |
| `ml_gpu` | Variable | GPU runner only |

#### Webhook delivery failures

Webhooks are best-effort with a 10-second timeout. Common causes:

1. **URL unreachable** -- verify the webhook URL is accessible from the server's network
2. **SSRF protection** -- mltk rejects webhook URLs targeting private IPs or localhost. Ensure your webhook receiver has a public or resolvable hostname
3. **Timeout** -- the webhook endpoint takes longer than 10 seconds to respond. Optimize the receiver or use an async queue
4. **SSL errors** -- if your webhook endpoint uses a self-signed certificate, the stdlib `urllib` will reject it. Use a publicly trusted certificate

Check webhook status:

```bash
# List registered webhooks
curl http://localhost:8080/api/webhooks

# Delete and re-create if URL changed
curl -X DELETE http://localhost:8080/api/webhooks/1 \
  -H "Authorization: Bearer mltk_your_key_here"
```

#### API key issues

```
{"detail": "API key required"}
```

**Fix:** Include the `Authorization` header:

```bash
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_your_key_here" \
  ...
```

```
{"detail": "Invalid API key"}
```

**Fix:** The key was not found. Generate a new one:

```bash
mltk server-create-key --project my-project --db /path/to/mltk_server.db
```

Make sure you are using the same `--db` path that the server is using.

#### Python version mismatch

```
SyntaxError: future feature annotations is not defined
```

**Fix:** mltk requires Python 3.10+. Check with `python --version` and upgrade if needed.

#### Missing Rust extension

```
[WARN] Rust extension not available — using pure-Python fallback
```

This is not an error. mltk includes optional Rust-accelerated functions (KS test, PSI, cosine similarity, PII scanning) that are 10-100x faster. If the Rust extension is not available, pure-Python fallbacks are used automatically. Performance-critical deployments should ensure the binary wheel is installed:

```bash
pip install mltk  # binary wheel includes Rust extension on supported platforms
```

---

## Quick Reference

### Cheat Sheet

```bash
# Install
pip install mltk[server]            # server + dashboard
pip install mltk[all]               # everything

# Server
mltk server                         # start (localhost:8080)
mltk server --host 0.0.0.0          # bind all interfaces
mltk server-create-key --project p  # generate API key

# Tests
pytest --mltk-report                # HTML report
pytest --mltk-export-json r.json    # JSON export
pytest -m ml_smoke                  # smoke tests only
pytest -m "not ml_slow"             # skip slow tests

# CI reporting
pytest --mltk-server URL --mltk-server-key KEY --mltk-server-project NAME

# Notifications
mltk notify slack --results-json results.json

# Diagnostics
mltk doctor                         # environment check
mltk version                        # print version

# Drift check
mltk drift ref.csv cur.csv --method psi

# Compliance
mltk compliance results.json --risk-level high
mltk fda-audit results.json --system-name "My Model"
```

### Essential curl Commands

```bash
# Health check
curl http://localhost:8080/api/health

# Submit run
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"project":"p","results":[{"name":"t","passed":true}]}'

# List runs
curl http://localhost:8080/api/runs?project=p&limit=5

# Get run details
curl http://localhost:8080/api/runs/42

# Score trends
curl http://localhost:8080/api/trends/p?limit=10

# Compare runs
curl "http://localhost:8080/api/compare?run_a=40&run_b=42"

# History summary
curl http://localhost:8080/api/summary/p?limit=20

# Create webhook
curl -X POST http://localhost:8080/api/webhooks \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://hooks.slack.com/...","events":["on_failure"]}'

# List webhooks
curl http://localhost:8080/api/webhooks

# Delete webhook
curl -X DELETE http://localhost:8080/api/webhooks/1 \
  -H "Authorization: Bearer $KEY"
```
