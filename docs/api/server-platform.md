# Server Platform

Self-hosted ML test result tracking — SonarQube for ML testing.

**Module:** `mltk.server`

---

## Overview

The mltk server platform provides:

- **Persistent storage** of test run history in SQLite
- **Live dashboard** at `http://localhost:8080`
- **REST API** for submitting runs, querying trends, and comparing results
- **Webhooks** for Slack/GitHub/PagerDuty alerting on failures
- **GitHub CI integration** — PR comments and check runs
- **API key authentication** for multi-project deployments

---

## Quick Start

```bash
# Install with server extras
pip install mltk[server]

# Generate an API key (run once)
mltk server-create-key --project my-project
# API key created for project 'my-project':
#   mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
# Store this key securely — it will not be shown again.

# Start the server
mltk server
# mltk server at http://127.0.0.1:8080

# Open dashboard in browser
open http://localhost:8080
```

The server auto-creates `mltk_server.db` on first run. All data is stored locally in SQLite with WAL journal mode for concurrent-read performance.

---

## pytest Integration

Submit test results to the server directly from your pytest run:

```bash
# Export results and post to server
pytest --mltk-export-json results.json
curl -s -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d "{\"project\": \"my-project\", \"results\": $(cat results.json | python -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\"results\", [])))')}"
```

For automated CI/CD integration, use the `--mltk-server` pytest flag (requires the server extra):

```bash
pytest --mltk-server http://localhost:8080 \
       --mltk-server-key mltk_your_key_here \
       --mltk-server-project my-project
```

---

## API Endpoints

All API endpoints are prefixed with `/api`. Authentication is required for write operations.

### `GET /api/health`

Health check — no authentication required.

```bash
curl http://localhost:8080/api/health
# {"status": "ok", "service": "mltk-server"}
```

---

### `POST /api/runs`

Submit test results from a completed test run. **Requires Bearer API key.**

**Request body:**

```json
{
  "project": "my-project",
  "results": [
    {
      "name": "data.schema",
      "passed": true,
      "severity": "critical",
      "message": "Schema OK",
      "details": {},
      "duration_ms": 12.4
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `project` | `str` | Project name (default: `"default"`) |
| `results` | `list[dict]` | Array of test result objects |

**Result object fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Test identifier |
| `passed` | `bool` | Whether the test passed |
| `severity` | `str` | `critical`, `error`, `warning`, or `info` |
| `message` | `str` | Human-readable result message |
| `details` | `dict` | Arbitrary extra data |
| `duration_ms` | `float` | Test execution time in milliseconds |

**Response:**

```json
{"run_id": 42, "status": "saved"}
```

---

### `GET /api/runs`

List recent test runs.

```bash
curl http://localhost:8080/api/runs
curl http://localhost:8080/api/runs?project=my-project&limit=10
```

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project` | — | Filter by project name |
| `limit` | `50` | Maximum number of runs to return |

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

Get full details of a specific run, including all per-test results.

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
      "name": "data.drift",
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

Get score trend over time for a project. Useful for dashboards and CI gates.

```bash
curl http://localhost:8080/api/trends/my-project
curl http://localhost:8080/api/trends/my-project?limit=10
```

**Query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | `20` | Number of data points (most recent) |

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

Trends are returned in chronological order (oldest to newest).

---

### `GET /api/compare`

Compare two test runs and return a structured diff of results.

```bash
curl "http://localhost:8080/api/compare?run_a=40&run_b=42"
```

**Query parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `run_a` | Yes | ID of the first run (baseline) |
| `run_b` | Yes | ID of the second run (comparison) |

**Response:**

```json
{
  "run_a": 40,
  "run_b": 42,
  "diff": {
    "improved": ["data.drift", "model.bias"],
    "regressed": [],
    "unchanged": ["data.schema", "inference.latency"],
    "new_failures": [],
    "new_passes": ["data.drift", "model.bias"]
  }
}
```

Returns `404` if either run ID does not exist.

---

## Authentication

The mltk server uses API keys with Bearer token authentication.

### Creating API Keys

```bash
# Create a key for a project
mltk server-create-key --project my-project
# API key created for project 'my-project':
#   mltk_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcd
# Store this key securely — it will not be shown again.
```

Keys are stored as SHA-256 hashes in the SQLite database. The raw key is shown only once.

### Using API Keys

Include the key as a `Bearer` token in the `Authorization` header:

```bash
curl -X POST http://localhost:8080/api/runs \
  -H "Authorization: Bearer mltk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "results": []}'
```

In Python:

```python
import urllib.request
import json

key = "mltk_your_key_here"
payload = json.dumps({"project": "my-project", "results": []}).encode()
req = urllib.request.Request(
    "http://localhost:8080/api/runs",
    data=payload,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(req) as resp:
    print(json.loads(resp.read()))
```

**Authentication errors:**

| HTTP Status | Meaning |
|-------------|---------|
| `401 API key required` | No `Authorization` header provided |
| `401 Invalid API key` | Key not found in database |

---

## Webhooks

Register webhooks to receive HTTP POST notifications when test runs complete.

### Supported Events

| Event | Triggers when |
|-------|---------------|
| `on_failure` | Any test in the run failed |
| `on_success` | All tests in the run passed |
| `on_drift` | Reserved for future drift detection alerts |

### Registering a Webhook

```bash
curl -X POST http://localhost:8080/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://hooks.slack.com/services/...",
    "events": ["on_failure"],
    "project": "my-project"
  }'
# {"webhook_id": 1, "status": "created"}
```

Omit `"project"` to receive notifications for all projects.

### Listing Webhooks

```bash
curl http://localhost:8080/api/webhooks
curl "http://localhost:8080/api/webhooks?project=my-project"
```

### Deleting a Webhook

```bash
curl -X DELETE http://localhost:8080/api/webhooks/1
# {"status": "deleted"}
```

### Webhook Payload

When a run triggers a webhook, the server POSTs this JSON body:

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

Webhook dispatch is best-effort and non-blocking (10-second timeout per endpoint).

---

## Dashboard

The server serves a live HTML dashboard at `http://localhost:8080`. The dashboard shows:

- Pass/fail counts and score for recent runs
- Score trend chart per project
- Per-test result table with severity and duration
- Links to compare adjacent runs

No configuration required — the dashboard is bundled with the server package.

---

## Server Options

```bash
mltk server [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host to bind to (use `0.0.0.0` for all interfaces) |
| `--port` | `8080` | Port to listen on |
| `--db` | `mltk_server.db` | SQLite database path |

The database path can also be set via the `MLTK_DB_PATH` environment variable (used by the Docker image).

---

## Docker Deployment

A `Dockerfile` and `docker-compose.yml` are provided in `server/`.

### docker-compose (recommended)

```yaml
# server/docker-compose.yml
services:
  mltk-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - mltk-data:/app/data
    environment:
      - MLTK_DB_PATH=/app/data/mltk_server.db
    restart: unless-stopped

volumes:
  mltk-data:
```

```bash
cd server/
docker-compose up -d

# Create an API key in the running container
docker-compose exec mltk-server \
  mltk server-create-key --project prod --db /app/data/mltk_server.db
```

### Dockerfile (standalone)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir mltk[server]
EXPOSE 8080
CMD ["mltk", "server", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
docker build -t mltk-server ./server
docker run -p 8080:8080 -v $(pwd)/data:/app/data \
  -e MLTK_DB_PATH=/app/data/mltk_server.db \
  mltk-server
```

---

## GitHub CI Integration

Post test result summaries as PR comments and GitHub Check Runs automatically.

```python
from mltk.server.github_ci import post_pr_comment, create_check_run

results = {
    "total": 50,
    "passed": 48,
    "failed": 2,
    "score": 96.0,
    "results": [...]  # per-test dicts
}

# Post a PR comment
token = "ghp_your_github_token"
post_pr_comment("myorg/ml-service", pr_number=123, results=results, token=token)

# Create a Check Run (blocks merge if tests fail)
create_check_run("myorg/ml-service", sha="abc123def456", results=results, token=token)
```

In a GitHub Actions workflow:

```yaml
- name: Run ML tests and post results
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    pytest --mltk-export-json results.json
    python - <<'EOF'
    import json
    from mltk.server.github_ci import post_pr_comment

    with open("results.json") as f:
        results = json.load(f)

    post_pr_comment(
        repo="${{ github.repository }}",
        pr_number=${{ github.event.pull_request.number }},
        results=results,
        token="${{ secrets.GITHUB_TOKEN }}",
    )
    EOF
```

### PR Comment Format

The generated comment includes:
- Pass/fail counts and score percentage
- Table of failed tests with severity and message
- Link to mltk on GitHub

### Check Run

Sets `conclusion` to `"success"` when all tests pass, `"failure"` when any fail. Adds per-test annotations for failing tests (capped at 50 per GitHub API limit).

---

## Storage Architecture

The server uses SQLite with several production-hardening features enabled at initialization:

### WAL Journal Mode

Write-Ahead Logging (WAL) is enabled via `PRAGMA journal_mode=WAL`, which allows concurrent readers while a write is in progress. This is critical for dashboard queries not blocking run submissions.

### Foreign Key Enforcement

`PRAGMA foreign_keys = ON` ensures referential integrity between `results.run_id` and `runs.id`. Attempts to insert orphan result rows will raise an `IntegrityError`.

### Performance Indexes

Three indexes are created automatically:

| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_runs_project` | `runs(project, id DESC)` | Fast project-filtered run listing |
| `idx_results_run` | `results(run_id)` | Fast per-run result lookups |
| `idx_api_keys_hash` | `api_keys(key_hash)` | Fast API key verification |

### Connection Pooling

A singleton `sqlite3.Connection` with `check_same_thread=False` is reused across all operations, avoiding the overhead of opening/closing connections per request. Call `storage.close()` for clean shutdown.

### Batch Inserts

`save_run()` uses `executemany()` for inserting result rows, which is significantly faster than individual `execute()` calls for large test suites.

### Webhook URL Validation

Webhook URLs are validated at creation time (`POST /api/webhooks`) before being persisted. Invalid URLs (private IPs, localhost, non-HTTP schemes) return HTTP 422. Redirect following is disabled in webhook dispatch to prevent SSRF bypass via 3xx responses.

---

## Python API

### `create_app`

```python
from mltk.server import create_app

app = create_app(db_path="mltk_server.db")
# Returns a FastAPI application instance
```

Use directly with uvicorn for custom deployments:

```python
import uvicorn
from mltk.server import create_app

app = create_app(db_path="/data/mltk.db")
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### `Storage`

```python
from mltk.server.storage import Storage

storage = Storage("mltk_server.db")

# Save a run
run_id = storage.save_run("my-project", results_list)

# Query
runs = storage.get_runs(project="my-project", limit=10)
run = storage.get_run(run_id)
trends = storage.get_trends("my-project", limit=20)
```

### `generate_api_key` / `hash_key`

```python
from mltk.server.auth import generate_api_key, hash_key

raw_key = generate_api_key("my-project")
key_hash = hash_key(raw_key)
storage.save_api_key(key_hash, "my-project")
```

---

## Requirements

```bash
pip install mltk[server]
# Installs: fastapi, uvicorn, pydantic (in addition to mltk core)
```

Python 3.10+ required.
