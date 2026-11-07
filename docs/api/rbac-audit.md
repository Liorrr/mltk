# RBAC & Audit Log

Role-based access control and compliance-grade audit logging for the mltk server.

**Modules:** `mltk.server.rbac`, `mltk.server.audit_log`

---

## Why RBAC for ML Testing

In a team environment, not everyone should have the same permissions on your ML test server:

- **Data scientists** need to READ test results, trends, and dashboards.
- **CI/CD pipelines** need to WRITE test results after each run.
- **Admins** manage API keys, webhooks, and server configuration.

Without RBAC, every API key has full access. If a CI token leaks (committed to a public repo, logged in plaintext, stolen from a build cache), the attacker can:

- Delete all test history
- Modify results to hide regressions
- Create new admin keys for persistent access
- Disable webhooks and alerting

RBAC limits the blast radius. A leaked `reader` key can only view data. A leaked `writer` key cannot manage keys or webhooks. Only `admin` keys have full control.

---

## Roles

mltk uses a strict three-tier hierarchy. A higher role automatically inherits all permissions of the roles below it.

| Role | Read Results | Write Results | Manage Keys | Manage Webhooks | View Audit Log | Export Audit |
|------|:-----------:|:------------:|:-----------:|:---------------:|:--------------:|:------------:|
| `admin` | Yes | Yes | Yes | Yes | Yes | Yes |
| `writer` | Yes | Yes | No | No | No | No |
| `reader` | Yes | No | No | No | No | No |

### Hierarchy Rules

- `admin` satisfies `admin`, `writer`, and `reader` requirements
- `writer` satisfies `writer` and `reader` requirements
- `reader` satisfies only `reader` requirements
- Unknown roles are always denied (fail closed)

---

## Assigning Roles to API Keys

Roles are stored alongside API keys in the server's `api_keys` table. When creating a key, specify the role:

```python
from mltk.server.auth import generate_api_key, hash_key
from mltk.server.rbac import Role

# In your server setup or admin script:
ci_key = generate_api_key()  # CI/CD pipeline key; give the raw key to the pipeline
storage.save_api_key(hash_key(ci_key), "fraud-model", role=Role.WRITER)

viewer_key = generate_api_key()  # Dashboard viewer key
storage.save_api_key(hash_key(viewer_key), "fraud-model", role=Role.READER)
```

The CLI does the same: `mltk server-create-key --project fraud-model --role reader`.

## Server access roles

The server supports `reader`, `writer`, and `admin` keys. Reads require `reader` or
higher; run submission requires `writer` or higher; webhook creation and deletion require
`admin`. Missing or invalid keys return 401. Valid keys without the required role return
403. Health and metrics endpoints remain open. With `MLTK_SERVER_PUBLIC_READ=1` (or
`true`, `yes`, or `on`), read endpoints also accept requests without a key, while supplied
keys continue to be checked.

Project-scoped keys can only access their own project. Reading another project's runs,
trends, summaries or webhooks, or deleting its webhooks, returns 404; submitting a run or
creating a webhook for another project returns 403. Global
webhooks are not included for scoped keys, and returned webhook URLs are masked to
`scheme://host[:port]/...`.

Use `mltk server-create-key --project my-project --role reader` to select a role. Existing
keys with no stored role resolve to `writer` and should be re-issued when a different role
is needed.

### Recommended Key Assignment

| Use Case | Recommended Role | Why |
|----------|:---------------:|-----|
| CI/CD pipeline | `writer` | Needs to push results, never manage keys |
| Dashboard / monitoring | `reader` | View-only, lowest risk if leaked |
| Admin scripts | `admin` | Key management, webhook config |
| Developer local testing | `writer` | Push results during development |
| Shared team dashboard | `reader` | Read-only access for the team |

---

## Protecting Server Routes

Use `require_role` as a FastAPI dependency to enforce permissions on any endpoint:

```python
from fastapi import Depends
from mltk.server.rbac import KeyPrincipal, require_role

@router.post("/api/runs")
async def submit_run(principal: KeyPrincipal = Depends(require_role("writer"))):
    # Only writer and admin keys reach this code; principal.project is the key's project
    ...

@router.get("/api/results")
async def list_results(principal: KeyPrincipal | None = Depends(require_role("reader"))):
    # Any valid key can read results. principal is None only for a request
    # without a key while MLTK_SERVER_PUBLIC_READ is set.
    ...

@router.delete("/api/keys/{key_id}")
async def delete_key(principal: KeyPrincipal = Depends(require_role("admin"))):
    # Only admin keys can manage other keys
    ...
```

### HTTP Error Codes

| Code | Meaning |
|------|---------|
| 401 | No API key provided, or key is invalid |
| 403 | Key is valid but role is insufficient for the operation |

---

## Checking Permissions Programmatically

```python
from mltk.server.rbac import check_permission

check_permission("admin", "reader")   # True  -- admin can do everything
check_permission("writer", "reader")  # True  -- writer can read
check_permission("reader", "writer")  # False -- reader cannot write
check_permission("unknown", "reader") # False -- unknown roles denied
```

### API Reference: `check_permission`

```python
check_permission(user_role: str, required_role: str) -> bool
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_role` | `str` | The role assigned to the API key (e.g. `"writer"`) |
| `required_role` | `str` | The minimum role needed for the operation (e.g. `"reader"`) |

Returns `True` if access should be granted, `False` otherwise. Unknown roles are always denied.

---

## Audit Log

Every API action is recorded with five dimensions:

| Dimension | Field | Example |
|-----------|-------|---------|
| **WHO** | `user_key_hash` | `a1b2c3d4...` (SHA-256, never the raw key) |
| **WHAT** | `action` | `create_run`, `delete_webhook`, `list_results` |
| **WHEN** | `timestamp` | `2026-03-27T14:30:00+00:00` (ISO 8601 UTC) |
| **WHERE** | `resource` | `/api/runs`, `/api/webhooks/wh_123` |
| **RESULT** | `result` + `status_code` | `success` / `failure` + HTTP status |

Each event also receives a unique UUID `id` and an optional `details` dict for additional context.

### Why Audit Logging

Compliance frameworks require an immutable, queryable record of system access:

- **SOC 2 Type II** -- requires evidence that only authorized users accessed data
- **ISO 27001** -- requires audit trails for information security events
- **HIPAA** -- requires audit controls for systems handling health data

Beyond compliance, audit logs are essential for:

- **Incident response** -- "What happened and who was affected?"
- **Access review** -- proving that only authorized keys wrote data
- **Debugging** -- correlating a bad test run with the API call that created it

---

## Using the Audit Logger

```python
from mltk.server.audit_log import AuditLogger

# Create logger with optional file persistence
logger = AuditLogger(storage_path="audit.jsonl")

# Log an action
event = logger.log_action(
    action="create_run",
    user_key_hash="a1b2c3d4e5f6...",
    resource="/api/runs",
    result="success",
    status_code=201,
    details={"run_id": "run_42", "project": "fraud-model"},
)
print(event["id"])         # "f47ac10b-58cc-..."
print(event["timestamp"])  # "2026-03-27T14:30:00+00:00"
```

### API Reference: `AuditLogger`

#### `__init__`

```python
AuditLogger(storage_path: str | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_path` | `str \| None` | `None` | Path to JSON Lines file for durable persistence. If `None`, events are only held in memory. |

#### `log_action`

```python
log_action(action, user_key_hash, resource, result, status_code=200, details=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `str` | *(required)* | Machine-readable action name (e.g. `"create_run"`) |
| `user_key_hash` | `str` | *(required)* | SHA-256 hex digest of the API key. **Never** pass the raw key. |
| `resource` | `str` | *(required)* | API endpoint path (e.g. `"/api/runs"`) |
| `result` | `str` | *(required)* | `"success"` or `"failure"` |
| `status_code` | `int` | `200` | HTTP status code of the response |
| `details` | `dict \| None` | `None` | Optional additional context |

Returns the full event dict including generated `id` and `timestamp`.

#### `get_log`

```python
get_log(action=None, user=None, since=None, limit=100)
```

Query the audit log with optional filters (combined with AND). Returns newest-first.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | `str \| None` | `None` | Filter by exact action name |
| `user` | `str \| None` | `None` | Filter by user key hash |
| `since` | `str \| None` | `None` | ISO 8601 timestamp cutoff |
| `limit` | `int` | `100` | Maximum events to return |

#### `export_csv`

```python
export_csv(output_path: str) -> str
```

Export the full audit log as CSV. Returns the output path.

---

## `assert_audit_log_complete`

Assert that required actions appear in the audit log. Use this as a CI gate to verify that a deployment performed all expected operations.

**Module:** `mltk.server.audit_log`

```python
from mltk.server.audit_log import AuditLogger, assert_audit_log_complete

logger = AuditLogger()
logger.log_action("create_run", "abc123", "/api/runs", "success")
logger.log_action("list_results", "abc123", "/api/results", "success")
logger.log_action("export_report", "abc123", "/api/export", "success")

entries = logger.get_log()

# Verify all expected actions were recorded
result = assert_audit_log_complete(
    entries,
    expected_actions=["create_run", "list_results", "export_report"],
)
assert result.passed
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audit_entries` | `list[dict]` | *(required)* | Audit events (from `AuditLogger.get_log()`) |
| `expected_actions` | `list[str]` | *(required)* | Action names that must appear at least once |

Returns `TestResult` (name: `audit.log_complete`). Severity: `CRITICAL` on failure, `INFO` on pass.

### pytest Integration

```python
def test_deployment_audit_trail(mltk_audit_logger):
    """Verify deployment produces complete audit trail."""
    # ... run deployment workflow ...

    entries = mltk_audit_logger.get_log()
    result = assert_audit_log_complete(
        entries,
        expected_actions=[
            "create_run",
            "list_results",
            "export_report",
        ],
    )
    assert result.passed, result.message
```

---

## CSV Export for Compliance Audits

Compliance auditors (SOC 2, ISO 27001) almost universally request spreadsheet-compatible exports. The `export_csv` method produces a standard CSV with one row per event:

```python
logger = AuditLogger(storage_path="audit.jsonl")

# ... after many API operations ...

csv_path = logger.export_csv("audit-export-2026-Q1.csv")
```

### CSV Columns

| Column | Description |
|--------|-------------|
| `id` | Unique event UUID |
| `timestamp` | ISO 8601 UTC timestamp |
| `action` | Action identifier |
| `user_key_hash` | SHA-256 hash of the API key |
| `resource` | API endpoint path |
| `result` | `success` or `failure` |
| `status_code` | HTTP status code |
| `details` | JSON-serialized additional context |

### Persistence Format

When `storage_path` is configured, events are persisted in [JSON Lines](https://jsonlines.org/) format (one JSON object per line). This format is:

- **Append-friendly** -- new events are appended without rewriting the file
- **Crash-safe** -- a crash mid-write loses at most one line, never corrupting earlier entries
- **Parseable** -- each line is independently valid JSON, compatible with `jq`, SIEM tools, and log aggregators
