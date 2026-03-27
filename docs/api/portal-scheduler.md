# Live Portal & Test Scheduler

Real-time test monitoring dashboard and scheduled test execution.

**Modules:** `mltk.server.portal`, `mltk.server.scheduler`

---

## Live Portal

The mltk live portal is a self-contained HTML page for real-time test result monitoring. It polls the mltk server API and updates the display in place -- no page reloads, no external dependencies, no build step.

### Why a Live Portal

The existing mltk dashboard shows *historical* results -- runs that already completed. The live portal shows what is happening *right now*: which tests are running, which just passed or failed, and how performance metrics are trending over the last few minutes.

Think of it like the difference between a git log and a CI pipeline view. The git log shows what shipped; the pipeline view shows what is building.

### What the Portal Shows

The portal displays four sections:

| Section | Content |
|---------|---------|
| **Banner** | Server URL being polled, refresh interval, last update time |
| **Health cards** | Total runs, pass rate (color-coded), latest score, average duration |
| **Run list** | One card per test run with pass/fail icon, project name, timestamp, pass/fail counts |
| **Footer** | Auto-refresh interval reminder |

### Health Status Colors

| Pass Rate | Color | Label |
|:---------:|:-----:|-------|
| >= 90% | Green | Healthy -- pipeline working as expected |
| 70-89% | Yellow | Degraded -- some tests failing, investigate |
| < 70% | Red | Failing -- significant problems, immediate action needed |

### Architecture

The portal is a **single self-contained HTML file** with embedded CSS and JavaScript. No external CDN, no build step, no framework. This design means:

- Works behind corporate firewalls (no outbound CDN requests)
- Works offline if the mltk server is on localhost
- Can be served by any HTTP server, or opened directly as a file
- No Node.js, no npm, no webpack -- just Python generating HTML

### Data Flow

1. Portal JavaScript calls `GET /api/results?limit=50` every `refresh_seconds`
2. Response is parsed and rendered into cards (one per test run)
3. Top banner shows aggregate health (pass rate, trend, last update time)
4. Failures are highlighted in red with expandable detail sections

---

## `create_portal_html`

Generate a self-contained HTML page for live test result monitoring.

**Module:** `mltk.server.portal`

### Example

```python
from mltk.server.portal import create_portal_html

# Generate the portal HTML
html_content = create_portal_html(
    server_url="http://ci-server:8080",
    refresh_seconds=15,
)

# Save to a file
with open("portal.html", "w") as f:
    f.write(html_content)

# Or serve from your mltk server route:
# @app.get("/portal")
# async def portal():
#     return HTMLResponse(create_portal_html())
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | `"http://localhost:8080"` | Base URL of the mltk server to poll. Must include protocol (`http://` or `https://`). |
| `refresh_seconds` | `int` | `30` | How often (in seconds) the portal fetches new data |

Returns a complete HTML document as a string.

### Refresh Interval Recommendations

| Scenario | Recommended Interval | Why |
|----------|:--------------------:|-----|
| Active CI pipeline | 10-15s | Fast feedback during builds |
| Team dashboard | 30s | Good balance of freshness and load |
| Monitoring display | 60s | Low overhead for wall-mounted screens |
| Testing locally | 5s | Rapid iteration during development |

### Security

The server URL and refresh interval are escaped to prevent XSS if they ever contain angle brackets or quotes (defense-in-depth). The portal uses `fetch()` with same-origin or CORS-enabled requests.

---

## `get_portal_data`

Get current portal data from storage for server-side rendering or API responses.

**Module:** `mltk.server.portal`

```python
from mltk.server.portal import get_portal_data

data = get_portal_data(storage)
# {
#   "runs": [...],        # up to 50 recent runs
#   "summary": {
#     "total_runs": 47,
#     "pass_rate": 91.5,
#     "avg_duration_ms": 1234.5,
#     "latest_score": 95.0,
#   },
#   "health": "healthy",  # "healthy" | "degraded" | "failing" | "unknown"
# }
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage` | `Storage \| None` | Storage instance with a `get_runs()` method. Pass `None` for an empty response. |

### Health Thresholds

| Pass Rate | Health Label |
|:---------:|:------------|
| >= 90% | `"healthy"` |
| >= 70% | `"degraded"` |
| < 70% | `"failing"` |
| No data | `"unknown"` |

---

## Test Scheduler

The mltk scheduler runs test suites on a recurring basis to catch silent ML degradation.

### Why Scheduled Tests

Machine learning systems degrade silently. A model that passes every test today might fail tomorrow because:

- **Data drift**: The distribution of incoming data shifted, and features that were predictive last month are now noise.
- **Dependency rot**: A library updated its numeric precision, changing model outputs just enough to cross a threshold.
- **Infrastructure decay**: A database connection pool started timing out, making data fetches fail intermittently.
- **Stale artifacts**: A feature store cache expired, and the fallback path returns different values.

Unlike traditional software where failures are immediate (crash, 500 error), ML failures are *gradual* -- accuracy drops 0.1% per day until someone notices weeks later. Scheduled test runs catch these issues early.

### Architecture

The scheduler is intentionally simple:

- No external dependencies, no daemon process, no cron binary
- Schedule definitions stored in memory
- Uses `time.time()` to determine when a schedule is due
- Test commands executed via `subprocess.run()`, capturing stdout/stderr/returncode
- Optional webhook notifications (Slack, PagerDuty, OpsGenie, etc.)

For production use, pair with a system cron job, systemd timer, or run inside the mltk server process as a background loop.

---

## `TestScheduler` API

**Module:** `mltk.server.scheduler`

### Creating a Scheduler

```python
from mltk.server.scheduler import TestScheduler

scheduler = TestScheduler()
```

### `add_schedule`

Add a scheduled test run.

```python
scheduler.add_schedule(
    name="nightly-drift-check",
    command="mltk run --tag drift",
    interval_seconds=86400,
    webhook_url="https://hooks.slack.com/services/T.../B.../...",
)

scheduler.add_schedule(
    name="hourly-latency-check",
    command="mltk run --tag latency",
    interval_seconds=3600,
    webhook_url=None,  # no webhook
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique identifier for this schedule |
| `command` | `str` | *(required)* | Shell command to execute (e.g. `"mltk run --tag drift"`) |
| `interval_seconds` | `int` | `3600` | Minimum seconds between consecutive runs |
| `webhook_url` | `str \| None` | `None` | URL to POST results to after each run |

Returns the schedule definition dict. Raises `ValueError` if name is empty, already exists, command is empty, or interval is not positive.

### Common Intervals

| Interval | Seconds | Use Case |
|----------|:-------:|----------|
| Hourly | `3600` | Latency checks, SLA monitoring |
| Every 6 hours | `21600` | Drift detection, data quality |
| Daily | `86400` | Full regression suite, compliance checks |
| Weekly | `604800` | Comprehensive bias audits, performance benchmarks |

### `should_run`

Check if a schedule is due to run.

```python
if scheduler.should_run("nightly-drift-check"):
    result = scheduler.execute("nightly-drift-check")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Schedule identifier to check |
| `current_time` | `float \| None` | `None` | Override for current timestamp (for testing). If `None`, uses `time.time()`. |

Returns `True` if the schedule exists and is due. Schedules that have never run (`last_run == 0.0`) are always due.

### `execute`

Execute a scheduled test run.

```python
result = scheduler.execute("nightly-drift-check")
print(f"Return code: {result['returncode']}")
print(f"Duration: {result['duration_seconds']:.1f}s")
print(f"Webhook sent: {result['webhook_sent']}")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Schedule identifier to execute |

Returns a result dictionary:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Schedule name |
| `command` | `str` | The command that was run |
| `returncode` | `int` | Process exit code (0 = success, -1 = timeout) |
| `stdout` | `str` | Captured standard output |
| `stderr` | `str` | Captured standard error |
| `executed_at` | `float` | Timestamp when the run started |
| `duration_seconds` | `float` | Wall-clock execution time |
| `webhook_sent` | `bool` | Whether webhook was delivered successfully |

Raises `ValueError` if no schedule with the given name exists. Commands timeout after 3600 seconds (1 hour).

### `list_schedules`

```python
for sched in scheduler.list_schedules():
    print(f"{sched['name']}: every {sched['interval_seconds']}s")
```

Returns a list of all schedule definition dicts.

### `remove_schedule`

```python
removed = scheduler.remove_schedule("nightly-drift-check")
# True if found and removed, False if not found
```

---

## Webhook Notifications

Each schedule can optionally specify a `webhook_url`. When a scheduled run completes, the full result dict is POSTed as JSON to that URL.

### Webhook Payload

```json
{
  "name": "nightly-drift-check",
  "command": "mltk run --tag drift",
  "returncode": 1,
  "stdout": "3 tests passed, 1 failed\n...",
  "stderr": "",
  "executed_at": 1711500000.0,
  "duration_seconds": 45.2,
  "webhook_sent": true
}
```

### Integration Examples

| Service | Webhook URL Format |
|---------|-------------------|
| Slack | `https://hooks.slack.com/services/T.../B.../...` |
| PagerDuty | `https://events.pagerduty.com/v2/enqueue` |
| OpsGenie | `https://api.opsgenie.com/v2/alerts` |
| Custom | Any HTTP endpoint accepting JSON POST |

Webhook delivery is best-effort (fire-and-forget). If the remote server is down, the failure is logged but not retried -- the test result is still recorded locally.

---

## Combined: Scheduled Runs with Portal Visualization

The scheduler and portal work together: scheduled runs produce results that the portal displays in real time.

### Setup

```python
from mltk.server.scheduler import TestScheduler
from mltk.server.portal import create_portal_html

# 1. Configure scheduled test runs
scheduler = TestScheduler()

scheduler.add_schedule(
    name="drift-check",
    command="mltk run --tag drift --export-json /data/results/drift.json",
    interval_seconds=21600,  # every 6 hours
)

scheduler.add_schedule(
    name="latency-check",
    command="mltk run --tag latency --export-json /data/results/latency.json",
    interval_seconds=3600,  # every hour
)

scheduler.add_schedule(
    name="full-regression",
    command="mltk run --export-json /data/results/regression.json",
    interval_seconds=86400,  # daily
    webhook_url="https://hooks.slack.com/services/T.../B.../...",
)

# 2. Generate portal pointing to the same server
portal_html = create_portal_html(
    server_url="http://ml-server:8080",
    refresh_seconds=30,
)
with open("/var/www/portal.html", "w") as f:
    f.write(portal_html)

# 3. Run the scheduler loop (in production, use cron or systemd)
import time

while True:
    for sched in scheduler.list_schedules():
        if scheduler.should_run(sched["name"]):
            result = scheduler.execute(sched["name"])
            print(f"[{sched['name']}] exit={result['returncode']} "
                  f"duration={result['duration_seconds']:.1f}s")
    time.sleep(60)  # check every minute
```

### Production Deployment

For production, avoid the Python loop and use system-level scheduling:

```bash
# crontab entry: check schedules every 5 minutes
*/5 * * * * cd /opt/mltk && python -c "
from mltk.server.scheduler import TestScheduler
import json, pathlib

state_file = pathlib.Path('/opt/mltk/scheduler-state.json')
scheduler = TestScheduler()

# Load schedules from config
for name, cfg in json.loads(pathlib.Path('schedules.json').read_text()).items():
    scheduler.add_schedule(name=name, **cfg)

# Run any due schedules
for sched in scheduler.list_schedules():
    if scheduler.should_run(sched['name']):
        scheduler.execute(sched['name'])
"
```

---

## pytest Integration

### Portal Generation Test

```python
from mltk.server.portal import create_portal_html

def test_portal_html_generation():
    """Portal HTML must be a complete, valid document."""
    html = create_portal_html(
        server_url="http://localhost:8080",
        refresh_seconds=10,
    )
    assert "<!DOCTYPE html>" in html
    assert "mltk Live Portal" in html
    assert "localhost:8080" in html
    assert "10 * 1000" in html  # interval in JS
```

### Scheduler Unit Tests

```python
import time
from mltk.server.scheduler import TestScheduler

def test_scheduler_should_run():
    """New schedules are immediately due."""
    scheduler = TestScheduler()
    scheduler.add_schedule(
        name="test-run",
        command="echo hello",
        interval_seconds=3600,
    )
    assert scheduler.should_run("test-run")  # never run before -> due

def test_scheduler_respects_interval():
    """Schedule is not due before interval elapses."""
    scheduler = TestScheduler()
    scheduler.add_schedule(
        name="test-run",
        command="echo hello",
        interval_seconds=3600,
    )
    result = scheduler.execute("test-run")
    assert result["returncode"] == 0

    # Not due yet (just ran)
    assert not scheduler.should_run("test-run")
```
