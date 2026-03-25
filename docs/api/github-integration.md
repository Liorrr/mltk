# GitHub, Slack & Plugin System

Extend mltk with GitHub Issues ticketing, Slack failure alerts, and a third-party plugin registry.

---

## GitHub Issues

**Module:** `mltk.integrations.github_adapter`

Create, search, and update GitHub Issues automatically from ML test failures.
No extra dependencies — uses Python's stdlib `urllib`.

### Quick Start

```python
from mltk.integrations import GitHubIssuesAdapter

adapter = GitHubIssuesAdapter("myorg/ml-service", token="ghp_...")

# Create an issue
url = adapter.create_issue(
    project="ml-service",          # ignored for GitHub; repo set at construction
    title="Drift detected: feature_age",
    description="PSI score 0.35 exceeds threshold 0.2",
    fields={"labels": ["ml-drift", "auto"], "assignees": ["ml-bot"]},
)
print(url)  # https://github.com/myorg/ml-service/issues/42

# Search issues
issues = adapter.search_issues("drift is:open")
# [{"key": "42", "summary": "Drift detected: feature_age", "url": "...", "state": "open"}]

# Update an issue
adapter.update_issue("42", {"state": "closed", "comment": "Resolved in v2.3"})
```

### Authentication

Provide a token directly or set the `GITHUB_TOKEN` environment variable:

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

```python
# Token from environment automatically
adapter = GitHubIssuesAdapter("myorg/ml-service")
```

### API

#### `GitHubIssuesAdapter(repo, token=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo` | `str` | Repository in `owner/name` format |
| `token` | `str \| None` | PAT; falls back to `GITHUB_TOKEN` env var |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `create_issue(project, title, description, fields=None)` | `str` | Creates issue, returns HTML URL |
| `search_issues(query)` | `list[dict]` | Searches via GitHub search API |
| `update_issue(issue_id, updates)` | `bool` | Updates issue fields or adds comment |

**`fields` keys for `create_issue`:**

| Key | Type | Description |
|-----|------|-------------|
| `labels` | `list[str]` | Labels to attach |
| `assignees` | `list[str]` | GitHub usernames to assign |
| `milestone` | `int` | Milestone ID |

**`updates` keys for `update_issue`:**

| Key | Type | Description |
|-----|------|-------------|
| `comment` | `str` | Post a new comment |
| `state` | `str` | `"open"` or `"closed"` |
| `labels` | `list[str]` | Replace labels |
| `title` | `str` | Update issue title |
| `body` | `str` | Update issue body |

---

## Slack Notifications

**Module:** `mltk.integrations.slack`

Post test failure summaries to Slack via an incoming webhook. No external deps.

### Quick Start

```python
from mltk.integrations import notify_slack

# Auto-summary from a TestSuite
notify_slack(webhook_url="https://hooks.slack.com/services/...", suite=my_suite)

# Custom message
notify_slack(webhook_url="https://hooks.slack.com/services/...", message="Nightly tests passed")
```

### CLI

```bash
# Send results from a JSON export
mltk notify slack --results-json mltk-reports/results.json

# Use environment variable for webhook URL
export MLTK_SLACK_WEBHOOK=https://hooks.slack.com/services/...
mltk notify slack --results-json mltk-reports/results.json

# Send a plain message
mltk notify slack --message "Drift detected in production — check dashboard"
```

### Message Format

When a `TestSuite` is provided, the Slack message includes:

- **Header**: "mltk Test Results"
- **Summary**: pass/fail counts and score percentage
- **Failures list**: up to 20 failed test names (bullet list)
- **Colour border**: green if all tests pass, red if any fail

### API

#### `notify_slack(webhook_url, suite=None, message=None, channel=None) -> bool`

| Parameter | Type | Description |
|-----------|------|-------------|
| `webhook_url` | `str` | Slack incoming webhook URL |
| `suite` | `TestSuite \| None` | Suite to summarise |
| `message` | `str \| None` | Custom text (appended as context if suite also provided) |
| `channel` | `str \| None` | Override webhook's default channel |

Returns `True` if Slack responded with HTTP 200, `False` otherwise.

#### `format_slack_message(suite) -> dict`

Returns a Block Kit webhook payload dict from a `TestSuite`. Use this to
inspect or customise the payload before sending:

```python
from mltk.integrations.slack import format_slack_message
import json

payload = format_slack_message(my_suite)
print(json.dumps(payload, indent=2))
```

---

## Plugin System

**Module:** `mltk.core.plugin`

Register and discover custom assertion functions from third-party packages.

### Registering Assertions

```python
from mltk.core.plugin import register_assertion

@register_assertion("assert_gini_below")
def assert_gini_below(series, threshold=0.4):
    gini = compute_gini(series)
    assert gini < threshold, f"Gini {gini:.3f} >= {threshold}"

# Or let mltk infer the name from the function name:
@register_assertion()
def assert_coverage_ratio(df, min_ratio=0.95):
    ratio = df.notna().mean().mean()
    assert ratio >= min_ratio, f"Coverage {ratio:.2%} < {min_ratio:.2%}"
```

### Discovering Installed Plugins

Third-party packages named `mltk_plugin_*` are discovered automatically:

```python
from mltk.core.plugin import discover_plugins, get_registered_assertions

# Import all installed mltk plugins (triggers @register_assertion decorators)
found = discover_plugins()
print(found)  # ['mltk_plugin_finance', 'mltk_plugin_cv']

# List all registered assertions
assertions = get_registered_assertions()
print(list(assertions.keys()))
# ['assert_gini_below', 'assert_coverage_ratio', ...]
```

### Building a Plugin Package

```
mltk_plugin_finance/
  __init__.py          ← @register_assertion decorators here
  assertions.py
  pyproject.toml       ← name = "mltk_plugin_finance"
```

```python
# mltk_plugin_finance/__init__.py
from mltk.core.plugin import register_assertion

@register_assertion("assert_sharpe_ratio")
def assert_sharpe_ratio(returns, min_sharpe=1.0):
    ...
```

### API

| Function | Description |
|----------|-------------|
| `register_assertion(name=None)` | Decorator — registers a function in the global registry |
| `get_registered_assertions()` | Returns `dict[str, Callable]` of all registered assertions |
| `discover_plugins(package_prefix="mltk_plugin_")` | Imports all matching installed packages, returns names found |

---
