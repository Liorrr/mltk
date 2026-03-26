# Resource Summarization

Analyze test run history to surface trends, recurring failures, flaky tests, and actionable recommendations.

**Module:** `mltk.report.summarizer`

---

## Overview

After accumulating multiple test runs, the summarizer answers three key questions:

1. **Is quality improving, degrading, or stable?** -- compares the average score of the earliest runs against the most recent.
2. **Which tests fail most often?** -- ranks failures by frequency so you fix the biggest pain points first.
3. **Which tests are flaky?** -- identifies tests that sometimes pass and sometimes fail, eroding trust in the suite.

The summarizer works both as a **Python function** (for scripts and notebooks) and as a **REST endpoint** on the mltk server.

---

## Python API

```python
from mltk.report import summarize_test_history

runs = [
    {
        "score": 72.0,
        "passed": 7,
        "failed": 3,
        "total": 10,
        "timestamp": "2025-06-01T10:00:00",
        "results": [
            {"name": "data.schema.types", "passed": True},
            {"name": "model.metric.accuracy", "passed": False},
            {"name": "data.drift.psi", "passed": True},
            # ...
        ],
    },
    # ... more runs
]

summary = summarize_test_history(runs)
print(summary)
```

### Return Value

| Key | Type | Description |
|-----|------|-------------|
| `trend` | `str` | `"improving"`, `"degrading"`, or `"stable"` |
| `avg_score` | `float` | Mean score across all runs |
| `most_common_failures` | `list[tuple[str, int]]` | `(test_name, fail_count)` sorted descending |
| `flaky_tests` | `list[str]` | Tests that passed in some runs and failed in others |
| `recommendations` | `list[str]` | Human-readable action items |

### Trend Detection

The trend is computed by comparing the average score of the **first 3 runs** (chronologically) against the **last 3 runs**. A difference greater than 2 percentage points triggers `"improving"` or `"degrading"`; otherwise the trend is `"stable"`.

---

## Server Endpoint

```
GET /summary/{project}?limit=20
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | *(required)* | Project name |
| `limit` | `int` | `20` | Maximum number of recent runs to analyze |

### Example Request

```bash
curl http://localhost:8080/summary/my-project?limit=10
```

### Example Response

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

## Edge Cases

- **No runs**: Returns `trend: "stable"`, `avg_score: 0.0`, empty failures/flaky lists.
- **Single run**: Trend is always `"stable"` (not enough data for comparison).
- **Runs without `results` list**: Trend and average score still work; failure/flaky analysis is skipped for those runs.
