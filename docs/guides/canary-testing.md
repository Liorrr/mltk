# Canary Testing Guide

Deploy new models safely using mltk to validate canary deployments before full rollout.

---

## What is Canary Testing?

A canary deployment serves a small percentage of traffic to a new model version while production handles the rest. mltk validates the canary by comparing its outputs, latency, and error rates against the production baseline. If the canary fails — Slack alerts fire and you roll back before users are impacted.

```text
Traffic ──┬──[95%]──→ Production Model (v1.2)
          │
          └──[5%]───→ Canary Model (v1.3)
                         │
                    mltk validates:
                    ├─ Output drift vs production?
                    ├─ Latency within SLA?
                    ├─ Error rate acceptable?
                    └─ Bias metrics stable?
                         │
                    ┌────┴────┐
                    │  PASS   │  → Promote canary to 100%
                    │  FAIL   │  → Alert Slack → Rollback
                    └─────────┘
```

---

## Step 1: Collect Baseline (Production)

Run mltk against your production model to establish the baseline:

```python
import numpy as np
from mltk.monitor import assert_no_output_drift, assert_sla

# Collect production predictions
production_outputs = model_v1.predict(test_data)
np.save("baseline_outputs.npy", production_outputs)

# Baseline SLA
assert_sla(latency_p99=45.0, error_rate=0.002,
           thresholds={"latency_p99_ms": 100, "error_rate": 0.01})
```

---

## Step 2: Validate Canary

Run the same tests against the canary model and compare:

```python
import numpy as np
from mltk.monitor import assert_no_output_drift, assert_sla, assert_no_degradation
from mltk.model import assert_metric, assert_no_bias

# Load baseline
baseline = np.load("baseline_outputs.npy")

# Canary predictions
canary_outputs = model_v2.predict(test_data)

# 1. Output distribution hasn't shifted
assert_no_output_drift(baseline, canary_outputs, method="ks", threshold=0.05)

# 2. Accuracy hasn't regressed
assert_metric(y_true, canary_outputs > 0.5, metric="f1", threshold=0.85)

# 3. Latency within SLA
assert_sla(latency_p99=canary_latency_p99,
           thresholds={"latency_p99_ms": 100})

# 4. Bias hasn't increased
assert_no_bias(y_true, canary_outputs > 0.5,
               sensitive_feature=demographics,
               method="demographic_parity", threshold=0.1)
```

---

## Step 3: Wire Slack Alerts

Get notified immediately when a canary fails:

### Option A: pytest + Slack flag

```bash
# Run canary validation, alert Slack on failure
pytest tests/test_canary.py --mltk-export-json canary-results.json -q

# If failures, notify Slack
python -c "
import json
from mltk.integrations.slack import notify_slack
from mltk.core.result import TestSuite, TestResult, Severity

results = json.load(open('canary-results.json'))
failed = [r for r in results if not r.get('passed')]
if failed:
    suite = TestSuite()
    for r in results:
        suite.add(TestResult(
            name=r['name'], passed=r['passed'],
            severity=Severity(r.get('severity', 'critical')),
            message=r.get('message', ''),
        ))
    notify_slack(
        webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        suite=suite,
        message='CANARY ALERT: Model v1.3 canary failed validation!',
    )
"
```

### Option B: mltk server webhooks

Configure the server to auto-alert on failures:

```bash
# Register a webhook that fires on failure
curl -X POST http://localhost:8080/api/webhooks \
  -H "Authorization: Bearer mltk_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "events": ["on_failure"],
    "project": "canary-v1.3"
  }'

# Push canary results to server (webhook fires automatically)
pytest tests/test_canary.py --mltk-server http://localhost:8080 -q
```

---

## Step 4: CI/CD Integration

### GitHub Actions Canary Workflow

```yaml
name: Canary Validation
on:
  workflow_dispatch:
    inputs:
      canary_endpoint:
        description: 'Canary model endpoint URL'
        required: true

jobs:
  validate-canary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install mltk
        run: pip install mltk[all]

      - name: Run canary validation
        run: |
          pytest tests/test_canary.py \
            --mltk-report \
            --mltk-export-json canary-results.json \
            -q
        env:
          CANARY_ENDPOINT: ${{ inputs.canary_endpoint }}
          PRODUCTION_ENDPOINT: ${{ secrets.PROD_ENDPOINT }}

      - name: Alert on failure
        if: failure()
        run: |
          mltk notify slack \
            --webhook-url ${{ secrets.SLACK_WEBHOOK }} \
            --results-json canary-results.json

      - name: Compare with last production run
        if: always()
        run: |
          # Push to server for comparison
          pytest tests/test_canary.py \
            --mltk-server ${{ secrets.MLTK_SERVER }} \
            -q || true
```

---

## Step 5: Automated Rollback

Use webhook + your deployment tool to auto-rollback:

```python
# webhook_handler.py — receives mltk webhook payloads
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess

class CanaryHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))

        if data.get("failed", 0) > 0:
            print(f"CANARY FAILED: {data['failed']} failures")
            # Rollback canary deployment
            subprocess.run(["kubectl", "rollout", "undo", "deployment/model-canary"])
            # Or: AWS CodeDeploy stop + rollback
            # Or: GCP Traffic Manager shift to 0%
        else:
            print("CANARY PASSED: promoting to production")
            subprocess.run(["kubectl", "set", "image", "deployment/model",
                          f"model=myregistry/model:{data.get('version', 'latest')}"])

        self.send_response(200)
        self.end_headers()

HTTPServer(("0.0.0.0", 9090), CanaryHandler).serve_forever()
```

---

## Step 6: Scheduled Canary Monitoring

Run canary validation on a schedule (e.g., every 15 minutes):

```yaml
# GitHub Actions scheduled canary
name: Canary Monitor
on:
  schedule:
    - cron: '*/15 * * * *'  # Every 15 minutes

jobs:
  canary-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install mltk[all]
      - run: pytest tests/test_canary.py --mltk-server $MLTK_SERVER -q
        env:
          MLTK_SERVER: ${{ secrets.MLTK_SERVER }}
      - name: Alert on failure
        if: failure()
        run: mltk notify slack --webhook-url ${{ secrets.SLACK_WEBHOOK }} --results-json canary-results.json
```

---

## Complete Canary Test File

```python
"""tests/test_canary.py — Canary deployment validation suite."""

import os
import numpy as np
import pytest
from mltk.monitor import assert_no_output_drift, assert_sla
from mltk.model import assert_metric, assert_no_regression, assert_no_bias
from mltk.inference import assert_latency

# Load endpoints from environment
CANARY_URL = os.environ.get("CANARY_ENDPOINT", "http://localhost:8001")
PROD_URL = os.environ.get("PRODUCTION_ENDPOINT", "http://localhost:8000")


@pytest.fixture
def test_data():
    """Load standard test dataset."""
    return np.load("data/test_features.npy")


@pytest.fixture
def baseline_outputs():
    """Load production model baseline predictions."""
    return np.load("data/production_baseline.npy")


@pytest.mark.ml_model
def test_canary_output_drift(test_data, baseline_outputs):
    """Canary outputs should match production distribution."""
    canary_outputs = call_model(CANARY_URL, test_data)
    assert_no_output_drift(baseline_outputs, canary_outputs,
                           method="ks", threshold=0.05)


@pytest.mark.ml_model
def test_canary_accuracy(test_data):
    """Canary accuracy should not regress from production."""
    y_true = np.load("data/test_labels.npy")
    canary_preds = call_model(CANARY_URL, test_data)
    assert_metric(y_true, canary_preds > 0.5,
                  metric="f1", threshold=0.85)


@pytest.mark.ml_inference
def test_canary_latency():
    """Canary latency must meet SLA."""
    assert_latency(
        lambda: call_model(CANARY_URL, sample_input()),
        p95=100.0, warmup=5
    )


@pytest.mark.ml_model
def test_canary_bias(test_data):
    """Canary should not introduce new bias."""
    y_true = np.load("data/test_labels.npy")
    demographics = np.load("data/test_demographics.npy")
    canary_preds = call_model(CANARY_URL, test_data)
    assert_no_bias(y_true, canary_preds > 0.5,
                   sensitive_feature=demographics,
                   method="demographic_parity", threshold=0.1)


def call_model(url, data):
    """Helper: call model endpoint and return predictions."""
    import urllib.request
    import json
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"features": data.tolist()}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return np.array(json.loads(resp.read())["predictions"])


def sample_input():
    """Helper: single sample for latency testing."""
    return np.random.randn(1, 128)
```

---

## Key Assertions for Canary Testing

| Assertion | What it checks | When to use |
|-----------|---------------|-------------|
| `assert_no_output_drift` | Output distribution shift | Always — primary canary gate |
| `assert_metric` | Accuracy/F1/AUC | Always — quality gate |
| `assert_no_regression` | Score vs baseline | When baseline metric is known |
| `assert_latency` | P95/P99 response time | Always — performance gate |
| `assert_sla` | Combined latency + error rate | Production SLA compliance |
| `assert_no_bias` | Fairness metrics | Regulated/sensitive models |
| `assert_no_degradation` | Sliding window decline | Long-running canary |

---
