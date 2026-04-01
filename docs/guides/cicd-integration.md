# CI/CD Integration Guide

Wire mltk into your CI/CD pipeline to catch ML issues before they reach production.

---

## GitHub Actions

Add to `.github/workflows/ml-tests.yml` in your project:

```yaml
name: ML Tests
on: [push, pull_request]

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

      - name: Run ML tests
        run: |
          pytest --mltk-report --mltk-export-json results.json -q

      - name: Upload report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: mltk-report
          path: mltk-reports/

      - name: Post PR comment
        if: github.event_name == 'pull_request'
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python -c "
          from mltk.server.github_ci import post_pr_comment
          import json, os
          results = json.load(open('results.json'))
          if isinstance(results, list):
              data = {'total': len(results), 'passed': sum(1 for r in results if r.get('passed')), 'failed': sum(1 for r in results if not r.get('passed')), 'score': 0, 'results': results}
              data['score'] = data['passed'] / data['total'] * 100 if data['total'] else 0
          else:
              data = results
          post_pr_comment(
              repo=os.environ['GITHUB_REPOSITORY'],
              pr_number=int(os.environ.get('PR_NUMBER', '0')),
              results=data,
              token=os.environ['GITHUB_TOKEN'],
          )
          "
```

---

## GitLab CI

Add to `.gitlab-ci.yml`:

```yaml
ml-tests:
  stage: test
  image: python:3.12
  script:
    - pip install mltk[all]
    - pytest --mltk-report --mltk-export-json results.json -q
  artifacts:
    when: always
    paths:
      - mltk-reports/
      - results.json
    expire_in: 30 days
```

---

## Jenkins

Add to your `Jenkinsfile`:

```groovy
pipeline {
    agent { docker { image 'python:3.12' } }

    stages {
        stage('Install') {
            steps {
                sh 'pip install mltk[all]'
                sh 'pip install -r requirements.txt'
            }
        }
        stage('ML Tests') {
            steps {
                sh 'pytest --mltk-report --mltk-export-json results.json -q'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'mltk-reports/*, results.json', allowEmptyArchive: true
                }
            }
        }
        stage('Compliance') {
            steps {
                sh 'mltk compliance results.json --risk-level high'
                sh 'mltk fda-audit results.json'
                archiveArtifacts artifacts: 'fda-audit-trail.md'
            }
        }
    }
}
```

---

## Azure DevOps

Add to `azure-pipelines.yml`:

```yaml
trigger:
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

  - task: PublishBuildArtifacts@1
    condition: always()
    inputs:
      pathToPublish: 'mltk-reports'
      artifactName: 'mltk-reports'
```

---

## With mltk Server

Push results to a central mltk server for team dashboards:

```yaml
# Add to any CI pipeline after tests run:
- name: Push to mltk server
  run: |
    pytest --mltk-report --mltk-export-json results.json --mltk-server $MLTK_SERVER_URL -q
  env:
    MLTK_SERVER_URL: https://mltk.internal.company.com
```

Or use the pytest flag directly:

```bash
pytest --mltk-server https://mltk.internal.company.com --mltk-report
```

---

## Environment Variables

Configure mltk in CI without config files:

```yaml
env:
  MLTK_DRIFT_METHOD: psi
  MLTK_DRIFT_THRESHOLD: "0.1"
  MLTK_REPORT_DIR: ./mltk-reports
  MLTK_PII_PATTERNS: email,phone,ssn,credit_card
```

---

## Test Tiering

Run different test suites for different triggers:

```yaml
# Smoke tests on every push (fast)
- pytest -m ml_smoke --mltk-report -q

# Full suite on PR (thorough)
- pytest --mltk-report --mltk-export-json results.json -q

# Drift + monitoring on schedule (nightly)
- pytest -m "ml_drift or ml_model" --mltk-report -q
```

Markers available:
- `ml_data` — data quality tests
- `ml_model` — model quality tests
- `ml_drift` — drift detection
- `ml_inference` — performance benchmarks
- `ml_smoke` — quick sanity checks
- `ml_slow` — long-running tests
- `ml_gpu` — GPU-required tests

---

## Compliance Reports in CI

Generate compliance artifacts automatically:

```yaml
- name: EU AI Act compliance
  run: |
    pytest --mltk-export-json results.json -q
    mltk compliance results.json --risk-level high --system-name "My Model"
    mltk fda-audit results.json --system-name "My Model" --operator "CI Bot"
    mltk compliance-pdf mltk-reports/eu-ai-act-*.html

- name: Upload compliance docs
  uses: actions/upload-artifact@v4
  with:
    name: compliance-reports
    path: |
      mltk-reports/
      fda-audit-trail.md
```

---

## Docker

Run mltk tests in a container:

```dockerfile
FROM python:3.12-slim
RUN pip install mltk[all]
COPY . /app
WORKDIR /app
CMD ["pytest", "--mltk-report", "--mltk-export-json", "results.json", "-q"]
```

---

## Security Scan in CI

Add `mltk security-scan` as a CI gate to red team your LLM on every build:

```yaml
# GitHub Actions -- add after ML tests
- name: Red team security scan
  run: |
    mltk security-scan myapp.llm:chat_fn \
      --attacks owasp-top7 \
      --mutations \
      --threshold 0.9 \
      --format json \
      --output security-scan.json

- name: Upload security report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: security-scan
    path: security-scan.json
```

The command exits with code 1 when the model fails the threshold, blocking the merge.
For multi-turn adaptive attacks, add `--strategy multi-turn --max-turns 5`.

See [security-scan CLI](../api/security-scan.md) for the full option reference and
[Red Team Framework](../api/red-team.md) for the assertion API.

---

## YAML-Driven Tests in CI

No Python code needed — QA teams write YAML:

```yaml
# In CI pipeline:
- name: Run ML quality checks
  run: mltk test tests/ml-quality.yaml
```

Where `tests/ml-quality.yaml`:
```yaml
data_source: env:TRAINING_DATA_PATH
tests:
  - name: Schema check
    assert: schema
    expected: { id: int64, score: float64 }
  - name: No PII
    assert: no_pii
    columns: [email_field, notes]
  - name: Row count
    assert: row_count
    min_rows: 1000
```

---
