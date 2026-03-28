# HTML Reports

Generate self-contained interactive HTML reports from test results. Dark theme by default, pure CSS/SVG charts (no external dependencies).

**Module:** `mltk.report`

---

## generate_report

```python
from mltk.report import generate_report

path = generate_report(results, output_dir="./mltk-reports", title="MLTK Test Report")
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results from MltkReportCollector |
| `output_dir` | `str` | `"./mltk-reports"` | Directory for HTML output |
| `title` | `str` | `"MLTK Test Report"` | Report title |

### Output

Single self-contained HTML file with:
- Pass/fail summary with counts
- **Pass/fail donut chart** (SVG ring with green/red segments, percentage in center)
- **Module breakdown bar chart** (horizontal stacked bars per module, pass/fail counts)
- Per-module breakdown
- Test details table with severity, duration, messages
- Dark theme (slate background, purple accent)

Charts are pure CSS/SVG — no external dependencies, no CDN scripts, no Plotly. The report is fully self-contained and works offline.

### pytest Integration

```bash
pytest --mltk-report
# Generates: ./mltk-reports/report-20260325-120000.html
```

---

## JUnit XML Export

Export test results in JUnit XML format for CI/CD integration. Jenkins, GitLab CI, Azure DevOps, CircleCI, and most CI/CD systems parse JUnit XML natively -- they display test results in dashboards, track trends across builds, and gate deployments on pass/fail status.

**Module:** `mltk.report.junit`

### Why JUnit XML

mltk's native JSON and HTML reports are great for humans and for the mltk server platform, but CI/CD dashboards cannot read them. JUnit XML is the universal language of test reporting:

| CI/CD System | JUnit XML support |
|-------------|-------------------|
| **Jenkins** | Native (JUnit plugin, Test Results Analyzer) |
| **GitLab CI** | Native (`artifacts:reports:junit`) |
| **Azure DevOps** | Native (PublishTestResults task) |
| **CircleCI** | Native (`store_test_results`) |
| **GitHub Actions** | Via third-party actions (dorny/test-reporter, mikepenz/action-junit-report) |

### export_junit_xml

```python
from mltk.report.junit import export_junit_xml

results = [
    {
        "name": "data.schema.check",
        "passed": True,
        "duration_ms": 50.0,
        "message": "Schema matches expected dtypes",
    },
    {
        "name": "model.metric.accuracy",
        "passed": False,
        "duration_ms": 120.0,
        "message": "accuracy 0.75 < 0.80 threshold",
    },
    {
        "name": "data.drift.psi",
        "passed": True,
        "duration_ms": 85.0,
        "message": "PSI 0.03 < 0.10 threshold",
    },
]

path = export_junit_xml(results, output_path="mltk-results.xml", suite_name="mltk")
# Returns: absolute path to the written XML file
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | List of result dicts. Each dict must have `name` (str) and `passed` (bool). Optional keys: `duration_ms` (float), `message` (str). |
| `output_path` | `str` | `"mltk-results.xml"` | Destination file path for the XML output. Parent directories are created automatically. |
| `suite_name` | `str` | `"mltk"` | Value for the `<testsuite name="...">` attribute. Use this to distinguish mltk results from other test suites in CI dashboards. |

#### Returns

The absolute path to the written XML file as a string.

### XML Structure

The generated XML follows the JUnit standard:

```xml
<?xml version='1.0' encoding='utf-8'?>
<testsuites>
  <testsuite name="mltk" tests="3" failures="1" errors="0" time="0.255">
    <testcase name="data.schema.check" classname="mltk.data.schema" time="0.050000">
    </testcase>
    <testcase name="model.metric.accuracy" classname="mltk.model.metric" time="0.120000">
      <failure message="accuracy 0.75 &lt; 0.80 threshold" type="MltkAssertionError">
        accuracy 0.75 &lt; 0.80 threshold
      </failure>
    </testcase>
    <testcase name="data.drift.psi" classname="mltk.data.drift" time="0.085000">
    </testcase>
  </testsuite>
</testsuites>
```

Key mapping:

| mltk result field | JUnit XML attribute |
|-------------------|---------------------|
| `name` | `<testcase name="...">` and derived `classname` |
| `passed` | Presence/absence of `<failure>` child element |
| `duration_ms` | `<testcase time="...">` (converted to seconds) |
| `message` | `<failure message="...">` text content |

The `classname` is derived automatically from the dotted test name: `"data.schema.check"` becomes `"mltk.data.schema"` (module path prefix). If there is no dot, classname defaults to `"mltk"`.

### End-to-end workflow with pytest

```bash
# Step 1: Run tests and export JSON results
pytest --mltk-export-json results.json

# Step 2: Convert to JUnit XML
python -c "
from mltk.report.junit import export_junit_xml
import json

with open('results.json') as f:
    results = json.load(f)

path = export_junit_xml(results, 'mltk-results.xml')
print(f'JUnit XML written to: {path}')
"
```

### CI Integration: Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('ML Tests') {
            steps {
                sh 'pip install mltk[cli]'
                sh 'pytest --mltk-export-json results.json tests/'
                sh '''python -c "
from mltk.report.junit import export_junit_xml
import json
with open('results.json') as f:
    results = json.load(f)
export_junit_xml(results, 'mltk-results.xml')
"'''
            }
            post {
                always {
                    junit 'mltk-results.xml'
                }
            }
        }
    }
}
```

### CI Integration: GitLab CI

```yaml
ml-tests:
  stage: test
  script:
    - pip install mltk[cli]
    - pytest --mltk-export-json results.json tests/
    - python -c "
        from mltk.report.junit import export_junit_xml;
        import json;
        results = json.load(open('results.json'));
        export_junit_xml(results, 'mltk-results.xml')
      "
  artifacts:
    when: always
    reports:
      junit: mltk-results.xml
```

### CI Integration: GitHub Actions

```yaml
- name: Run ML tests
  run: |
    pip install mltk[cli]
    pytest --mltk-export-json results.json tests/
    python -c "
    from mltk.report.junit import export_junit_xml
    import json
    results = json.load(open('results.json'))
    export_junit_xml(results, 'mltk-results.xml')
    "

- name: Publish test results
  uses: dorny/test-reporter@v1
  if: always()
  with:
    name: ML Test Results
    path: mltk-results.xml
    reporter: java-junit
```

---
