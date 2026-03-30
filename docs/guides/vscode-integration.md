# VS Code Integration

mltk ships a dedicated VS Code extension (**mltk-vscode**) that brings ML
test results, model health scanning, and failure inspection directly into
your editor. No terminal switching, no context loss.

---

## Installation

Install from a local `.vsix` package (not yet on Marketplace):

```bash
code --install-extension mltk-vscode-0.4.0.vsix
```

The extension activates automatically when it detects `mltk.yaml`,
`pyproject.toml`, or `*-tests.yaml` in your workspace.

---

## Features at a Glance

| Feature | What it does |
|---------|-------------|
| **Test Results tree** | Sidebar showing pass/fail for every mltk assertion |
| **Model Health tree** | Scan findings grouped by severity (Critical/Warning/Info) |
| **Test Inspector panel** | Webview with failure details, trend sparkline, one-click actions |
| **Native Test Explorer** | VS Code Testing API integration with per-test execution |
| **CodeLens** | Inline `PASS`/`FAIL` badges above test functions |
| **Gutter decorations** | Green/red icons in the editor margin |
| **Hover tooltips** | Full result details on hover over test functions |
| **Dashboard** | Run history, score trends, summary cards |
| **Model scan** | Right-click a model file to run all 8 scanners |
| **Auto-run on save** | Re-run tests automatically when you save a Python file |

---

## 1. Running Tests

### From the Command Palette

`Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and type:

- **mltk: Run Tests** -- runs all mltk tests via pytest
- **mltk: View Results** -- focuses the Test Results sidebar

### From the Test Explorer

The extension registers a native **TestController** with VS Code's Testing
API. Tests appear in the standard Test Explorer sidebar (beaker icon):

```
Test Explorer
  test_model.py
    TestAccuracy
      test_overall_accuracy
      test_slice_age
    test_drift_detection
  test_data.py
    test_no_nulls
    test_no_pii
```

You can run individual tests, classes, or entire files by clicking the
play button next to each item. Results map back to the tree with
pass/fail/skip states.

### Auto-run on Save

Enable automatic test execution when saving Python files:

```json
// .vscode/settings.json
{
  "mltk.autoRun": true
}
```

---

## 2. Inline Results

### CodeLens

After running tests, every `test_*` function gets inline CodeLens
annotations:

```python
# [Run mltk test]  [PASS (12.3ms)]
def test_accuracy(model, data):
    ...

# [Run mltk test]  [FAIL: Accuracy 0.75 < 0.80 threshold]
def test_slice_performance(model, data):
    ...
```

- **PASS** lenses link to the results sidebar
- **FAIL** lenses open the **Test Inspector** panel with full details

### Gutter Decorations

Green checkmark or red X icons appear in the editor gutter next to
test functions, providing an at-a-glance status view.

### Hover Tooltips

Hover over a test function to see a rich Markdown tooltip with the
test name, status, severity, duration, and all detail key-value pairs.

---

## 3. Test Inspector Panel

The **killer feature**. Click any failed test or scan finding to open a
webview panel beside your code showing everything needed to understand
and fix the failure.

### What you see

- **Header**: Severity badge (color-coded accent strip), test name,
  failure message, duration, and scanner name (for scan findings)
- **Details table**: All key-value pairs from the test result
- **Trend sparkline**: 7-day score history from the mltk server
  (muted placeholder when server is offline)
- **Action buttons**:
  - **Copy Test Skeleton** -- copies an assertion template to clipboard
  - **Generate Test** -- opens a new Python file with test code
  - **Jump to Code** -- navigates to the test function in your source
  - **Re-run Test** -- runs the test again (TestResult only)

### Entry points

The Inspector opens from four places:

1. Click a failed test in the **Test Results** tree
2. Click a finding in the **Model Health** tree
3. Click a `FAIL` CodeLens above a test function
4. Right-click context menu (coming soon)

### Conditional actions

| Action | Test Result | Scan Finding |
|--------|:-----------:|:------------:|
| Copy Test Skeleton | Available | Available (from suggested_test) |
| Generate Test | Available | Available (from suggested_test) |
| Jump to Code | Available | Hidden (no source file) |
| Re-run Test | Available | Hidden (not a test) |
| Trend sparkline | Available | Hidden (no history) |

---

## 4. Model Health Scanning

Right-click any model file (`.pkl`, `.joblib`, `.onnx`, `.pt`, `.h5`,
`.keras`) in the Explorer and select **mltk: Run Model Health Scan**.

The extension:

1. Asks you to select a test data file (CSV or Parquet)
2. Asks for the target column name
3. Runs all 8 built-in scanners with a progress notification
4. Populates the **Model Health** tree with findings grouped by severity
5. Shows a notification with finding count and severity breakdown

### Model Health Tree

```
Model Health
  X Critical (2)
    Accuracy drops to 0.58 for age > 55    [SliceScanner]
    Demographic parity violation on gender  [BiasScanner]
  ! Warning (1)
    Model uncalibrated (ECE: 0.15)          [CalibrationScanner]
  i Info (0)
  Scan Info
    Model: classifier
    Samples: 10,000
    Features: 15
    Duration: 12.3s
    Scanners: Data, Drift, Leakage, Slice, Bias, Calibration, Robustness
```

Click any finding to open the **Test Inspector** panel. Right-click
a finding to **Copy Test** (copies the suggested pytest code to
clipboard).

### Cancellation

Click the cancel button on the progress notification to abort a
long-running scan. The subprocess is terminated immediately.

---

## 5. Dashboard

Open the dashboard via `Ctrl+Shift+P` > **mltk: Open Dashboard** or
the graph icon in the editor title bar.

When connected to an mltk server, the dashboard shows:

- **Summary cards**: Latest score, average score, total runs, total failures
- **Score trend chart**: Bar chart of recent run scores
- **Run history table**: ID, project, timestamp, passed/failed/score

When no server is configured, it shows setup instructions.

---

## 6. Server Configuration

Connect to an mltk server for trend tracking, run history, and
the dashboard:

```json
// .vscode/settings.json
{
  "mltk.serverUrl": "http://localhost:8080",
  "mltk.apiKey": "your-api-key"
}
```

**API key security**: On first activation with an API key in settings,
the extension automatically migrates it to VS Code's
**SecretStorage** (encrypted, OS keychain-backed) and clears the
plaintext setting. After migration, the key is never stored in
`settings.json` again.

---

## 7. Configuration Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `mltk.pythonPath` | `string` | auto-detect | Path to Python interpreter |
| `mltk.serverUrl` | `string` | `""` | mltk server URL (e.g., `http://localhost:8080`) |
| `mltk.apiKey` | `string` | `""` | API key (migrates to SecretStorage on first use) |
| `mltk.markers` | `string[]` | `[]` | pytest markers to filter (e.g., `["ml_data", "ml_model"]`) |
| `mltk.autoRun` | `boolean` | `false` | Re-run tests on Python file save |

### Python interpreter detection

The extension finds Python in this order:

1. `mltk.pythonPath` setting (if set)
2. Python extension's active interpreter (`ms-python.python`)
3. System `python3` / `python` on PATH

---

## 8. Commands

| Command | Title | Description |
|---------|-------|-------------|
| `mltk.runTests` | Run Tests | Execute all mltk tests via pytest |
| `mltk.viewResults` | View Results | Focus the Test Results sidebar |
| `mltk.openDashboard` | Open Dashboard | Open the run history dashboard |
| `mltk.scanModel` | Run Model Health Scan | Scan a model file for issues |
| `mltk.openTestInspector` | Open Test Inspector | Open failure detail panel |
| `mltk.copyScanTest` | Copy Test | Copy suggested test for a scan finding |
| `mltk.showResultDetail` | Show Result Detail | Show result in output channel |
| `mltk.showScanDetail` | Show Scan Finding Detail | Show finding in output channel |

---

## 9. Workflow Example

A typical workflow combining mltk-vscode features:

```
1. Open your ML project in VS Code
2. Run tests (Ctrl+Shift+P > mltk: Run Tests)
3. See results in the Test Results tree (sidebar)
4. Click a failed test -> Test Inspector opens beside your code
5. Read the failure details and trend sparkline
6. Click "Jump to Code" -> navigate to the assertion
7. Fix the issue in your code
8. Click "Re-run Test" -> verify the fix
9. Right-click your model file -> Run Model Health Scan
10. Review findings in Model Health tree
11. Click "Copy Test Skeleton" -> paste into your test file
12. Commit the generated tests to your repo
```

---

## 10. Compatibility

| Requirement | Minimum |
|-------------|---------|
| VS Code | 1.85.0+ |
| Python | 3.10+ |
| mltk | 0.6.0+ (installed in workspace Python environment) |
| Node.js | Not required (extension is pre-built) |

### Recommended companion extensions

| Extension | Why |
|-----------|-----|
| **Python** (`ms-python.python`) | Test discovery, debugging, linting |
| **Live Server** | View HTML reports with hot reload |
| **REST Client** | Test server API endpoints directly |
