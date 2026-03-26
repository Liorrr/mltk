# VS Code Integration Guide

Use mltk with Visual Studio Code for ML test automation.

---

## 1. pytest Test Explorer

mltk's pytest plugin auto-registers on install. VS Code's Python extension discovers mltk tests automatically.

**Setup:**
1. Install Python extension (`ms-python.python`)
2. Open your mltk project in VS Code
3. Tests appear in the Testing sidebar (beaker icon)
4. Run tests with play button — mltk markers (`@pytest.mark.ml_data`, etc.) show as tags

**settings.json:**
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["--mltk-report"]
}
```

---

## 2. HTML Reports in VS Code

After running tests with `--mltk-report`, open the HTML file:
- Right-click the HTML file in Explorer → "Open with Live Server" (if Live Server extension installed)
- Or: `Ctrl+Shift+P` → "Simple Browser: Show" → paste the file path

---

## 3. Tasks for Common Commands

Add to `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "mltk: Run Tests + Report",
            "type": "shell",
            "command": "pytest --mltk-report --mltk-export-json results.json",
            "group": "test"
        },
        {
            "label": "mltk: Doctor",
            "type": "shell",
            "command": "mltk doctor"
        },
        {
            "label": "mltk: Start Server",
            "type": "shell",
            "command": "mltk server --port 8080",
            "isBackground": true
        },
        {
            "label": "mltk: Scan Data",
            "type": "shell",
            "command": "mltk scan ${input:dataFile}"
        }
    ]
}
```

---

## 4. Live Dashboard

Start the mltk server for a live dashboard:

1. Run `mltk server` in the terminal
2. Open Simple Browser: `Ctrl+Shift+P` → "Simple Browser: Show" → `http://localhost:8080`
3. The dashboard shows test history, trends, and run details
4. Configure `--mltk-server http://localhost:8080` in pytest to auto-push results

---

## 5. Recommended Extensions

| Extension | Why |
|-----------|-----|
| **Python** (`ms-python.python`) | Test discovery, debugging, linting |
| **Python Test Explorer** | Tree view of mltk test markers |
| **Live Server** | View HTML reports with hot reload |
| **REST Client** | Test server API endpoints directly |
| **Docker** | Manage mltk server containers |

---
