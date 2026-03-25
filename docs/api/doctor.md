# mltk doctor

Auto-diagnose your ML testing environment. Run on Day 1 at a new company to find issues.

**Module:** `mltk.doctor`

**CLI:** `mltk doctor`

---

## Quick Start

```bash
mltk doctor
```

Output:
```
mltk doctor
  [OK]   Python version: 3.12.1 (>= 3.10)
  [OK]   Core deps: numpy 1.26.4, pandas 2.2.0
  [WARN] Optional dep: scipy not installed (pip install mltk[scipy])
  [OK]   Config: mltk.yaml found
  [OK]   Report dir: ./mltk-reports exists
  [FAIL] Baseline dir: ./mltk-baselines not found
  [OK]   Rust extension: loaded
  [OK]   pytest plugin: registered
  [OK]   Config valid: no misconfigurations
```

---

## Diagnostic Checks

| # | Check | OK | WARN | FAIL |
|---|-------|-----|------|------|
| 1 | Python version | >= 3.10 | — | < 3.10 |
| 2 | Core dependencies | numpy + pandas installed | — | Missing |
| 3 | Optional dependencies | All installed | Some missing | — |
| 4 | Config file | Found + parseable | Not found | Parse error |
| 5 | Report directory | Exists + writable | — | Not writable |
| 6 | Baseline directory | Exists | Not found | — |
| 7 | Rust extension | Loaded | Not available | — |
| 8 | pytest plugin | Registered | — | Not found |
| 9 | Config validation | No issues | Suspicious values | — |

Each check returns a **fix hint** when it fails or warns.

---

## Python API

```python
from mltk.doctor import diagnose

results = diagnose()
for r in results:
    print(f"[{r.status}] {r.name}: {r.message}")
    if r.fix_hint:
        print(f"  Fix: {r.fix_hint}")
```

---
