# FDA Audit Trail

Generate FDA 21 CFR Part 11 compliant audit trails from mltk test results.

**Module:** `mltk.compliance`

**CLI:** `mltk fda-audit results.json --system-name "ML Pipeline" --operator "QA Lead"`

---

## Quick Start

```bash
pytest --mltk-export-json results.json
mltk fda-audit results.json --system-name "Medical AI v2" --operator "Dr. Smith"
```

## Generated Sections

1. **System Information** — name, date, generator
2. **Operator Identification** — name, timestamp
3. **Test Evidence** — pass/fail table with severity and messages
4. **Digital Signature** — placeholder for manual/electronic signature
5. **Regulatory Notes** — 21 CFR Part 11 compliance statement

## Python API

```python
from mltk.compliance import generate_fda_audit_trail

generate_fda_audit_trail(
    results_path="results.json",
    system_name="Medical AI v2",
    operator="Dr. Smith",
    output_path="fda-audit-trail.md",
)
```

---
