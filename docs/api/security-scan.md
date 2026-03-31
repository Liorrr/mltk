# security-scan CLI

`mltk security-scan` runs a red team scan against any
model exposed as a Python callable or HTTP endpoint.
It fires attack payloads from the red team catalog,
optionally applies encoding mutations, and exits with
a non-zero code when the model fails the threshold.

Designed as a CI gate: add one line to your pipeline
and get OWASP-mapped security coverage on every build.

**Related:** `docs/api/red-team.md` -- assertion API,
architecture, and attack category descriptions.

---

## Command Syntax

```
mltk security-scan TARGET [OPTIONS]
```

`TARGET` is a Python import path to the callable under
test, using `module.path:callable_name` syntax.

```
mltk security-scan myapp.llm:chat_fn
mltk security-scan src.models.chatbot:ChatBot.respond
mltk security-scan tests.helpers:mock_llm
```

The callable must accept either:
- `str -> str` (single-turn)
- `list[dict] -> str` (multi-turn, messages format)

Single-message callables are auto-wrapped for multi-turn
strategies.

---

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--attacks` | `str` | `owasp-top4` | Preset name or comma-separated category list. Presets: `owasp-top4`, `owasp-top7`. Categories: `prompt_injection`, `system_prompt_extraction`, `data_extraction`, `jailbreak`, `bias_elicitation`, `excessive_agency`, `output_injection`. |
| `--strategy` | `str` | `single` | `single` -- fire each payload once. `multi-turn` -- run stateful conversation chains. `adaptive` -- requires `--llm`. |
| `--threshold` | `float` | `0.9` | Fraction of attacks that must be resisted. `0.9` means at most 10% may pass. |
| `--mutations` | flag | off | Apply all 8 encoding mutations to each payload, expanding the test pool from 143+ to 1,000+ variants. |
| `--mutation-types` | `str` | all | Comma-separated list of specific mutations: `base64`, `rot13`, `leetspeak`, `homoglyphs`, `zero-width`, `pig-latin`, `morse`, `reverse`. |
| `--max-turns` | `int` | `5` | Turn budget per chain when `--strategy multi-turn`. |
| `--llm` | `str` | none | Callable path for LLM-augmented mode (`--strategy adaptive`). Same `module:fn` syntax as `TARGET`. |
| `--format` | `str` | `text` | Output format: `text` (human-readable), `json`, `junit` (JUnit XML for CI integrations). |
| `--output` | `path` | stdout | Write results to a file instead of stdout. |
| `--categories-only` | flag | off | Print available category names and exit. No scan runs. |
| `--verbose` | flag | off | Print each payload and response as they are evaluated. |

---

## Examples

### Basic scan

```bash
# Single-turn OWASP top-4 scan, text output
mltk security-scan myapp.llm:chat_fn
```

### With encoding mutations

```bash
# Expand each payload with all 8 encoding variants
mltk security-scan myapp.llm:chat_fn --mutations
```

### Multi-turn strategy

```bash
mltk security-scan myapp.llm:chat_fn \
  --attacks jailbreak,prompt_injection \
  --strategy multi-turn \
  --max-turns 5
```

### Category filter

```bash
# Only test system prompt leakage and data extraction
mltk security-scan myapp.llm:chat_fn \
  --attacks system_prompt_extraction,data_extraction \
  --threshold 1.0
```

### JSON output for audit reports

```bash
mltk security-scan myapp.llm:chat_fn \
  --attacks owasp-top7 \
  --mutations \
  --threshold 0.9 \
  --format json \
  --output reports/security-scan.json
```

Output structure (`--format json`):

```json
{
  "passed": true,
  "threshold": 0.9,
  "total_attacks": 143,
  "resisted": 132,
  "pass_rate": 0.923,
  "categories": {
    "prompt_injection": {
      "owasp_id": "LLM01",
      "total": 25,
      "resisted": 24,
      "pass_rate": 0.96
    },
    "jailbreak": {
      "owasp_id": "LLM01",
      "total": 20,
      "resisted": 18,
      "pass_rate": 0.90
    }
  },
  "failures": [
    {
      "category": "jailbreak",
      "payload": "...",
      "response_excerpt": "..."
    }
  ]
}
```

### JUnit XML (for GitHub Actions, Jenkins, GitLab CI)

```bash
mltk security-scan myapp.llm:chat_fn \
  --format junit \
  --output test-results/security.xml
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All attack categories passed the threshold. CI gate: green. |
| `1` | One or more categories failed the threshold. CI gate: red. |
| `2` | Configuration error (invalid target path, unknown category name, missing `--llm` for adaptive strategy). |

---

## CI/CD Integration

### GitHub Actions

```yaml
- name: Security scan
  run: |
    mltk security-scan myapp.llm:chat_fn \
      --attacks owasp-top4 \
      --threshold 0.9 \
      --format junit \
      --output test-results/security.xml

- name: Publish scan results
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: security-scan
    path: test-results/security.xml
```

### Full pipeline example

```bash
# Run after model eval, before release gate.
# Fail the build if the model resists fewer than
# 80% of prompt injection attacks.
mltk security-scan myapp.llm:chat_fn \
  --attacks prompt_injection \
  --threshold 0.8
```

### Pre-commit hook (fast local gate)

```bash
# .git/hooks/pre-push
mltk security-scan myapp.llm:chat_fn \
  --attacks prompt_injection,jailbreak \
  --threshold 0.9
```

---

## Assertion API Equivalent

Every `mltk security-scan` invocation corresponds to
a pytest assertion. Use the CLI for quick audits and
one-off scans; use the assertion API for test suites.

```python
# CLI:
# mltk security-scan myapp.llm:chat_fn \
#   --attacks owasp-top4 --threshold 0.9

# Equivalent pytest assertion:
from mltk.domains.llm.red_team import (
    assert_red_team_resilient,
)
from myapp.llm import chat_fn

def test_security_gate():
    assert_red_team_resilient(
        model_fn=chat_fn,
        attacks="owasp-top4",
        threshold=0.9,
    )
```

See `docs/api/red-team.md` for the full assertion
reference including `assert_no_session_jailbreak`,
`assert_owasp_llm_coverage`, and
`assert_encoding_mutation_resilience`.
