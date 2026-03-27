# HIPAA & Custom Compliance

Map mltk test results to HIPAA rules, and build your own compliance framework in YAML.

**Modules:** `mltk.compliance.hipaa`, `mltk.compliance.custom`

---

## Overview: 7 Built-in Compliance Frameworks

mltk ships with seven compliance frameworks. Each maps assertion-name prefixes to regulatory requirements, enabling automated gap analysis and CI coverage gates.

| Framework | Module | Rules/Categories | Use Case |
|-----------|--------|:----------------:|----------|
| EU AI Act | `mltk.compliance.eu_ai_act` | 6 articles | AI regulation in the European Union |
| OWASP ML Top 10 | `mltk.compliance.owasp` | 10 risks | ML security vulnerabilities |
| NIST AI RMF | `mltk.compliance.nist_ai_rmf` | 4 functions | US voluntary AI risk management |
| ISO 42001 | `mltk.compliance.iso_42001` | 6 clauses | AI management system standard |
| FDA 21 CFR Part 11 | `mltk.compliance` | 5 sections | Medical device / pharma audit trail |
| **HIPAA** | `mltk.compliance.hipaa` | 4 rules | Healthcare data privacy and security |
| **Custom** | `mltk.compliance.custom` | User-defined | Your own policies in YAML |

---

## HIPAA Compliance Mapping

The Health Insurance Portability and Accountability Act governs how Protected Health Information (PHI) is used, stored, and transmitted. When ML models are trained on or make predictions about health data, HIPAA compliance requires demonstrable controls around privacy, security, and breach notification.

### Why ML Teams Care About HIPAA

If your model touches patient records, diagnostic images, claims data, or any individually identifiable health information, you are likely a covered entity or business associate under HIPAA. Even de-identified data requires proof that de-identification was done correctly.

HIPAA violations carry penalties of $100 to $50,000 per violation, with a maximum of $1.5 million per year per violation category.

### The Four HIPAA Rules

mltk maps test results to the four major HIPAA rule categories:

#### 1. Privacy Rule (45 CFR 164.500-534)

Protects individually identifiable health information. ML systems must demonstrate that training data either contains no PHI, or that PHI has been properly de-identified.

| mltk Assertion Prefix | What It Proves |
|----------------------|----------------|
| `data.pii` | PII/PHI detection scans found no unprotected identifiers |
| `data.no_pii` | Data confirmed free of personally identifiable information |
| `data.synthetic.dcr_safe` | Synthetic data does not leak real patient records |
| `data.synthetic.novelty` | Generated data is novel, not memorized from training set |

#### 2. Security Rule - Administrative (45 CFR 164.308)

Requires risk analysis, workforce training, and contingency plans. For ML, this translates to bias audits (risk analysis), calibration checks (trust in model outputs), and leakage detection (preventing training data from memorizing PHI).

| mltk Assertion Prefix | What It Proves |
|----------------------|----------------|
| `model.bias` | Bias audit completed (risk analysis) |
| `model.calibration` | Model outputs are well-calibrated (workforce trust) |
| `training.no_target_leakage` | No data leakage from training that could expose PHI |

#### 3. Security Rule - Technical (45 CFR 164.312)

Requires access controls, audit controls, integrity mechanisms, and transmission security. For ML, this maps to SLA monitoring, degradation detection, and latency monitoring.

| mltk Assertion Prefix | What It Proves |
|----------------------|----------------|
| `monitor.sla` | SLA monitoring active (audit trail of model behavior) |
| `monitor.degradation` | Degradation detection active (integrity of predictions) |
| `inference.latency` | System latency within acceptable bounds (operational) |

#### 4. Breach Notification Rule (45 CFR 164.400-414)

Requires notification when unsecured PHI is breached. ML systems must detect when privacy controls fail and when model behavior degrades in ways that could expose sensitive information.

| mltk Assertion Prefix | What It Proves |
|----------------------|----------------|
| `data.pii` | PII leakage detection active |
| `monitor.degradation` | Degradation that could expose PHI is caught |

---

## `assert_hipaa_coverage`

Assert that your test results cover a minimum fraction of HIPAA rules. Use this as a CI gate to prevent models from shipping without adequate HIPAA-relevant test coverage.

**Module:** `mltk.compliance.hipaa`

```python
from mltk.compliance.hipaa import assert_hipaa_coverage

results = [
    {"name": "data.pii.email_scan", "passed": True, "message": "No PHI found"},
    {"name": "model.bias.gender", "passed": True, "message": "No bias"},
    {"name": "monitor.sla.p99", "passed": True, "message": "SLA met"},
    {"name": "monitor.degradation.accuracy", "passed": True, "message": "Stable"},
]

result = assert_hipaa_coverage(results, min_coverage=0.8)
assert result.passed
# coverage = 100% (4/4 rules covered)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results with `name` and `passed` keys |
| `min_coverage` | `float` | `0.8` | Minimum fraction of HIPAA rules that must be covered (0.0-1.0) |

Returns `TestResult` (name: `compliance.hipaa.coverage`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `covered_count` | `int` | Number of HIPAA rules with at least one matching test |
| `total` | `int` | Total HIPAA rules (4) |
| `coverage` | `float` | `covered_count / total` |
| `min_coverage` | `float` | The threshold that was required |
| `gaps` | `list[str]` | Rule IDs with no matching tests |

### Coverage Calculation

Coverage = rules with at least one matching test / total rules (4).

At the default threshold of 0.8 (80%), all 4 rules must be covered because 3/4 = 75% which is below 80%. This is intentional -- HIPAA is a strict regulatory framework.

---

## Helper Functions

### `map_results_to_rules`

Group test results by HIPAA rule for compliance reporting:

```python
from mltk.compliance.hipaa import map_results_to_rules

grouped = map_results_to_rules(results)
# {
#   "privacy_rule": [{"name": "data.pii.email_scan", "passed": True, "rule": "privacy_rule"}],
#   "security_rule_admin": [...],
#   ...
# }
```

Results whose names do not match any HIPAA rule prefix are placed in the `"uncategorised"` bucket.

### `find_gaps`

Identify untested HIPAA rules:

```python
from mltk.compliance.hipaa import find_gaps

gaps = find_gaps(results)
# ["breach_notification"]  -- this rule has no matching tests
```

---

## pytest Integration

```python
import pytest
from mltk.compliance.hipaa import assert_hipaa_coverage

def test_hipaa_compliance_gate():
    """CI gate: all HIPAA rules must have test coverage."""
    results = collect_test_results()  # your results

    result = assert_hipaa_coverage(results, min_coverage=1.0)
    assert result.passed, f"HIPAA gaps: {result.details['gaps']}"
```

---

## Custom Compliance Frameworks

Every organization has internal policies, industry-specific regulations, or client contracts that define ML testing requirements beyond what any single standard covers. Instead of waiting for mltk to add built-in support for your regulation, define it yourself in YAML.

### Why Custom Frameworks

- A hospital network's internal "ML Model Governance Policy v3.2"
- A fintech's SOC 2 Type II controls mapped to model testing
- A defense contractor's CMMC Level 3 requirements for AI systems
- A client SLA that requires specific bias and latency thresholds

### YAML Format Specification

```yaml
# ml-policy.yaml
name: "Internal ML Governance Policy"
version: "3.2"
categories:
  data_quality:
    title: "Data Quality Requirements"
    description: "All training data must pass quality gates before model training"
    assertions:
      - "data.schema"
      - "data.no_nulls"
      - "data.drift"
      - "data.freshness"

  model_validation:
    title: "Model Validation"
    description: "Models must meet accuracy, fairness, and robustness standards"
    assertions:
      - "model.metric"
      - "model.no_regression"
      - "model.bias"
      - "model.adversarial"

  security_controls:
    title: "Security Controls"
    description: "Models must not leak PII and must maintain audit trails"
    assertions:
      - "data.pii"
      - "data.no_pii"
      - "audit.log_complete"

  operational_readiness:
    title: "Operational Readiness"
    description: "Models must meet latency and reliability SLAs"
    assertions:
      - "inference.latency"
      - "inference.throughput"
      - "monitor.sla"
      - "monitor.degradation"
```

### YAML Structure Rules

| Field | Required | Type | Description |
|-------|:--------:|------|-------------|
| `name` | Yes | `str` | Framework name (shown in reports) |
| `version` | No | `str` | Version string (default `"1.0"`) |
| `categories` | Yes | `mapping` | One or more category definitions |
| `categories.*.title` | Yes | `str` | Human-readable category name |
| `categories.*.description` | No | `str` | What this category covers |
| `categories.*.assertions` | Yes | `list[str]` | Assertion-name prefixes matched via `startswith` |

### Why YAML and Not JSON

YAML supports comments, which are essential for compliance documents where auditors annotate *why* a particular assertion maps to a particular policy clause. JSON does not support comments.

---

## `load_custom_framework`

Load and validate a custom framework from YAML:

```python
from mltk.compliance.custom import load_custom_framework

framework = load_custom_framework("ml-policy.yaml")
print(framework["name"])      # "Internal ML Governance Policy"
print(framework["version"])   # "3.2"
print(list(framework["categories"].keys()))
# ["data_quality", "model_validation", "security_controls", "operational_readiness"]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `yaml_path` | `str` | Path to the YAML file (absolute or relative) |

Returns a normalized framework dict. Raises `FileNotFoundError` if the file does not exist, `ValueError` if validation fails.

### Validation

The loader validates the YAML structure and raises `ValueError` if:

- The file cannot be parsed as YAML
- The top-level structure is not a mapping
- The `name` key is missing
- `categories` is missing or not a mapping
- Any category is missing a `title`
- Any category's `assertions` is not a list

---

## `assert_custom_coverage`

CI gate for custom frameworks. Loads the YAML, computes coverage, and returns a pass/fail result.

```python
from mltk.compliance.custom import assert_custom_coverage

results = [
    {"name": "data.schema.columns", "passed": True},
    {"name": "model.metric.accuracy", "passed": True},
    {"name": "data.pii.scan", "passed": True},
    {"name": "inference.latency.p99", "passed": True},
]

result = assert_custom_coverage(
    results,
    framework_yaml="ml-policy.yaml",
    min_coverage=0.75,
)
assert result.passed
# coverage = 100% (4/4 categories covered)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `list[dict]` | *(required)* | Test results with `name` and `passed` keys |
| `framework_yaml` | `str` | *(required)* | Path to the YAML file defining the framework |
| `min_coverage` | `float` | `0.8` | Minimum fraction of categories that must be covered (0.0-1.0) |

Returns `TestResult` (name: `compliance.custom.coverage`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `framework_name` | `str` | Name from the YAML file |
| `covered_count` | `int` | Categories with at least one matching test |
| `total` | `int` | Total categories in the framework |
| `coverage` | `float` | `covered_count / total` |
| `min_coverage` | `float` | The threshold that was required |
| `gaps` | `list[str]` | Category IDs with no matching tests |

---

## Helper Functions

### `map_results_to_custom`

Group test results by custom framework categories:

```python
from mltk.compliance.custom import load_custom_framework, map_results_to_custom

framework = load_custom_framework("ml-policy.yaml")
grouped = map_results_to_custom(results, framework)
# {"data_quality": [...], "model_validation": [...], ...}
```

### `find_custom_gaps`

Identify categories with no matching tests:

```python
from mltk.compliance.custom import load_custom_framework, find_custom_gaps

framework = load_custom_framework("ml-policy.yaml")
gaps = find_custom_gaps(results, framework)
# ["security_controls"]  -- no tests match this category
```

---

## pytest Integration

```python
def test_internal_policy_compliance():
    """CI gate: internal ML policy must be fully covered."""
    results = collect_test_results()

    result = assert_custom_coverage(
        results,
        framework_yaml="compliance/ml-policy.yaml",
        min_coverage=1.0,
    )
    assert result.passed, f"Policy gaps: {result.details['gaps']}"
```

---

## Combining Multiple Frameworks

In regulated industries, you often need to satisfy multiple frameworks simultaneously. Run each coverage check independently:

```python
from mltk.compliance.hipaa import assert_hipaa_coverage
from mltk.compliance.nist_ai_rmf import assert_nist_rmf_coverage
from mltk.compliance.custom import assert_custom_coverage

results = collect_test_results()

# Check all three frameworks
hipaa = assert_hipaa_coverage(results, min_coverage=1.0)
nist = assert_nist_rmf_coverage(results, min_coverage=0.75)
custom = assert_custom_coverage(results, "internal-policy.yaml", min_coverage=1.0)

# All must pass
assert hipaa.passed, f"HIPAA gaps: {hipaa.details['gaps']}"
assert nist.passed, f"NIST gaps: {nist.details['gaps']}"
assert custom.passed, f"Internal gaps: {custom.details['gaps']}"
```
