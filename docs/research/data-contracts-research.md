# Data Contract Support for mltk -- Research & Proposal

**Date:** 2026-03-25
**Status:** Research Complete -- Ready for Implementation Planning
**Author:** Research Agent

---

## Executive Summary

Data contracts are the missing link between "data looks okay" and "data is guaranteed correct for ML." mltk already has the building blocks (schema validation, distribution checks, drift detection, freshness, PII scanning), but users must wire them together manually in Python. A data contract layer would let users define all expectations in a single YAML file and have mltk auto-generate the full test suite. This is the highest-leverage feature mltk can add -- it turns mltk from a testing library into a data quality platform.

---

## 1. What Is a Data Contract?

### Definition

A data contract is a **formal, versioned agreement between a data producer and its consumers** that defines:

| Layer | What a Schema Covers | What a Contract Adds |
|-------|---------------------|---------------------|
| Structure | Column names, data types | Same |
| Semantics | -- | Business meaning, descriptions, classifications |
| Quality | -- | Null rate thresholds, uniqueness, value ranges, outlier bounds |
| Freshness | -- | SLAs on data recency (e.g., "< 24h old") |
| Volume | -- | Expected row counts, min/max bounds |
| Distribution | -- | Statistical properties, drift thresholds |
| Privacy | -- | PII flags, data classification levels |
| Ownership | -- | Producer team, contact, escalation |
| Lineage | -- | Where the data comes from, how it was transformed |
| Versioning | -- | Breaking vs non-breaking changes, compatibility rules |

### Schema vs Contract

A schema says: "column `age` is an integer."
A contract says: "column `age` is an integer, required (0% nulls allowed), unique within `user_id`, must be in range [0, 150], must not drift from training baseline by more than PSI 0.1, is not PII, is owned by the user-data team, and must be refreshed within 24 hours."

**A schema is a subset of a data contract.** The contract wraps schema with quality guarantees, SLAs, ownership, and governance metadata.

### Why Contracts Matter for ML

Models are uniquely sensitive to data changes:
- A column type change from float64 to object silently corrupts features
- A distribution shift in a training feature degrades predictions without any error
- Stale data (concept drift) makes models predict yesterday's world
- PII leaking into training data creates GDPR/CCPA liability
- Missing labels or imbalanced classes bias the model

Data contracts catch all of these at the **point of production**, not after the model has already been retrained on bad data.

---

## 2. Existing Standards

### 2.1 Open Data Contract Standard (ODCS) -- Bitol / Linux Foundation

- **Origin:** PayPal's internal data contract template, now a Linux Foundation AI & Data project
- **Current version:** v3.1.0
- **Format:** YAML
- **License:** Apache 2.0
- **GitHub:** github.com/bitol-io/open-data-contract-standard
- **Key sections:** Fundamentals, Schema, Data Quality (text/SQL/library/custom), Pricing, Stakeholders, Roles, SLA
- **Strength:** Most comprehensive standard, institutional backing
- **Weakness:** Heavy for small teams, enterprise-focused

ODCS v3 supports quality checks as:
- `text` -- human-readable description
- `library` -- predefined attributes (rowCount, unique, freshness)
- `sql` -- custom SQL queries
- `custom` -- vendor-specific (Soda, Great Expectations)

### 2.2 Data Contract Specification (datacontract.com)

- **Current version:** 0.9.3
- **Format:** YAML
- **License:** MIT
- **GitHub:** github.com/datacontract/datacontract-specification
- **Key sections:** models, fields (with type + quality), servicelevels, quality
- **Companion tool:** datacontract-cli (Python) -- lint, test, export to Great Expectations
- **Strength:** Practical, tool-first, supports multiple quality engines (Soda, GX, Monte Carlo)
- **Weakness:** Pre-1.0, still evolving

Example structure:
```yaml
dataContractSpecification: 0.9.3
id: orders-contract
info:
  title: Orders Dataset
  version: 1.0.0
  owner: data-engineering
models:
  orders:
    fields:
      order_id:
        type: string
        required: true
        unique: true
        pii: false
      amount:
        type: decimal
        required: true
        minimum: 0
      created_at:
        type: timestamp
        required: true
servicelevels:
  availability:
    percentage: 99.9%
  freshness:
    threshold: 25h
    timestampField: orders.created_at
  frequency:
    type: batch
    cron: "0 8 * * *"
quality:
  type: SodaCL
  specification:
    checks for orders:
      - row_count > 1000
      - duplicate_percent(order_id) < 1%
```

### 2.3 Soda Data Contracts

- **Format:** YAML (SodaCL)
- **GitHub:** github.com/sodadata/soda-core
- **Approach:** Contract = dataset YAML + checks
- **Key sections:** dataset, columns (with schema check), checks (row count, freshness, duplicates, custom SQL)
- **Strength:** Battle-tested quality engine, cloud platform for monitoring
- **Weakness:** Coupled to Soda ecosystem

### 2.4 dbt Model Contracts

- **Format:** YAML (within .yml model files)
- **Approach:** Contracts as model config -- enforces columns, types, constraints at build time
- **Strength:** Native to dbt, enforced during `dbt build`
- **Weakness:** Only covers schema + constraints, no distribution/drift/PII. dbt-only.

### 2.5 DataHub Data Contracts

- **Format:** YAML (DataHub metadata model)
- **Approach:** Combines schema + freshness + volume contracts with assertion framework
- **Integrations:** Great Expectations, dbt, custom assertions
- **Strength:** Integrates with metadata catalog
- **Weakness:** Requires DataHub platform

### Comparison Matrix

| Feature | ODCS (Bitol) | datacontract.com | Soda | dbt | **mltk (proposed)** |
|---------|:---:|:---:|:---:|:---:|:---:|
| Schema validation | Yes | Yes | Yes | Yes | **Yes** |
| Data types | Yes | Yes | Yes | Yes | **Yes** |
| Null checks | Yes | Yes | Yes | Yes | **Yes** |
| Uniqueness | Yes | Yes | Yes | Yes | **Yes** |
| Value ranges | Yes | Yes | Yes | -- | **Yes** |
| Freshness SLA | Yes | Yes | Yes | -- | **Yes** |
| Row count bounds | Yes | Yes | Yes | -- | **Yes** |
| Distribution drift | -- | -- | Partial | -- | **Yes (4 methods)** |
| PII detection | -- | Partial (flag) | -- | -- | **Yes (11 patterns)** |
| Label quality | -- | -- | -- | -- | **Yes** |
| Model metrics | -- | -- | -- | -- | **Yes** |
| Inference testing | -- | -- | -- | -- | **Yes** |
| Bias/fairness | -- | -- | -- | -- | **Yes** |
| pytest native | -- | -- | -- | -- | **Yes** |
| Rust acceleration | -- | -- | -- | -- | **Yes** |

**mltk's differentiator: the only data contract system that covers the full ML lifecycle -- data, model, inference, and fairness -- all in one YAML file, executable as pytest tests.**

---

## 3. Industry Adoption

### Who Uses Data Contracts

| Company | Approach | Scale |
|---------|----------|-------|
| **PayPal** | Created the original data contract template (now ODCS). Uses for data mesh at enterprise scale. | 400M+ accounts |
| **Spotify** | Backstage catalog with YAML descriptors for services and data products. Workflow plugin for pipelines with alerts. | 600M+ users |
| **Netflix** | DataOps with contract-driven validation. Migrated away from monolithic data warehouse to domain-owned products. | 260M+ subscribers |
| **Airbnb** | Version-controlled data pipelines with automated testing frameworks. Minerva metrics layer with semantic definitions. | 150M+ users |
| **Uber** | Databook catalog with schema/quality contracts. uDeploy for data pipeline CI/CD. | 130M+ users |
| **DoorDash** | Data quality contracts with automated monitoring and alerting. | 37M+ consumers |
| **Shopify** | Contract-driven data products with ownership model. | 2M+ merchants |

### Adoption Trends (2025-2026)

- Companies adopting data contracts report **50-90% fewer data pipeline failures** within months
- Data contracts are now a core component of the **data mesh architecture** pattern
- Regulatory pressure (GDPR, CCPA, EU AI Act) is accelerating adoption -- contracts provide auditable proof of data governance
- The trend is moving from "optional best practice" to "non-negotiable infrastructure"

---

## 4. Data Mesh Architecture -- How Contracts Fit

### Producer-Consumer Model

```
+------------------+       data contract        +------------------+
|  DATA PRODUCER   | =========================> |  DATA CONSUMER   |
|  (domain team)   |    schema + quality +       |  (ML team)       |
|                  |    freshness + SLA           |                  |
+------------------+                             +------------------+
        |                                                |
        v                                                v
   Publishes data                                  Validates data
   with guarantees                                 against contract
        |                                                |
        v                                                v
   Contract is the                               mltk auto-generates
   API for data                                  pytest tests from
                                                 contract YAML
```

### Key Principles

1. **Domain ownership:** The team that produces the data owns the contract
2. **Contract as API:** Consumers depend on the contract, not the implementation
3. **Shift left:** Validate at the point of production, not after consumption
4. **Version control:** Contracts are YAML files checked into git -- diffable, reviewable, auditable
5. **Automated enforcement:** CI/CD runs contract tests on every data pipeline change

### Where mltk Fits

mltk sits on the **consumer side** of the contract. The ML team defines (or receives) a data contract, and mltk:
1. Validates incoming data against the contract
2. Runs distribution/drift checks beyond what schema alone catches
3. Scans for PII violations
4. Checks freshness SLAs
5. Reports results through pytest with rich failure messages

mltk can also sit on the **producer side** during CI/CD -- producers run `mltk contract validate` before publishing data to catch violations before they reach consumers.

---

## 5. Proposed YAML Format for mltk Data Contracts

### Design Principles

1. **One file, full coverage.** Schema + quality + drift + PII + freshness + lineage in a single YAML
2. **ML-first.** Include label quality, distribution properties, and drift baselines -- things no other contract standard covers
3. **Executable.** Every field in the contract maps directly to an `mltk.data.assert_*` call
4. **Compatible.** Support import/export to ODCS and datacontract.com formats
5. **Minimal by default.** Only `columns` is required. Everything else is opt-in.

### Full Specification

```yaml
# mltk Data Contract v1
# File: contracts/training_data.contract.yaml

contract: mltk/v1
id: training-data-users
version: 1.2.0

info:
  title: User Training Dataset
  description: Cleaned user data for churn prediction model
  owner: ml-team@company.com
  domain: user-analytics
  tags: [ml, training, churn]

# ============================================================
# SCHEMA: columns, types, constraints
# ============================================================
columns:
  user_id:
    type: int64
    required: true          # assert_no_nulls
    unique: true            # assert_unique
    pii: false
    description: Unique user identifier

  email:
    type: object
    required: true
    pii: true               # assert_no_pii will flag if pii: false
    pii_action: hash        # guidance: should be hashed before training
    description: User email address

  age:
    type: int64
    required: true
    range: [0, 150]         # assert_range(min_val=0, max_val=150)
    no_outliers: true       # assert_no_outliers(method="iqr")
    outlier_threshold: 3.0  # IQR multiplier
    description: User age in years

  monthly_spend:
    type: float64
    required: true
    range: [0, null]        # min=0, no max
    distribution:
      type: log-normal      # informational, for documentation
      drift_method: psi     # assert_no_drift(method="psi")
      drift_threshold: 0.1  # PSI < 0.1 = stable
    description: Average monthly spend in USD

  signup_date:
    type: datetime64[ns]
    required: true
    description: Account creation timestamp

  plan:
    type: object
    required: true
    allowed_values: [free, pro, enterprise]   # assert allowed values
    description: Subscription plan

  churned:
    type: int64
    required: true
    is_label: true                   # marks this as the ML label column
    allowed_values: [0, 1]
    label_balance:
      max_ratio: 5.0                # assert_label_balance(max_ratio=5.0)
    label_coverage:
      expected: ["0", "1"]          # assert_label_coverage
      min_samples: 100
    description: Binary churn label (1 = churned)

# ============================================================
# QUALITY: table-level quality rules
# ============================================================
quality:
  row_count:
    min: 10000                       # assert_row_count(min_rows=10000)
    max: 10000000                    # assert_row_count(max_rows=10000000)

  freshness:
    column: signup_date              # assert_freshness(date_column=...)
    max_age_days: 30                 # assert_freshness(max_age_days=30)

  no_duplicate_rows: true            # full-row dedup check

  null_threshold: 0.02              # max 2% nulls across all non-required cols

  custom:
    - name: spend_age_correlation
      description: Monthly spend should correlate with age > 0.1
      severity: warning

# ============================================================
# PII: privacy rules
# ============================================================
pii:
  scan_columns: [email]             # columns to actively scan
  blocked_patterns: [ssn, credit_card, api_key]  # must NOT appear
  allowed_patterns: [email]         # acceptable if pii:true on column

# ============================================================
# DRIFT: baseline comparison rules
# ============================================================
drift:
  baseline: baselines/users_v1.2.parquet  # reference distribution
  method: psi                        # default method for all columns
  threshold: 0.1                     # default threshold
  columns:                           # per-column overrides
    age:
      method: ks
      threshold: 0.05
    monthly_spend:
      method: psi
      threshold: 0.15               # more lenient for spend

# ============================================================
# LINEAGE: where this data comes from
# ============================================================
lineage:
  sources:
    - name: raw_users
      system: postgres
      table: public.users
    - name: payment_events
      system: kafka
      topic: payments.completed
  transformations:
    - join raw_users with payment_events on user_id
    - aggregate monthly_spend from payment_events
    - hash email column

# ============================================================
# SLA: service-level agreements
# ============================================================
sla:
  availability: 99.5%
  update_frequency: daily
  max_delivery_delay: 6h
  support: ml-platform@company.com

# ============================================================
# VERSIONING: contract change management
# ============================================================
versioning:
  breaking_changes:
    - Removing a column
    - Changing a column type
    - Tightening a range constraint
    - Adding a required column
  non_breaking_changes:
    - Adding an optional column
    - Loosening a range constraint
    - Updating descriptions
```

---

## 6. How mltk Would Use Data Contracts

### 6.1 Auto-Generated Test Suite

The core value proposition: **one YAML file generates a complete pytest test module.**

```python
# Auto-generated by: mltk contract generate contracts/training_data.contract.yaml

import pandas as pd
import pytest
from mltk.data import (
    assert_schema,
    assert_no_nulls,
    assert_freshness,
    assert_no_drift,
    assert_no_outliers,
    assert_no_pii,
    assert_range,
    assert_row_count,
    assert_unique,
    assert_label_balance,
    assert_label_coverage,
)


CONTRACT_SCHEMA = {
    "user_id": "int64",
    "email": "object",
    "age": "int64",
    "monthly_spend": "float64",
    "signup_date": "datetime64[ns]",
    "plan": "object",
    "churned": "int64",
}


@pytest.fixture
def df():
    """Load the dataset under test."""
    return pd.read_parquet("data/users_training.parquet")


@pytest.mark.ml_data
class TestTrainingDataContract:
    """Auto-generated from contracts/training_data.contract.yaml v1.2.0"""

    def test_schema(self, df):
        assert_schema(df, CONTRACT_SCHEMA, allow_extra_columns=False)

    def test_required_columns_no_nulls(self, df):
        assert_no_nulls(df, columns=["user_id", "email", "age",
                                      "monthly_spend", "signup_date",
                                      "plan", "churned"])

    def test_user_id_unique(self, df):
        assert_unique(df, columns=["user_id"])

    def test_age_range(self, df):
        assert_range(df["age"], min_val=0, max_val=150)

    def test_age_no_outliers(self, df):
        assert_no_outliers(df["age"], method="iqr", threshold=3.0)

    def test_monthly_spend_range(self, df):
        assert_range(df["monthly_spend"], min_val=0, max_val=float("inf"))

    def test_plan_allowed_values(self, df):
        invalid = df[~df["plan"].isin(["free", "pro", "enterprise"])]
        assert len(invalid) == 0, f"{len(invalid)} rows with invalid plan values"

    def test_churned_allowed_values(self, df):
        invalid = df[~df["churned"].isin([0, 1])]
        assert len(invalid) == 0, f"{len(invalid)} rows with invalid churned values"

    def test_label_balance(self, df):
        assert_label_balance(df["churned"], max_ratio=5.0)

    def test_label_coverage(self, df):
        assert_label_coverage(df["churned"], expected_labels={"0", "1"},
                              min_samples=100)

    def test_row_count(self, df):
        assert_row_count(df, min_rows=10000, max_rows=10000000)

    def test_freshness(self, df):
        assert_freshness(df, date_column="signup_date", max_age_days=30)

    def test_no_pii_leakage(self, df):
        # Scan non-PII columns for accidental PII
        non_pii_cols = [c for c in df.columns
                        if c not in ["email"]]
        assert_no_pii(df, columns=[c for c in non_pii_cols
                                    if df[c].dtype == "object"])

    def test_drift_age(self, df):
        baseline = pd.read_parquet("baselines/users_v1.2.parquet")
        assert_no_drift(baseline["age"], df["age"], method="ks",
                        threshold=0.05)

    def test_drift_monthly_spend(self, df):
        baseline = pd.read_parquet("baselines/users_v1.2.parquet")
        assert_no_drift(baseline["monthly_spend"], df["monthly_spend"],
                        method="psi", threshold=0.15)
```

### 6.2 CLI Commands

```bash
# Initialize a contract from an existing dataset (infer schema + stats)
mltk contract init data/users.parquet -o contracts/users.contract.yaml

# Validate a dataset against a contract (no code needed)
mltk contract validate contracts/users.contract.yaml data/users.parquet

# Generate pytest test file from contract
mltk contract generate contracts/users.contract.yaml -o tests/test_users_contract.py

# Check contract for breaking changes between versions
mltk contract diff contracts/users.contract.yaml contracts/users.contract.yaml.bak

# Lint a contract YAML for correctness
mltk contract lint contracts/users.contract.yaml

# Export to ODCS or datacontract.com format
mltk contract export contracts/users.contract.yaml --format odcs
mltk contract export contracts/users.contract.yaml --format datacontract
```

### 6.3 Programmatic API

```python
from mltk.contract import DataContract

# Load contract
contract = DataContract.from_yaml("contracts/users.contract.yaml")

# Validate a DataFrame
results = contract.validate(df)
# Returns: list[TestResult] -- one per check

# Check if all passed
assert contract.validate(df).all_passed()

# Get failures only
failures = [r for r in contract.validate(df) if not r.passed]

# Infer contract from data (bootstrap)
contract = DataContract.from_dataframe(df, infer_ranges=True, infer_pii=True)
contract.to_yaml("contracts/inferred.contract.yaml")

# Diff two contracts
from mltk.contract import diff_contracts
changes = diff_contracts(old_contract, new_contract)
# Returns: list of Breaking/NonBreaking changes
```

### 6.4 pytest Plugin Integration

```python
# In conftest.py -- register contract as a fixture
import pytest
from mltk.contract import DataContract

@pytest.fixture
def user_contract():
    return DataContract.from_yaml("contracts/users.contract.yaml")

# In test file -- use contract directly
@pytest.mark.ml_data
def test_user_data(user_contract, user_df):
    results = user_contract.validate(user_df)
    assert results.all_passed(), results.summary()
```

Or with zero Python via the pytest plugin:

```ini
# pyproject.toml
[tool.mltk]
contracts = ["contracts/*.contract.yaml"]
contract_data_dir = "data/"
```

```bash
# pytest auto-discovers contracts and generates tests
pytest -m ml_contract --mltk-report
```

---

## 7. Implementation Plan

### Phase 1: Core Contract Engine (1 sprint)

New files:
- `src/mltk/contract/__init__.py` -- public API
- `src/mltk/contract/schema.py` -- YAML parsing, contract dataclasses
- `src/mltk/contract/validator.py` -- validate DataFrame against contract
- `src/mltk/contract/generator.py` -- auto-generate pytest files from contract
- `src/mltk/contract/inferrer.py` -- infer contract from a DataFrame

Key classes:
```python
@dataclass
class ColumnContract:
    name: str
    type: str
    required: bool = True
    unique: bool = False
    pii: bool = False
    range: tuple[float | None, float | None] | None = None
    allowed_values: list[Any] | None = None
    no_outliers: bool = False
    outlier_threshold: float = 1.5
    distribution: DistributionContract | None = None
    is_label: bool = False
    label_balance: LabelBalanceContract | None = None
    label_coverage: LabelCoverageContract | None = None
    description: str = ""

@dataclass
class DataContract:
    id: str
    version: str
    columns: dict[str, ColumnContract]
    quality: QualityContract | None = None
    pii: PiiContract | None = None
    drift: DriftContract | None = None
    sla: SlaContract | None = None
    lineage: LineageContract | None = None
```

### Phase 2: CLI + pytest Integration (1 sprint)

- `mltk contract init` -- infer contract from CSV/Parquet
- `mltk contract validate` -- headless validation
- `mltk contract generate` -- emit test file
- `mltk contract lint` -- check YAML correctness
- `mltk contract diff` -- detect breaking changes
- pytest plugin: auto-discover `*.contract.yaml` files, generate parametrized tests

### Phase 3: Interoperability (1 sprint)

- Import from ODCS v3.1 YAML
- Import from datacontract.com v0.9 YAML
- Export to both formats
- Import from dbt model .yml contracts
- JSON Schema for mltk contract format (for editor autocomplete)

### Phase 4: Advanced Features (future)

- Contract versioning with git integration (auto-detect breaking changes in CI)
- Contract registry (share contracts across teams)
- Contract monitoring (continuous validation in production, not just CI)
- Rust acceleration for contract validation (batch all checks in one pass)

---

## 8. Why This Is mltk's Killer Feature

### The Gap in the Market

| Tool | Tests Data? | Tests Models? | Tests Inference? | From YAML? | pytest Native? |
|------|:-----------:|:-------------:|:----------------:|:----------:|:--------------:|
| Great Expectations | Yes | -- | -- | JSON | -- |
| Soda | Yes | -- | -- | YAML | -- |
| dbt contracts | Schema only | -- | -- | YAML | -- |
| Evidently | Yes | Partial | -- | Python | -- |
| Deepchecks | Yes | Yes | -- | Python | -- |
| **mltk contracts** | **Yes** | **Yes** | **Yes** | **YAML** | **Yes** |

**No existing tool covers the full ML lifecycle from a single YAML contract.** Every competitor either:
- Only covers data quality (Great Expectations, Soda, dbt)
- Requires Python code, not YAML (Deepchecks, Evidently)
- Does not integrate with pytest
- Does not include ML-specific checks (drift with 4 methods, PII with 11 patterns, label quality, bias/fairness)

### The Pitch

> "Define your ML data expectations in YAML. mltk auto-generates pytest tests that validate schema, quality, drift, PII, and label integrity. One file. Zero boilerplate. Full lifecycle coverage."

### Competitive Advantages

1. **ML-native:** Label quality, distribution drift, PII detection -- these are not afterthoughts, they are first-class contract fields
2. **pytest-native:** No new test runner to learn. Contracts become pytest tests. Works with existing CI/CD.
3. **Rust-accelerated:** Drift detection at 10-100x speed for large datasets
4. **Zero-to-tests in one command:** `mltk contract init data.parquet` infers the contract, `mltk contract generate` emits the test file
5. **Interoperable:** Import/export ODCS and datacontract.com formats -- not a walled garden
6. **Progressive:** Start with just `columns:` (schema validation). Add quality, drift, PII rules as needed. Never all-or-nothing.

### Target Users

1. **ML Engineers** building training pipelines -- catch data issues before they corrupt models
2. **Data Engineers** publishing data products -- provide guarantees to downstream consumers
3. **MLOps teams** monitoring production -- continuous contract validation
4. **Regulated industries** (finance, healthcare) -- auditable proof of data quality governance

---

## 9. Open Questions

1. **File extension:** `.contract.yaml` vs `.mltk.yaml` vs `.datacontract.yaml`?
   - Recommendation: `.contract.yaml` -- clear, not tied to any vendor

2. **Compatibility target:** Should mltk contracts be a superset of datacontract.com spec, or a separate format that imports/exports?
   - Recommendation: Separate format (ML-focused), with import/export bridges

3. **Runtime validation:** Should contracts also work at inference time (validate each request against column contracts)?
   - Recommendation: Yes, via `contract.validate_row(dict)` for single-record checks

4. **Contract inheritance:** Should contracts support `extends: base_contract.yaml` for shared rules?
   - Recommendation: Yes, in Phase 3

5. **Baseline management:** How to handle drift baselines -- store in git, or separate artifact store?
   - Recommendation: Git for small baselines (< 10MB), configurable path for external stores

---

## 10. Sources

- [What Is a Data Contract? -- Tacnode (2026)](https://tacnode.io/post/what-is-a-data-contract)
- [How Data Contracts Guarantee Pipeline Reliability -- Acceldata](https://www.acceldata.io/blog/how-data-contracts-guarantee-pipeline-reliability-data-quality-slas)
- [Data Contracts for ML: Schema Evolution & Governance -- DataScienceVerse](https://www.datascienceverse.com/data-contracts-for-ml-automated-schema-evolution-validation-and-governance-best-practices/)
- [Data Contracts for Reliable Pipelines -- Conduktor](https://www.conduktor.io/glossary/data-contracts-for-reliable-pipelines)
- [Open Data Contract Standard (ODCS) v3.1.0 -- Bitol](https://bitol-io.github.io/open-data-contract-standard/v3.1.0/)
- [ODCS GitHub Repository -- bitol-io](https://github.com/bitol-io/open-data-contract-standard)
- [PayPal Data Contract Template -- GitHub](https://github.com/paypal/data-contract-template)
- [Data Contract Specification v0.9 -- datacontract.com](https://datacontract-specification.com/)
- [Data Contract CLI -- datacontract](https://cli.datacontract.com/)
- [datacontract-cli GitHub Repository](https://github.com/datacontract/datacontract-cli)
- [Soda Guide to Data Contracts](https://soda.io/blog/guide-to-data-contracts)
- [Soda Contract Language Reference (v4)](https://docs.soda.io/soda-v4/reference/contract-language-reference)
- [dbt Model Contracts -- dbt Developer Hub](https://docs.getdbt.com/docs/mesh/govern/model-contracts)
- [Data Mesh Architecture -- datamesh-architecture.com](https://www.datamesh-architecture.com/)
- [From Monolith to Contract-Driven Data Mesh -- Towards Data Science](https://towardsdatascience.com/from-monolith-to-contract-driven-data-mesh/)
- [Data Contracts Explained -- Atlan (2026)](https://atlan.com/data-contracts/)
- [Data Contracts: How They Work -- Monte Carlo](https://www.montecarlodata.com/blog-data-contracts-explained/)
- [Data Contracts -- PW Skills](https://pwskills.com/blog/data-contracts/)
- [DataHub Data Contract](https://docs.datahub.com/docs/generated/metamodel/entities/datacontract)
- [Great Expectations Schema Validation](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/schema/)
- [Data Contract Wikipedia](https://en.wikipedia.org/wiki/Data_contract)
- [MLOps and Data Quality -- Provectus](https://provectus.com/blog/data-quality-mlops-ml-production/)
