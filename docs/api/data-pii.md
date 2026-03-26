# PII Detection

PII (Personally Identifiable Information) in training data is a compliance and security risk. ML models can memorize and later reproduce PII from training data. GDPR, CCPA, and the EU AI Act all require organizations to detect and handle PII in ML pipelines.

**Module:** `mltk.data.pii`

**ML Lifecycle Stage:** Data Collection / Data Preprocessing

**When to use:**
- Before training: scan datasets for PII that shouldn't be there
- Data labeling QA: verify annotators didn't include personal data in labels
- Feature engineering: ensure derived features don't leak PII
- Compliance audits: document PII scanning as part of the ML pipeline

---

## scan_pii

Scan text for PII patterns. Returns a list of matches with type and position.

```python
from mltk.data.pii import scan_pii

matches = scan_pii("Contact john@example.com or call 555-123-4567")
# [PiiMatch(type="email", ...), PiiMatch(type="phone", ...)]
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Text to scan for PII |
| `patterns` | `list[str] \| None` | `None` | Pattern categories to check. None = all patterns |
| `allowlist` | `list[str] \| None` | `None` | Exact strings to suppress. Matches whose text appears in this list are skipped. Useful for known-safe values like internal service addresses or test fixtures. |

### Returns

`list[PiiMatch]` where PiiMatch is a dataclass:

```python
@dataclass
class PiiMatch:
    type: str        # e.g., "email", "phone", "ssn", "api_key"
    start: int       # start position in text
    end: int         # end position in text
    matched_text: str # the matched string
```

### Built-in Patterns (14 patterns in 6 categories)

| Pattern | Category | Examples |
|---------|----------|----------|
| `email` | `email` | `user@domain.com` |
| `phone` | `phone` | `555-123-4567`, `5551234567` |
| `ssn` | `ssn` | `123-45-6789` |
| `credit_card` | `credit_card` | `4111-1111-1111-1111` |
| `api_key_openai_project` | `api_key` | `sk-proj-...` |
| `api_key_openai` | `api_key` | `sk-...` (20+ chars) |
| `api_key_anthropic` | `api_key` | `sk-ant-...` |
| `api_key_stripe` | `api_key` | `pk_...` |
| `api_key_aws` | `api_key` | `AKIA...` (16 uppercase chars) |
| `api_key_groq` | `api_key` | `gsk_...` |
| `api_key_xai` | `api_key` | `xai-...` |
| `api_key_github` | `api_key` | `ghp_...` (36 chars) |
| `api_key_gitlab` | `api_key` | `glpat-...` |
| `password` | `password` | `password=secret123` |

When filtering with `patterns=["api_key"]`, all `api_key_*` sub-patterns are checked. You can also filter by specific sub-pattern names (e.g., `patterns=["api_key_openai"]`).

---

## assert_no_pii

Assert no PII detected in DataFrame text columns.

```python
from mltk.data import assert_no_pii

assert_no_pii(df, columns=["user_notes", "feedback_text"])
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame to scan |
| `columns` | `list[str] \| None` | `None` | Text columns to scan. None = all object/string columns |
| `patterns` | `list[str] \| None` | `None` | Pattern categories to check. None = all |
| `allowlist` | `list[str] \| None` | `None` | Exact strings to suppress across all columns. Passed through to `scan_pii()` for each cell value. |

### Returns

`TestResult` with details:
- `total_matches` -- total PII matches found
- `matches_by_column` -- dict of column name to match count
- `matches_by_type` -- dict of PII type to match count
- `columns_scanned` -- list of columns that were checked

### Why it matters for ML

ML models can **memorize** PII from training data. GPT-style models have been shown to reproduce phone numbers, email addresses, and API keys from their training data when prompted. Beyond privacy, PII in features creates:
- **Legal risk**: GDPR fines up to 4% of global revenue
- **Security risk**: API keys in training data = leaked credentials
- **Model risk**: PII-correlated features create unfair biases

### Example

```python
import pandas as pd
import pytest
from mltk.data import assert_no_pii

@pytest.mark.ml_data
def test_training_data_no_pii():
    """Training data should not contain personal information."""
    df = pd.read_csv("data/customer_feedback.csv")
    assert_no_pii(df, columns=["feedback_text", "notes"])

@pytest.mark.ml_data
def test_labels_no_api_keys():
    """Annotation labels should not contain leaked credentials."""
    df = pd.read_csv("data/annotations.csv")
    assert_no_pii(df, columns=["label_text"], patterns=["api_key_openai", "api_key_aws"])
```

### Edge Cases

- **Only scans string/object columns** when `columns=None`
- **Large DataFrames**: scans all rows — consider sampling for very large datasets
- **Custom patterns**: pass a subset of pattern names to focus scanning
- **False positives**: 10-digit numbers may match phone patterns. Use targeted columns.

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_email_detected` | Finds email addresses in text |
| `test_phone_detected` | Finds US phone numbers |
| `test_ssn_detected` | Finds Social Security Numbers |
| `test_credit_card_detected` | Finds 16-digit card numbers |
| `test_api_key_detected` | Finds OpenAI/AWS/GitHub API keys |
| `test_clean_text_passes` | Text without PII passes |
| `test_subset_columns` | Only scans specified columns |
| `test_custom_patterns` | Only checks specified pattern types |

---
