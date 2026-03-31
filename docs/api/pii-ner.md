# NER-Based PII Detection

Regex catches structured PII -- social security numbers,
credit cards, API keys, IBANs. These patterns have rigid
formats that regular expressions match reliably.

But most personal data is not structured. Names,
organizations, and locations have no fixed format. The
string "Contact John Smith at 123 Main St, Springfield
IL 62701" contains three pieces of personally
identifiable information that regex will never find.

GDPR Article 4(1) defines personal data as "any
information relating to an identified or identifiable
natural person." Names and locations count. Addresses
count. Organization affiliations count. A regex-only
scanner creates a false sense of compliance.

mltk solves this with multi-method PII dispatch: the same
`assert_no_pii` function, the same interface, but with a
`method` parameter that selects the detection engine.

**Module:** `mltk.data.pii`

**ML Lifecycle Stage:** Data validation / Pre-training
audit / CI gate / Compliance check

**Bugs caught:**

- Training data containing names, locations, and
  organizations that regex cannot detect
- GDPR-regulated personal data leaking into model
  training pipelines
- Healthcare records with patient names and medical
  record numbers
- Legal documents with attorney names and case numbers
- Customer-facing outputs that memorize and reproduce
  PII from training data

---

## The Regex Blind Spot

Consider this text from a customer feedback dataset:

```
Contact John Smith at Acme Corp,
123 Main St, Springfield IL 62701
```

| Engine | Catches | Misses |
|--------|---------|--------|
| Regex | *nothing* | Everything |
| NER | John Smith, Acme Corp, 123 Main St Springfield IL 62701 | -- |

Regex has 40+ patterns in mltk -- emails, phones, SSNs,
credit cards, API keys, IBANs, national IDs from 8
countries. But none of those patterns match a person's
name or a company name. The text above passes a
regex-only scan with zero findings.

NER (Named Entity Recognition) models are trained to
identify entities by context. "John Smith" follows a
first-name last-name pattern. "Acme Corp" ends with
a corporate suffix. The street address follows US postal
conventions. NER catches all three.

---

## Methods Overview

| Method | Engine | Best For |
|--------|--------|----------|
| `regex` | mltk built-in (40+) | Structured PII (SSN, CC, keys, IBAN) |
| `ner` | Presidio + spaCy | Names, orgs, locations, phones |
| `gliner` | GLiNER zero-shot | Domain-specific (healthcare, legal) |
| `hybrid` | regex + NER union | Maximum coverage |

**Install:** `regex` needs no extra deps. `ner` and
`hybrid` need `pip install mltk[ner]`. `gliner` needs
`pip install gliner`.

All four methods share the same function signature. The
only difference is the `method` parameter.

---

## Usage

```python
import pandas as pd
from mltk.data import assert_no_pii

df = pd.DataFrame({"text": [
    "Contact John Smith at john@example.com",
    "Patient MRN: 123456, Dr. Jane Doe",
]})

# Regex only -- default, backward compatible
result = assert_no_pii(df, columns=["text"])

# NER -- catches names, orgs, locations
result = assert_no_pii(
    df, columns=["text"], method="ner",
)

# GLiNER -- zero-shot for custom entity types
result = assert_no_pii(
    df,
    columns=["text"],
    method="gliner",
    entity_types=[
        "patient name",
        "medical record number",
    ],
)

# Hybrid -- maximum coverage (regex + NER union)
result = assert_no_pii(
    df, columns=["text"], method="hybrid",
)
```

### Method-Specific Parameters

Each method accepts the shared parameters (`df`,
`columns`, `patterns`, `allowlist`, `severity`) plus
method-specific options:

```python
# NER with custom confidence threshold
result = assert_no_pii(
    df,
    columns=["text"],
    method="ner",
    score_threshold=0.7,     # default 0.5
    entity_types=["PERSON", "LOCATION"],
    language="en",
)

# GLiNER with domain-specific entities
result = assert_no_pii(
    df,
    columns=["text"],
    method="gliner",
    score_threshold=0.7,     # default 0.7
    entity_types=[
        "attorney name",
        "case number",
        "court jurisdiction",
    ],
)

# Hybrid with allowlist
result = assert_no_pii(
    df,
    columns=["text"],
    method="hybrid",
    allowlist=["Springfield"],  # suppress known city
    score_threshold=0.5,
)
```

---

## How Presidio Works

Microsoft Presidio is an open-source framework for PII
detection and anonymization. mltk uses it as the NER
backend because it adds validation and context scoring
on top of raw spaCy NER -- reducing false positives
significantly compared to using spaCy alone.

**Citation:** Microsoft Presidio (2018). MIT License.
3K+ GitHub stars. Originally developed for Azure
Cognitive Services, open-sourced for on-premise use.

### Architecture

```
Text input
    |
    v
AnalyzerEngine
    |
    +---> NlpEngine (spaCy)
    |        |
    |        +---> Tokenization
    |        +---> NER (en_core_web_lg)
    |        +---> Part-of-speech tagging
    |
    +---> RecognizerRegistry
             |
             +---> SpacyRecognizer (PERSON, ORG, LOC)
             +---> PhoneRecognizer (regex + context)
             +---> MedicalLicenseRecognizer
             +---> ... (30+ built-in recognizers)
```

### Why Presidio over raw spaCy

Raw spaCy NER labels every entity it finds. If your text
says "I went to Paris in March," spaCy labels both
"Paris" (GPE) and "March" (DATE). For PII detection, you
want "Paris" but not "March" -- dates alone are not PII.

Presidio solves this with a three-layer validation:

1. **NLP pass.** spaCy provides the initial entity
   candidates with positions and labels.
2. **Recognizer validation.** Each entity type has a
   dedicated recognizer that applies context rules.
   A phone number recognizer checks that digits near the
   word "phone" or "call" score higher than isolated
   digit sequences.
3. **Confidence scoring.** Every match gets a score
   from 0.0 to 1.0. Context words ("Dr." before a name,
   "at" before an address) boost the score. Isolated
   matches without context score lower.

The result: fewer false positives than raw NER, while
still catching contextual PII that regex misses entirely.

### spaCy Model

Presidio uses spaCy for its NLP engine. The default
English model is `en_core_web_lg` (560 MB). On first use,
mltk downloads this model automatically.

The model provides three capabilities Presidio relies on:

- **Tokenization** -- splitting text into words and
  sentences
- **Named entity recognition** -- labeling PERSON, ORG,
  GPE, LOC, DATE, etc.
- **Part-of-speech tagging** -- helping recognizers
  disambiguate context (e.g., "Apple" as a company vs.
  a fruit based on surrounding syntax)

---

## How GLiNER Works

Standard NER models (spaCy, Flair, Stanza) are trained
on fixed entity taxonomies: PERSON, ORG, LOC, DATE.
If your domain has entities outside that taxonomy --
medical record numbers, patent IDs, legal case numbers
-- these models cannot detect them without fine-tuning.

GLiNER eliminates this limitation with zero-shot NER.
You specify entity types at inference time as plain
English strings. The model has never seen "medical record
number" during training, but it understands the concept
well enough to find them in text.

**Citation:** Zaratiana et al., "GLiNER: Generalist
Model for Named Entity Recognition using Bidirectional
Transformer." NAACL 2024. Apache 2.0 License.

### How it works

GLiNER uses a bidirectional transformer architecture
with a novel token-entity matching mechanism:

1. **Input encoding.** Both the text and the entity type
   labels are encoded by the same transformer. The text
   tokens and entity labels share a unified embedding
   space.
2. **Span extraction.** The model identifies candidate
   spans in the text (contiguous token sequences that
   could be entities).
3. **Entity-span matching.** Each candidate span is
   scored against each entity type label. The score
   represents how well the span matches the concept
   described by the label.
4. **Thresholding.** Spans above the confidence
   threshold are returned as detected entities.

Because the entity labels are encoded at inference time
(not hard-coded during training), you can specify *any*
entity type in natural language.

### Default model

mltk uses `urchade/gliner_medium-v2.1`:

| Property | Value |
|----------|-------|
| Parameters | 209M |
| Architecture | Bidirectional Transformer |
| Training data | Pile-NER (50+ entity types) |
| License | Apache 2.0 |
| Zero-shot | Yes -- any entity type string |

### When to use GLiNER

GLiNER is not a replacement for Presidio. It fills a
specific gap: **domain-specific entities that standard
NER models do not cover.**

| Domain | Entity types to pass |
|--------|---------------------|
| Healthcare | `"patient name"`, `"medical record number"`, `"diagnosis"` |
| Legal | `"attorney name"`, `"case number"`, `"court"` |
| Finance | `"account holder"`, `"portfolio ID"`, `"SWIFT code"` |
| HR | `"employee name"`, `"employee ID"`, `"salary"` |

For standard PII (names, orgs, locations, phones),
Presidio is faster and more accurate because its
recognizers are purpose-built with context rules. Use
GLiNER when you need entities Presidio does not cover.

---

## Entity Type Mapping

When `method="ner"`, Presidio returns entity types using
its own taxonomy. mltk maps these to `PiiMatch.type`
values for consistency with regex results.

| Presidio Entity | mltk PiiMatch.type | Example |
|-----------------|-------------------|---------|
| `PERSON` | `person` | John Smith |
| `ORGANIZATION` | `organization` | Acme Corp |
| `LOCATION` | `location` | Springfield IL |
| `PHONE_NUMBER` | `phone` | 555-123-4567 |
| `EMAIL_ADDRESS` | `email` | john@example.com |
| `CREDIT_CARD` | `credit_card` | 4111-1111-1111-1111 |
| `US_SSN` | `ssn` | 123-45-6789 |
| `IBAN_CODE` | `iban` | GB29NWBK60161331926819 |
| `MEDICAL_LICENSE` | `medical_license` | MD-12345 |
| `IP_ADDRESS` | `ipv4` | 192.168.1.1 |
| `DATE_TIME` | `date_time` | January 5, 1990 |
| `NRP` | `nationality` | American, Jewish |
| `US_DRIVER_LICENSE` | `us_driver_license` | D12345678 |

When `method="gliner"`, the entity type is whatever
string you passed in `entity_types`. For example, if you
passed `"medical record number"`, matches will have
`PiiMatch.type = "medical record number"`.

---

## Configuration Reference

### Shared parameters (all methods)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `DataFrame` | *(required)* | Data to scan |
| `columns` | `list[str] \| None` | `None` | Columns to scan. None = all string/object columns |
| `patterns` | `list[str] \| None` | `None` | Regex pattern categories (regex/hybrid only) |
| `allowlist` | `list[str] \| None` | `None` | Exact strings to suppress |
| `severity` | `Severity` | `CRITICAL` | Assertion severity |
| `method` | `str` | `"regex"` | Detection method: `regex`, `ner`, `gliner`, `hybrid` |

### NER-specific parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.5` | Minimum Presidio confidence score (0.0-1.0) |
| `entity_types` | `list[str] \| None` | `None` | Presidio entity types to detect. None = all |
| `language` | `str` | `"en"` | spaCy model language code |

### GLiNER-specific parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.7` | Minimum GLiNER confidence (0.0-1.0) |
| `entity_types` | `list[str]` | *(required)* | Entity type labels (free-form English strings) |

### Why different default thresholds?

Presidio's confidence scores are calibrated with context
rules. A score of 0.5 from Presidio means "moderate
confidence with some context support." GLiNER's scores
come from a raw transformer softmax -- a 0.5 from GLiNER
is less reliable because it lacks Presidio's context
validation layer. The higher default (0.7) compensates
for this.

Adjust thresholds based on your tolerance:

| Goal | `score_threshold` |
|------|-------------------|
| Catch everything (more false positives) | 0.3 |
| Balanced (recommended) | 0.5 (NER) / 0.7 (GLiNER) |
| High precision (fewer false positives) | 0.8+ |

---

## Hybrid Deduplication

When `method="hybrid"`, mltk runs both regex and NER on
every text cell, then merges the results. The same PII
can be found by both engines -- "john@example.com" is
caught by the regex email pattern and by Presidio's
`EMAIL_ADDRESS` recognizer. Returning both would
double-count.

### Deduplication rules

When regex and NER find overlapping spans in the same
text:

1. **Exact overlap.** Both engines found the same
   start and end positions. Keep the NER result for
   entity types where NER has semantic understanding
   (PERSON, ORG, LOCATION). Keep the regex result for
   structured patterns (API keys, SSNs, IBANs) where
   regex has checksum validation.

2. **Partial overlap.** One span contains the other.
   Keep the longer span (it captured more context).

3. **No overlap.** Both results are unique. Keep both.

### Why NER wins for names

Regex cannot validate a person's name -- there is no
pattern for "valid human name." NER assigns a confidence
score based on context: "Dr." before a name boosts the
score, a name appearing after "Dear" boosts the score.
The NER result carries more information.

### Why regex wins for structured patterns

Presidio does not have recognizers for API keys (OpenAI,
Anthropic, Stripe, AWS, GitHub, etc.) or cryptocurrency
addresses. mltk's regex patterns cover 40+ structured
formats with checksum validation where applicable. For
these categories, regex is the only engine that detects
them.

### Practical result

Hybrid mode gives you the union of both engines with
intelligent deduplication. In practice:

- **Names, orgs, locations**: NER-only (regex has no
  patterns for these)
- **Emails, phones**: deduplicated (both engines find
  them, NER result kept)
- **API keys, SSNs, IBANs, crypto**: regex-only (NER
  has no recognizers for these)
- **Medical licenses, driver licenses**: NER-only
  (Presidio has specialized recognizers)

---

## Method Decision Flowchart

```
Is your data structured (SSN, CC, API keys)?
  YES --> method="regex"
          Fast, no extra deps, checksum validation.
  NO
  |
  v
Does it contain names, orgs, addresses?
  YES --> method="ner"
          Presidio + spaCy, context-aware scoring.
  NO
  |
  v
Do you have domain-specific entities?
  (healthcare MRN, legal case numbers, etc.)
  YES --> method="gliner"
          Zero-shot, specify entity types at runtime.
  NO
  |
  v
Want maximum coverage?
  YES --> method="hybrid"
          Regex + NER union, intelligent dedup.
  NO
  |
  v
Default --> method="regex"
  Backward compatible, no extra dependencies.
```

**Rules of thumb:**

- **CI/CD pipelines** where speed matters and PII is
  structured: `regex`. No model downloads, no GPU,
  sub-millisecond per text.
- **Pre-release data audits** where GDPR compliance
  matters: `ner` or `hybrid`. Names and locations are
  personal data under GDPR Article 4(1).
- **Healthcare or legal domains** with non-standard
  entity types: `gliner`. Specify your entity taxonomy
  at runtime without fine-tuning any model.
- **You just want everything caught**: `hybrid`. The
  performance cost of NER is the dominant factor anyway
  -- regex adds negligible overhead on top.

---

## Performance Characteristics

Approximate timings per text cell. Actual numbers depend
on text length, hardware, and model caching.

| Method | Per-text latency | First-call overhead | Dependencies |
|--------|-----------------|---------------------|--------------|
| `regex` | ~1ms | None | None |
| `ner` | ~50-100ms | ~5-10s (spaCy model download) | `presidio-analyzer`, `spacy` |
| `gliner` | ~100-200ms | ~5-10s (transformer download) | `gliner` |
| `hybrid` | ~100-150ms | ~5-10s (spaCy model download) | `presidio-analyzer`, `spacy` |

### Batch scaling

| Texts | `regex` | `ner` | `gliner` | `hybrid` |
|-------|---------|-------|----------|----------|
| 10 | <10ms | ~0.5-1s | ~1-2s | ~1-1.5s |
| 100 | ~100ms | ~5-10s | ~10-20s | ~10-15s |
| 1,000 | ~1s | ~50-100s | ~100-200s | ~100-150s |
| 10,000 | ~10s | ~8-17min | ~17-33min | ~17-25min |

### Model loading

The first call to `method="ner"` or `method="hybrid"`
triggers a spaCy model download (`en_core_web_lg`,
~560 MB). Subsequent calls in the same process reuse
the cached model. Across processes, the model is cached
on disk in the spaCy data directory.

GLiNER downloads `urchade/gliner_medium-v2.1` (~800 MB)
on first use. Cached in `~/.cache/huggingface/`.

### Memory usage

| Method | Approximate RAM |
|--------|----------------|
| `regex` | Negligible |
| `ner` | ~500 MB (spaCy model) |
| `gliner` | ~1 GB (transformer model) |
| `hybrid` | ~500 MB (same as `ner`) |

---

## Installation

### Regex only (default)

```bash
pip install mltk
```

No extra dependencies. All 40+ regex patterns are
built-in.

### NER (Presidio + spaCy)

```bash
pip install mltk[ner]
```

This installs `presidio-analyzer` and `spacy`. The
spaCy language model (`en_core_web_lg`) is downloaded
automatically on first use.

### GLiNER

```bash
pip install gliner
```

GLiNER is a standalone package. The model is downloaded
from HuggingFace Hub on first use.

### Hybrid

```bash
pip install mltk[ner]
```

Hybrid mode uses regex (built-in) + NER (Presidio).
Same install as `ner`.

### What happens without dependencies?

If you call `method="ner"` without `presidio-analyzer`
installed, you get a clear error:

```
ImportError: presidio-analyzer is required for
NER-based PII detection. Install with:
    pip install mltk[ner]
```

Same pattern for GLiNER:

```
ImportError: gliner is required for zero-shot
NER PII detection. Install with:
    pip install gliner
```

---

## Examples

### Training data audit

Scan a dataset before model training to ensure no
personal data leaks into the model.

```python
import pandas as pd
from mltk.data import assert_no_pii

# Load training data
df = pd.read_csv("training_data.csv")

# Quick regex scan (CI gate)
result = assert_no_pii(
    df, columns=["text", "label"],
)

# Deep scan before release (GDPR compliance)
result = assert_no_pii(
    df,
    columns=["text", "label"],
    method="hybrid",
    score_threshold=0.5,
)

if not result.passed:
    print(f"Found {result.details['total_matches']} "
          f"PII matches")
    print(result.details["matches_by_type"])
```

### Healthcare data validation

```python
# Scan for healthcare-specific PII
result = assert_no_pii(
    df,
    columns=["clinical_notes"],
    method="gliner",
    entity_types=[
        "patient name",
        "medical record number",
        "diagnosis",
        "prescription",
        "insurance ID",
    ],
    score_threshold=0.6,
)
```

### Pytest integration

```python
import pytest
import pandas as pd
from mltk.data import assert_no_pii


@pytest.fixture
def training_data():
    return pd.read_csv("data/train.csv")


def test_no_regex_pii(training_data):
    """CI gate: fast regex scan on every commit."""
    result = assert_no_pii(
        training_data, columns=["text"],
    )
    assert result.passed, result.message


def test_no_contextual_pii(training_data):
    """Pre-release: deep NER scan for names/orgs."""
    result = assert_no_pii(
        training_data,
        columns=["text"],
        method="ner",
    )
    assert result.passed, result.message
```

### Allowlist usage

Suppress known-safe values that appear in your data:

```python
result = assert_no_pii(
    df,
    columns=["text"],
    method="hybrid",
    allowlist=[
        "support@example.com",
        "Main Street",  # common placeholder
    ],
)
```

---

## Design Decisions

### Why Presidio instead of raw spaCy?

Raw spaCy NER labels everything. "March" is a DATE,
"Apple" is an ORG, "Jordan" is both a PERSON and a GPE.
Without context validation, false positive rates are
30-50% in general text (measured internally on synthetic
datasets with known PII distributions).

Presidio adds three layers of validation:

1. **Recognizer-specific rules.** Each entity type has
   a dedicated recognizer with domain logic. The phone
   recognizer validates digit patterns. The credit card
   recognizer runs Luhn checksums.
2. **Context scoring.** Words near the entity boost or
   reduce confidence. "Dr." before a name boosts it.
   "The company formerly known as" before a name
   reduces it.
3. **Deny lists and allow lists.** Known non-PII
   strings (company names in your domain, product
   names) can be suppressed at the engine level.

The net effect: Presidio typically achieves 80-90%
precision on general English text, compared to 50-70%
for raw spaCy NER alone (varies by entity type and
domain).

### Why `regex` remains the default method?

Backward compatibility. Every existing user of
`assert_no_pii` gets the same behavior without code
changes. The regex engine has zero external dependencies
and sub-millisecond performance. Users opt into NER
explicitly when they need it.

### Why GLiNER instead of fine-tuning spaCy?

Fine-tuning requires labeled training data, training
infrastructure, and maintenance of a custom model.
GLiNER achieves competitive accuracy on novel entity
types with zero training data -- you just describe the
entity type in English. For teams that need to scan for
domain-specific PII without an ML training pipeline,
GLiNER is the pragmatic choice.

**Citation support:** Zaratiana et al. (NAACL 2024)
demonstrated that GLiNER matches or exceeds fine-tuned
NER models on zero-shot entity types across 8
benchmark datasets. The model generalizes well to
unseen entity descriptions because it learns the
*concept* of entity matching, not a fixed label set.

### Why separate `ner` and `hybrid` methods?

Some teams want *only* NER results (they handle
structured PII with separate tooling or have existing
regex pipelines). Other teams want everything in one
pass. Keeping them separate gives clear performance
expectations:

- `ner`: you pay for NER inference, you get NER results
- `hybrid`: you pay for NER inference + regex, you get
  the union with deduplication

---

## FAQ

### Which method should I start with?

Start with `regex` for your CI pipeline. Add
`method="hybrid"` for pre-release data audits or
GDPR compliance checks. Use `gliner` only when you
have domain-specific entity types that Presidio does
not cover.

### Do I need a GPU?

No. spaCy's `en_core_web_lg` and GLiNER's medium model
both run on CPU. Performance is acceptable for batch
scanning (see Performance Characteristics above). A GPU
speeds up GLiNER by 3-5x but is not required.

### Can I use a non-English spaCy model?

Yes. Pass `language="de"` for German, `language="fr"`
for French, etc. Presidio will use the corresponding
spaCy model (e.g., `de_core_news_lg`). The model is
downloaded on first use.

### What about multi-language text?

Presidio processes one language at a time. If your
dataset contains mixed-language text, run multiple
passes with different `language` values, or use
`method="gliner"` which handles multilingual text
natively (the transformer model supports 100+
languages).

### How does the allowlist interact with NER?

The `allowlist` parameter works the same across all
methods. After detection, any match whose exact text
appears in the allowlist is suppressed. This applies
to both regex matches and NER matches.

### Can I combine GLiNER and Presidio?

Not in a single `assert_no_pii` call. Run two
separate calls:

```python
# Presidio for standard PII
result_ner = assert_no_pii(
    df, columns=["text"], method="ner",
)

# GLiNER for domain-specific PII
result_gliner = assert_no_pii(
    df,
    columns=["text"],
    method="gliner",
    entity_types=["medical record number"],
)
```

### What if Presidio misclassifies an entity?

Use the `allowlist` to suppress known false positives,
or raise `score_threshold` to filter low-confidence
matches. For systematic misclassification, consider
adding a custom Presidio recognizer (see Presidio's
documentation for the `RecognizerRegistry.add_recognizer`
API).
