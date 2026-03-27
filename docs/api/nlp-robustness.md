# Text Noise Robustness

Test NLP model robustness against realistic text corruptions: typos, OCR errors, and keyboard mistakes. These tools generate corrupted variants of clean text and verify that model predictions remain stable.

**Module:** `mltk.domains.nlp.robustness`

**ML Lifecycle Stage:** Post-training evaluation / Pre-deployment gate

---

## Why Test Text Robustness?

Production text is messy. Models trained on clean, well-formatted text encounter a different reality in production:

- **User-generated content** -- typos, autocorrect artifacts, abbreviations, missing punctuation. A customer typing "teh product is gret" means the same thing as "the product is great," but a fragile classifier may disagree.

- **OCR output** -- scanned documents introduce character substitutions ("rn" becomes "m"), missing characters, and inserted artifacts. A medical records system that misclassifies a scanned diagnosis because of OCR noise is a safety issue.

- **Keyboard and touchscreen errors** -- fat fingers on mobile keyboards produce adjacent-key substitutions ("hrllo" instead of "hello"). Voice-to-text systems introduce homophones and phonetic errors.

- **Copy-paste and encoding issues** -- curly quotes, zero-width characters, Unicode normalization differences. Text that looks identical to a human may differ at the byte level.

A sentiment classifier that flips from "positive" to "negative" because of a single typo is a liability. A support ticket router that sends a customer to the wrong department because of a misspelling is a cost. These are not edge cases -- they are everyday inputs.

Robustness testing generates controlled perturbations of clean text and measures whether the model's predictions change. A robust model handles noise gracefully; a fragile model reveals exactly which corruption types it cannot tolerate.

---

## TextPerturber

`TextPerturber` generates realistic text corruptions using four perturbation methods. Each method simulates a different real-world noise source.

```python
class TextPerturber:
    def __init__(self, seed: int | None = None):
        ...

    def char_swap(self, text: str, rate: float = 0.05) -> str: ...
    def char_delete(self, text: str, rate: float = 0.05) -> str: ...
    def char_insert(self, text: str, rate: float = 0.05) -> str: ...
    def keyboard_proximity(self, text: str, rate: float = 0.05) -> str: ...
    def perturb(self, text: str, method: str = "char_swap", rate: float = 0.05) -> str: ...
```

### The Four Perturbation Methods

#### 1. char_swap -- Adjacent Character Transposition

Swaps two neighboring characters. Simulates fast typing where fingers hit keys in the wrong order.

| Real-world source | Example |
|-------------------|---------|
| Fast typing | "the" becomes "teh" |
| Touchscreen errors | "receive" becomes "receiev" |
| Dyslexia-related | "friend" becomes "freind" |

```python
from mltk.domains.nlp.robustness import TextPerturber

p = TextPerturber(seed=42)

# Single swap -- simulates a quick typo
p.char_swap("the product is great", rate=0.05)
# → "teh product is great"

# Higher rate -- simulates very sloppy typing
p.char_swap("the product is great", rate=0.20)
# → "teh prodcut si gerat"
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | The clean text to perturb. |
| `rate` | `float` | `0.05` | Probability of swapping each adjacent character pair (0-1). |

---

#### 2. char_delete -- Character Deletion

Removes individual characters. Simulates touchscreen key misses where the finger does not quite reach the target key.

| Real-world source | Example |
|-------------------|---------|
| Touchscreen misses | "hello" becomes "hllo" |
| SMS/chat shorthand | "tomorrow" becomes "tmrw" (extreme) |
| Network packet loss | truncated text in streaming |

```python
p = TextPerturber(seed=42)

p.char_delete("hello world", rate=0.10)
# → "hllo wrld"

p.char_delete("The quick brown fox", rate=0.05)
# → "The quik brown fx"
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | The clean text to perturb. |
| `rate` | `float` | `0.05` | Probability of deleting each character (0-1). |

---

#### 3. char_insert -- Character Duplication/Insertion

Inserts a duplicate of a character next to itself. Simulates sticky keys, key bounce on mechanical keyboards, and touchscreen double-taps.

| Real-world source | Example |
|-------------------|---------|
| Sticky keys | "hello" becomes "helllo" |
| Key bounce | "press" becomes "preess" |
| Touchscreen double-tap | "good" becomes "goood" |

```python
p = TextPerturber(seed=42)

p.char_insert("hello world", rate=0.10)
# → "helllo worldd"

p.char_insert("This is a test", rate=0.05)
# → "Thiss is a test"
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | The clean text to perturb. |
| `rate` | `float` | `0.05` | Probability of inserting a duplicate after each character (0-1). |

---

#### 4. keyboard_proximity -- Adjacent Key Substitution

Replaces a character with a neighboring key on the QWERTY keyboard layout. Simulates "fat finger" errors where the user presses the key next to the intended one.

| Real-world source | Example |
|-------------------|---------|
| Fat fingers on physical keyboard | "hello" becomes "hrllo" (e→r) |
| Touchscreen imprecision | "good" becomes "giod" (o→i) |
| One-handed mobile typing | "search" becomes "searxh" (c→x) |

```python
p = TextPerturber(seed=42)

p.keyboard_proximity("hello world", rate=0.10)
# → "hrllo wprld"

p.keyboard_proximity("great product", rate=0.05)
# → "grear product"
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | The clean text to perturb. |
| `rate` | `float` | `0.05` | Probability of replacing each character with a QWERTY neighbor (0-1). |

### QWERTY Adjacency Map

The `keyboard_proximity` method uses a standard QWERTY keyboard layout to determine neighboring keys. Each key maps to its physically adjacent keys:

```
  q w e r t y u i o p
   a s d f g h j k l
    z x c v b n m
```

For example:
- `e` is adjacent to `w`, `r`, `s`, `d`
- `f` is adjacent to `d`, `g`, `r`, `t`, `c`, `v`
- `a` is adjacent to `q`, `w`, `s`, `z`

Characters not on the QWERTY layout (digits, punctuation, uppercase, Unicode) are left unchanged. The method lowercases the character for lookup but preserves the original case in the output -- if a replacement occurs on an uppercase letter, the replacement is also uppercased.

---

### The perturb Method

A unified interface that dispatches to any of the four methods by name:

```python
p = TextPerturber(seed=42)

# These are equivalent:
p.perturb("hello", method="char_swap", rate=0.10)
p.char_swap("hello", rate=0.10)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | The clean text to perturb. |
| `method` | `str` | `"char_swap"` | Perturbation method: `"char_swap"`, `"char_delete"`, `"char_insert"`, or `"keyboard_proximity"`. |
| `rate` | `float` | `0.05` | Perturbation rate passed to the underlying method. |

---

### Seeding and Reproducibility

Pass a `seed` to the constructor for reproducible perturbations:

```python
# Reproducible -- same seed, same output
p1 = TextPerturber(seed=42)
p2 = TextPerturber(seed=42)
assert p1.char_swap("hello world") == p2.char_swap("hello world")

# Random -- no seed, different each time
p3 = TextPerturber()
p4 = TextPerturber()
# p3.char_swap("hello world") != p4.char_swap("hello world")  (usually)
```

---

## assert_text_robust

The main assertion: generates N corrupted variants of each input text, runs all variants through the model, and checks that predictions remain stable.

```python
@timed_assertion
def assert_text_robust(
    model_fn: Callable[[str], Any],
    texts: list[str],
    method: str = "char_swap",
    rate: float = 0.05,
    n_perturbations: int = 10,
    min_stability: float = 0.90,
    seed: int | None = None,
) -> TestResult
```

### What it tests

For each input text:

1. Gets the model's prediction on the clean text
2. Generates `n_perturbations` corrupted variants using the specified method
3. Gets the model's prediction on each variant
4. Computes stability = (number of variants with same prediction as clean) / n_perturbations

The assertion passes if the *average stability across all texts* meets the `min_stability` threshold.

```
For each text:
    clean_pred = model_fn(text)
    stable = 0
    for i in range(n_perturbations):
        noisy = perturb(text, method, rate)
        if model_fn(noisy) == clean_pred:
            stable += 1
    per_text_stability = stable / n_perturbations

overall_stability = mean(per_text_stability for all texts)
PASS if overall_stability >= min_stability
```

### Why it matters for ML

A model that is 99% accurate on a clean test set but drops to 70% when inputs contain a single typo is not production-ready. Standard evaluation metrics (accuracy, F1, AUC) measured on clean data do not reveal this vulnerability. Robustness testing explicitly measures the gap between clean-data performance and noisy-data performance.

This is especially important for:

- **Customer-facing NLP** -- chatbots, search, support ticket classification where users type freely
- **Document processing** -- OCR pipelines where every scanned page may have character-level errors
- **Safety-critical text classification** -- content moderation, medical text analysis, legal document review where a misclassification due to a typo has real consequences

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_fn` | `Callable[[str], Any]` | *(required)* | A function that takes a string and returns a prediction. The prediction can be any type -- it is compared with `==`. |
| `texts` | `list[str]` | *(required)* | Clean input texts to test. |
| `method` | `str` | `"char_swap"` | Perturbation method: `"char_swap"`, `"char_delete"`, `"char_insert"`, or `"keyboard_proximity"`. |
| `rate` | `float` | `0.05` | Perturbation rate. Higher values introduce more noise per text. |
| `n_perturbations` | `int` | `10` | Number of corrupted variants to generate per text. Higher values give more reliable stability estimates but increase runtime. |
| `min_stability` | `float` | `0.90` | Minimum required average stability (0-1). |
| `seed` | `int \| None` | `None` | Random seed for reproducible perturbations. |

### Returns

`TestResult` with:

- `name`: `"nlp.text_robust[{method}]"`
- `passed`: `True` if average stability meets threshold
- `severity`: `CRITICAL`
- `details.overall_stability`: average stability across all texts
- `details.min_stability`: the configured threshold
- `details.method`: perturbation method used
- `details.rate`: perturbation rate used
- `details.n_perturbations`: number of variants per text
- `details.num_texts`: total number of texts tested
- `details.per_text_stability`: list of per-text stability scores (one float per input text)
- `details.least_stable_text`: the input text with the lowest stability score
- `details.least_stable_score`: the stability score of the least stable text

### Example

```python
import pytest
from mltk.domains.nlp.robustness import assert_text_robust
from mltk.core.assertion import MltkAssertionError

def mock_sentiment(text: str) -> str:
    """A simple keyword-based sentiment classifier."""
    positive_words = {"great", "good", "excellent", "love", "amazing", "wonderful"}
    negative_words = {"bad", "terrible", "awful", "hate", "horrible", "worst"}
    words = set(text.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"

def test_sentiment_robust_to_typos():
    """Sentiment classifier should handle common typos."""
    texts = [
        "This product is great and wonderful",
        "Terrible experience, absolutely horrible",
        "The service was good overall",
    ]

    result = assert_text_robust(
        model_fn=mock_sentiment,
        texts=texts,
        method="char_swap",
        rate=0.05,
        min_stability=0.80,
        seed=42,
    )
    assert result.passed
    print(f"Overall stability: {result.details['overall_stability']:.2f}")

def test_catches_fragile_classifier():
    """A classifier that depends on exact spelling is fragile."""
    def exact_match_classifier(text: str) -> str:
        if "urgent" in text:
            return "high_priority"
        return "low_priority"

    texts = ["This is urgent please help"]

    # With keyboard noise, "urgent" becomes "urgeny" or "urgwnt"
    # and the classifier falls back to "low_priority"
    with pytest.raises(MltkAssertionError):
        assert_text_robust(
            model_fn=exact_match_classifier,
            texts=texts,
            method="keyboard_proximity",
            rate=0.15,
            min_stability=0.90,
            seed=42,
        )
```

### Edge Cases

- **Empty strings** -- perturbing an empty string returns an empty string. The model receives the same input for clean and noisy, so stability is 1.0.
- **Single-character strings** -- some perturbation methods may not apply. `char_swap` requires at least two characters; on a single-character string, the text is returned unchanged.
- **Model returns non-hashable types** -- predictions are compared with `==`. If the model returns numpy arrays or lists, ensure they support equality comparison. Consider wrapping with `tuple()` or `.tolist()`.
- **Stochastic models** -- if `model_fn` itself is non-deterministic (e.g., uses sampling), instability from the model will be attributed to the perturbation. Set the model's seed internally to isolate perturbation effects.

---

## Real-World Examples

### Example 1: Sentiment Classifier Robustness

Test whether a sentiment analysis model handles typos in product reviews. This is critical for e-commerce platforms where user reviews drive recommendations.

```python
import pytest
from mltk.domains.nlp.robustness import assert_text_robust

@pytest.fixture
def sentiment_model():
    """Load your trained sentiment model."""
    from my_models import load_sentiment_model
    model = load_sentiment_model("models/sentiment_v3.pkl")
    return model.predict

class TestSentimentRobustness:
    """Sentiment model must handle noisy user-generated text."""

    def test_robust_to_typos(self, sentiment_model):
        """Typos should not flip sentiment."""
        reviews = [
            "This product is absolutely wonderful, I love it",
            "Terrible quality, broke after one day of use",
            "Pretty good for the price, would recommend",
            "Not worth the money, very disappointed",
            "Average product, nothing special but does the job",
        ]
        assert_text_robust(
            model_fn=sentiment_model,
            texts=reviews,
            method="char_swap",
            rate=0.05,
            min_stability=0.90,
            seed=42,
        )

    def test_robust_to_mobile_input(self, sentiment_model):
        """Mobile keyboard errors should not flip sentiment."""
        reviews = [
            "This product is absolutely wonderful, I love it",
            "Terrible quality, broke after one day of use",
        ]
        assert_text_robust(
            model_fn=sentiment_model,
            texts=reviews,
            method="keyboard_proximity",
            rate=0.08,
            min_stability=0.85,
            seed=42,
        )

    @pytest.mark.parametrize("method", [
        "char_swap", "char_delete", "char_insert", "keyboard_proximity",
    ])
    def test_robust_across_all_noise_types(self, sentiment_model, method):
        """Model should be stable across all perturbation types."""
        reviews = [
            "Great product, highly recommended",
            "Awful experience, do not buy",
        ]
        assert_text_robust(
            model_fn=sentiment_model,
            texts=reviews,
            method=method,
            rate=0.05,
            min_stability=0.80,
            seed=42,
        )
```

---

### Example 2: Named Entity Recognition with OCR Input

Test whether an NER model handles OCR-style character errors. This is critical for document processing pipelines that scan physical documents and extract entities (names, dates, addresses).

```python
from mltk.domains.nlp.robustness import TextPerturber, assert_text_robust

def test_ner_handles_ocr_noise():
    """NER model should extract entities despite OCR artifacts."""

    def ner_extract(text: str) -> list[str]:
        """Simplified NER that returns entity labels found."""
        from my_models import ner_model
        entities = ner_model.predict(text)
        # Return sorted entity types for consistent comparison
        return tuple(sorted(set(e["label"] for e in entities)))

    # Scanned medical documents with expected entities
    documents = [
        "Patient John Smith was prescribed Aspirin 100mg on 2025-01-15",
        "Dr. Sarah Johnson referred the patient to Memorial Hospital",
        "Lab results from Quest Diagnostics showed elevated glucose levels",
    ]

    # OCR errors often delete characters (smudged ink) or insert artifacts
    assert_text_robust(
        model_fn=ner_extract,
        texts=documents,
        method="char_delete",
        rate=0.03,        # OCR typically has 1-5% character error rate
        min_stability=0.85,
        seed=42,
    )

def test_ner_manual_inspection():
    """Manual inspection of how OCR noise affects entity extraction."""
    p = TextPerturber(seed=42)

    original = "Patient John Smith was prescribed Aspirin on 2025-01-15"

    print("Original:", original)
    for i in range(5):
        p_variant = TextPerturber(seed=i)
        noisy = p_variant.char_delete(original, rate=0.03)
        print(f"  Variant {i}: {noisy}")
    # Inspect: are entity-bearing words ("John Smith", "Aspirin") corrupted?
    # If so, does the model still find them?
```

---

### Example 3: Text Classification for Support Tickets

Test whether a support ticket classifier routes tickets correctly despite customer typos. Misrouted tickets increase response time and frustrate customers.

```python
from mltk.domains.nlp.robustness import assert_text_robust

def test_ticket_routing_stable():
    """Support ticket classifier should not misroute on typos."""

    def classify_ticket(text: str) -> str:
        from my_models import ticket_classifier
        return ticket_classifier.predict(text)

    tickets = [
        "My account is locked and I cannot reset my password",
        "I was charged twice for the same order last week",
        "The app keeps crashing when I try to upload photos",
        "I need to update my shipping address for order 12345",
        "How do I cancel my subscription and get a refund",
    ]

    # Test with character insertion -- common on mobile keyboards
    result = assert_text_robust(
        model_fn=classify_ticket,
        texts=tickets,
        method="char_insert",
        rate=0.05,
        n_perturbations=20,  # More variants for higher confidence
        min_stability=0.85,
        seed=42,
    )

    # Check which tickets are most vulnerable
    for i, (text, score) in enumerate(
        zip(tickets, result.details["per_text_stability"])
    ):
        if score < 0.90:
            print(f"Vulnerable ticket: '{text[:50]}...' (stability: {score:.2f})")

def test_routing_stability_report():
    """Generate a per-method stability report for the classifier."""

    def classify_ticket(text: str) -> str:
        from my_models import ticket_classifier
        return ticket_classifier.predict(text)

    tickets = [
        "My account is locked and I cannot reset my password",
        "I was charged twice for the same order last week",
        "The app keeps crashing when I try to upload photos",
    ]

    methods = ["char_swap", "char_delete", "char_insert", "keyboard_proximity"]

    print("Method               | Stability")
    print("-" * 40)
    for method in methods:
        result = assert_text_robust(
            model_fn=classify_ticket,
            texts=tickets,
            method=method,
            rate=0.05,
            min_stability=0.0,  # Don't fail -- we want to see all scores
            seed=42,
        )
        score = result.details["overall_stability"]
        status = "PASS" if score >= 0.85 else "FAIL"
        print(f"{method:20s} | {score:.2f} [{status}]")
```

---

## Choosing Perturbation Parameters

### Which method to use

| Your use case | Recommended method | Why |
|---------------|--------------------|-----|
| General typo robustness | `char_swap` | Most common typo type across all input methods |
| Mobile/touchscreen input | `keyboard_proximity` | Simulates fat-finger errors on phone keyboards |
| OCR/scanned documents | `char_delete` | OCR most commonly drops characters from smudged or faded text |
| Keyboard bounce / sticky keys | `char_insert` | Simulates repeated keystrokes |
| Comprehensive testing | All four methods | Run parametrized tests across all methods |

### What perturbation rate to use

| Rate | Corruption level | Simulates |
|------|-----------------|-----------|
| 0.01-0.03 | Light | Careful typing, high-quality OCR |
| 0.05-0.08 | Moderate | Normal user-generated text, average OCR |
| 0.10-0.15 | Heavy | Sloppy typing, poor OCR quality, voice-to-text |
| 0.20+ | Extreme | Stress testing only -- text may become unreadable |

### What stability threshold to use

| Threshold | Meaning | When to use |
|-----------|---------|-------------|
| 0.95+ | Very strict | Safety-critical: medical text, legal documents, content moderation |
| 0.85-0.95 | Standard | Production NLP: sentiment, classification, ticket routing |
| 0.70-0.85 | Lenient | Advisory systems where misclassification is correctable |
| < 0.70 | Diagnostic only | Use `min_stability=0.0` to measure without failing, for baseline assessment |

---

## Integration with pytest

```python
import pytest
from mltk.domains.nlp.robustness import assert_text_robust

@pytest.fixture
def classifier():
    from my_models import load_classifier
    return load_classifier("models/latest.pkl").predict

@pytest.fixture
def test_texts():
    return [
        "Please cancel my subscription immediately",
        "I have been charged incorrectly on my last bill",
        "How do I change my email address",
        "The download is not working on my phone",
    ]

@pytest.mark.parametrize("method,rate", [
    ("char_swap", 0.05),
    ("char_delete", 0.03),
    ("char_insert", 0.05),
    ("keyboard_proximity", 0.08),
])
def test_classifier_robustness(classifier, test_texts, method, rate):
    """Classifier should be stable across all noise types."""
    assert_text_robust(
        model_fn=classifier,
        texts=test_texts,
        method=method,
        rate=rate,
        n_perturbations=15,
        min_stability=0.85,
        seed=42,
    )
```
