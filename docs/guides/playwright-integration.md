# Playwright Integration

Test the UI **and** the ML quality behind it in one pipeline. Playwright validates what the user sees (clicks, forms, navigation). mltk validates what the model produces (relevancy, coherence, toxicity, latency). Together they give you true end-to-end coverage for ML-powered applications.

---

## Why combine them?

| Layer | Tool | What it catches |
|-------|------|-----------------|
| **UI behavior** | Playwright | Broken forms, missing elements, navigation regressions |
| **ML quality** | mltk | Bad predictions, toxic outputs, slow inference, hallucinations |
| **Integration** | Both | Model output renders correctly, latency meets SLA under real DOM |

Without mltk, Playwright tells you "the search results appeared." With mltk, you also know "the search results are relevant." Without Playwright, mltk tells you "the model scored 0.85." With Playwright, you also know "the score renders correctly in the dashboard."

---

## Setup

```bash
pip install mltk[all] playwright
playwright install chromium
```

Verify both are working:

```bash
python -c "import mltk; print(mltk.__version__)"
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

---

## Testing ML-Powered UI Features

### Search results quality

The search bar works (Playwright) and the results are relevant to the query (mltk).

```python
import pytest
from playwright.sync_api import sync_playwright
from mltk.domains.llm import assert_answer_relevancy


@pytest.mark.ml_model
def test_search_relevancy():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000")

        query = "machine learning frameworks"

        # UI: type search query and submit
        page.fill("#search-input", query)
        page.click("#search-button")
        page.wait_for_selector(".results")

        # Extract ML output from the rendered page
        results = page.query_selector_all(".result-item")
        result_texts = [r.inner_text() for r in results]

        # Basic UI assertion: results appeared
        assert len(result_texts) > 0, "No search results rendered"

        # mltk: each result is relevant to the query
        for text in result_texts:
            assert_answer_relevancy(
                question=query,
                answer=text,
                min_score=0.3,
            )

        browser.close()
```

### Recommendation system

Recommendations load in the UI (Playwright) and they are diverse and de-duplicated (mltk + plain assertions).

```python
@pytest.mark.ml_model
def test_recommendation_quality():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/user/123")

        # UI: navigate to recommendations
        page.click("#recommendations-tab")
        page.wait_for_selector(".recommendation-card")

        # Extract recommendations from the DOM
        cards = page.query_selector_all(".recommendation-card")
        rec_ids = [c.get_attribute("data-item-id") for c in cards]

        # No duplicate recommendations
        assert len(rec_ids) == len(set(rec_ids)), "Duplicate recommendations found"

        # Recommendations are diverse (at least 3 categories)
        categories = [c.get_attribute("data-category") for c in cards]
        unique_categories = len(set(categories))
        assert unique_categories >= 3, f"Only {unique_categories} categories — not diverse enough"

        browser.close()
```

### Chatbot / LLM response quality

The chat UI sends and receives messages (Playwright). The bot response is non-toxic, coherent, and grounded in context (mltk).

```python
from mltk.domains.llm import (
    assert_no_toxicity,
    assert_coherence,
    assert_faithfulness,
)


@pytest.mark.ml_model
def test_chatbot_response_quality():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/chat")

        # UI: send a message
        page.fill("#chat-input", "What are your return policies?")
        page.click("#send-button")
        page.wait_for_selector(".bot-response")

        response = page.inner_text(".bot-response")

        # Ground truth context (from your knowledge base)
        context = (
            "Our return policy allows returns within 30 days of purchase. "
            "Items must be unused and in original packaging. "
            "Refunds are processed within 5 business days."
        )

        # mltk: response is safe
        assert_no_toxicity([response])

        # mltk: response is internally coherent
        assert_coherence(response, min_score=0.3)

        # mltk: response is grounded in the knowledge base
        assert_faithfulness(response, context, min_score=0.5)

        browser.close()
```

### Image classification UI

The upload flow works (Playwright). The prediction is confident and fast (mltk).

```python
from mltk.inference import assert_latency


@pytest.mark.ml_model
def test_image_upload_prediction():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/classify")

        # UI: upload an image
        page.set_input_files("#file-upload", "test_data/cat.jpg")
        page.click("#classify-button")
        page.wait_for_selector("#prediction-result")

        prediction = page.inner_text("#prediction-result")
        confidence = float(page.inner_text("#confidence-score"))

        # ML quality: prediction is valid and confident
        assert confidence > 0.8, f"Low confidence: {confidence}"
        assert prediction in ["cat", "dog", "bird"], f"Unexpected prediction: {prediction}"

        browser.close()


@pytest.mark.ml_inference
def test_classification_latency():
    """Verify the ML endpoint responds within SLA."""
    import requests

    assert_latency(
        lambda: requests.post(
            "http://localhost:3000/api/classify",
            files={"image": open("test_data/cat.jpg", "rb")},
        ),
        p95=2000.0,  # 2 seconds max at P95
    )
```

### RAG pipeline end-to-end

The user asks a question in the UI (Playwright). The answer is faithful to the retrieved context, and the context is relevant to the question (mltk RAG assertions).

```python
from mltk.domains.llm import (
    assert_answer_relevancy,
    assert_faithfulness,
    assert_context_relevancy,
    assert_no_hallucination,
)


@pytest.mark.ml_model
def test_rag_pipeline_e2e():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:3000/ask")

        question = "What safety certifications does the product have?"

        # UI: ask a question
        page.fill("#question-input", question)
        page.click("#ask-button")
        page.wait_for_selector(".answer-text")

        answer = page.inner_text(".answer-text")

        # Extract the retrieved context chunks (if exposed in the UI)
        context_elements = page.query_selector_all(".source-chunk")
        context_chunks = [el.inner_text() for el in context_elements]
        context = " ".join(context_chunks)

        # mltk: context is relevant to the question
        assert_context_relevancy(
            question=question,
            context=context,
            min_score=0.3,
        )

        # mltk: answer is relevant to the question
        assert_answer_relevancy(
            question=question,
            answer=answer,
            min_score=0.3,
        )

        # mltk: answer is grounded in the context
        assert_faithfulness(
            answer=answer,
            context=context,
            min_score=0.5,
        )

        # mltk: no hallucinated claims
        assert_no_hallucination(
            claims=[answer],
            sources=context_chunks,
            min_coverage=0.3,
        )

        browser.close()
```

---

## Organizing tests with pytest markers

mltk registers markers that work well alongside Playwright tests. Use them to run UI-only or ML-only subsets.

```python
import pytest

# UI-only test — no ML assertions
def test_login_flow():
    ...

# ML-only test — no browser
@pytest.mark.ml_model
def test_model_accuracy():
    ...

# Combined: browser + ML
@pytest.mark.ml_model
def test_search_relevancy():
    ...

# Slow E2E that should not run on every PR
@pytest.mark.ml_slow
def test_full_rag_pipeline():
    ...

# Inference performance test
@pytest.mark.ml_inference
def test_api_latency():
    ...
```

Run subsets:

```bash
# All tests
pytest tests/e2e/

# Only ML quality tests
pytest tests/e2e/ -m ml_model

# Only inference/latency tests
pytest tests/e2e/ -m ml_inference

# Skip slow tests in PR CI
pytest tests/e2e/ -m "not ml_slow"

# With mltk HTML report
pytest tests/e2e/ --mltk-report -q
```

---

## Fixtures for browser reuse

Launching a browser per test is slow. Use a pytest fixture to share it.

```python
import pytest
from playwright.sync_api import sync_playwright, Browser, Page


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Browser):
    page = browser.new_page()
    yield page
    page.close()


@pytest.mark.ml_model
def test_search(page: Page):
    page.goto("http://localhost:3000")
    page.fill("#search-input", "python testing")
    page.click("#search-button")
    page.wait_for_selector(".results")

    results = page.query_selector_all(".result-item")
    for r in results:
        assert_answer_relevancy(
            question="python testing",
            answer=r.inner_text(),
            min_score=0.3,
        )
```

---

## Running in CI

```yaml
# .github/workflows/e2e-ml-tests.yml
name: E2E + ML Tests
on: [push, pull_request]

jobs:
  e2e-ml:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install mltk[all] playwright
          playwright install chromium --with-deps

      - name: Start application
        run: |
          docker compose up -d
          # Wait for the app to be ready
          timeout 60 bash -c 'until curl -s http://localhost:3000 > /dev/null; do sleep 2; done'

      - name: Run E2E + ML tests
        run: |
          pytest tests/e2e/ --mltk-report --mltk-export-json results.json -q

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: e2e-ml-reports
          path: mltk-reports/
```

---

## Suggested project structure

```
tests/
  e2e/
    conftest.py           # browser fixture, base URL config
    test_search.py        # search UI + answer relevancy
    test_chat.py          # chatbot UI + toxicity + coherence
    test_recommendations.py  # rec cards + diversity
    test_upload.py        # image upload + prediction confidence
    test_latency.py       # API latency (no browser needed)
  unit/
    test_model.py         # model accuracy, bias (mltk only)
    test_data.py          # schema, drift, PII (mltk only)
```

---

## Best practices

1. **Separate failure modes.** A UI failure ("button did not appear") and an ML failure ("answer not relevant") have different root causes. Keep them in separate assertions so failures are immediately actionable.

2. **Use realistic thresholds.** ML models are probabilistic. A `min_score=0.9` will produce flaky tests. Start with `0.3` and tighten based on observed distributions.

3. **Test with representative data.** Use real user queries and real images from production, not synthetic strings. What passes on "hello world" may fail on actual traffic.

4. **Use markers for CI speed.** Run `@pytest.mark.ml_smoke` on every PR. Run `@pytest.mark.ml_slow` nightly. Never block PRs on slow E2E + ML suites.

5. **Monitor latency under real DOM.** The browser adds overhead. Test ML endpoint latency directly (no browser) to isolate model performance from rendering performance.

6. **Share the browser.** Use `scope="session"` fixtures to launch the browser once, not per test. This cuts suite time by 50%+ on large suites.

7. **Generate reports.** Always run with `--mltk-report` in CI. The HTML report shows pass/fail, scores, and timing for every assertion -- useful for debugging failures in headless environments.

---

## Combining with mltk server

Push both E2E and ML results to the mltk dashboard for historical tracking:

```bash
pytest tests/e2e/ --mltk-server http://localhost:8080 --mltk-report
```

This sends results to the [server platform](../api/server-platform.md), where you can track score trends over time and get alerted on regressions.
