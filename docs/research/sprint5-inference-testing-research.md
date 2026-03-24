# Sprint 5 Research: ML Inference Testing Patterns

Research completed: March 25, 2026

---

## 1. Latency Testing (assert_latency)

### How Companies Test Inference Latency

Production ML systems measure latency using **percentile distributions**, never averages. An average hides tail latency -- a P50 of 20ms with a P99 of 2000ms means 1 in 100 users waits 100x longer than the median user.

**Standard percentiles:**
- **P50 (median):** Typical user experience. The baseline.
- **P95:** SLA-grade. 95% of requests are this fast or faster. Most common SLA target.
- **P99:** Near worst-case. Catches tail latency from GC pauses, cold caches, network jitter.

### Industry SLA Thresholds (2025-2026)

| Use Case | P50 | P95 | P99 | Source |
|----------|-----|-----|-----|--------|
| Real-time classification (image, tabular) | < 10ms | < 50ms | < 100ms | Industry standard |
| NLP / text classification | < 50ms | < 200ms | < 500ms | Common practice |
| LLM TTFT (chatbot) | < 500ms | < 1s | < 2s | MLPerf v5.1 |
| LLM TTFT (code assistant) | < 100ms | < 300ms | < 500ms | MLPerf v5.1 |
| LLM TPOT (interactive) | < 30ms | < 50ms | < 100ms | MLPerf v5.1 |
| Object detection (video) | < 33ms | < 50ms | < 100ms | 30fps requirement |
| Recommendation API | < 20ms | < 50ms | < 100ms | Industry standard |

### Implementation Pattern for mltk

```python
import time
import numpy as np

def assert_latency(
    func,                          # callable to benchmark
    *args,                         # positional args to func
    p50: float | None = None,      # max P50 in ms
    p95: float | None = None,      # max P95 in ms
    p99: float | None = None,      # max P99 in ms
    iterations: int = 100,         # number of timed calls
    warmup: int = 5,               # warm-up calls (excluded from stats)
    **kwargs,                      # keyword args to func
) -> TestResult:
    # 1. Warm-up phase (exclude from measurement)
    for _ in range(warmup):
        func(*args, **kwargs)

    # 2. Measurement phase
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    arr = np.array(latencies)

    # 3. Compute percentiles
    actual_p50 = float(np.percentile(arr, 50))
    actual_p95 = float(np.percentile(arr, 95))
    actual_p99 = float(np.percentile(arr, 99))

    # 4. Check thresholds
    failures = []
    if p50 is not None and actual_p50 > p50:
        failures.append(f"P50 {actual_p50:.2f}ms > {p50}ms")
    if p95 is not None and actual_p95 > p95:
        failures.append(f"P95 {actual_p95:.2f}ms > {p95}ms")
    if p99 is not None and actual_p99 > p99:
        failures.append(f"P99 {actual_p99:.2f}ms > {p99}ms")

    # 5. Return TestResult with full distribution in details
```

### Key Design Decisions

1. **Warm-up is mandatory.** First calls after model load are 10-100x slower (JIT compilation, cache population, GPU kernel compilation). Default 5 warm-up iterations.
2. **Use `time.perf_counter()`**, not `time.time()`. It has nanosecond resolution and is monotonic.
3. **Return the full distribution** in `details` (min, max, mean, std, P50, P95, P99, all raw latencies) so reports can render histograms.
4. **At least one threshold required.** If user passes no percentile args, raise ValueError -- the function must assert something.
5. **Severity = CRITICAL by default.** Latency violations in production are SLA breaches.

---

## 2. Throughput Testing (assert_throughput)

### How Companies Test Throughput

Throughput measures how many inferences the system can sustain per second. Two modes matter:

**Single-client throughput:** How fast can one caller fire sequential requests?
**Concurrent throughput:** How many requests/second at N concurrent clients?

### Tools Used in Industry

| Tool | Language | Best For | Key Feature |
|------|----------|----------|-------------|
| **Locust** | Python | ML APIs | Python scripting, distributed, real-time web UI |
| **k6** | Go (JS scripts) | HTTP APIs | Constant VU/RPS modes, Grafana integration |
| **vegeta** | Go | Constant-rate testing | Precise RPS control, histogram output |
| **wrk** | C | Raw HTTP perf | Lua scripting, very high throughput |
| **hey** | Go | Quick benchmarks | Simple CLI, percentile output |
| **LLM Locust** | Python | LLM endpoints | Token-level metrics, streaming support |
| **pytest-benchmark** | Python | In-process | Native pytest integration, regression tracking |

### MLPerf Inference Scenarios (Industry Standard)

MLPerf v5.1 (2025) defines 4 scenarios that map to real deployment patterns:

1. **Single-Stream:** One request at a time, measure latency. Maps to edge/mobile.
2. **Multi-Stream:** Multiple concurrent streams, measure max streams within latency. Maps to autonomous driving.
3. **Server:** Poisson-distributed arrivals, measure throughput under latency constraint. Maps to cloud APIs.
4. **Offline:** Process entire dataset, measure raw throughput. Maps to batch inference.

### Implementation Pattern for mltk

```python
import time
import concurrent.futures

def assert_throughput(
    func,                          # callable to benchmark
    *args,
    min_rps: float,                # minimum requests per second
    duration: float = 5.0,         # test duration in seconds
    concurrency: int = 1,          # number of concurrent workers
    warmup: int = 5,               # warm-up calls
    **kwargs,
) -> TestResult:
    # 1. Warm-up
    for _ in range(warmup):
        func(*args, **kwargs)

    # 2. Run for `duration` seconds with `concurrency` workers
    completed = 0
    errors = 0
    start = time.perf_counter()
    deadline = start + duration

    if concurrency == 1:
        # Sequential mode
        while time.perf_counter() < deadline:
            try:
                func(*args, **kwargs)
                completed += 1
            except Exception:
                errors += 1
    else:
        # Concurrent mode using ThreadPoolExecutor
        # (or ProcessPoolExecutor for CPU-bound inference)
        ...

    elapsed = time.perf_counter() - start
    actual_rps = completed / elapsed

    # 3. Assert
    passed = actual_rps >= min_rps
```

### Key Design Decisions

1. **Duration-based, not count-based.** Throughput is "how many in N seconds", not "how fast to do N calls." Default 5 seconds.
2. **Concurrency parameter.** Single-threaded by default (simplest), but `concurrency > 1` uses ThreadPoolExecutor to simulate concurrent API clients.
3. **Track errors separately.** A system that hits 1000 RPS but 30% are errors is not actually achieving 1000 RPS of useful work (the "goodput" concept from MLPerf).
4. **Return error_rate in details.** Report `completed`, `errors`, `actual_rps`, `error_rate`, `duration`.

---

## 3. API Contract Testing (assert_api_contract)

### How Companies Validate ML API Schemas

ML API contract testing validates that inference endpoints accept the expected input and produce the expected output. This catches:

- **Schema drift:** Model retrained with different features, but API caller sends old schema
- **Type mismatches:** Float vs int, string vs categorical encoding
- **Missing fields:** New required field added to model input
- **Output shape changes:** Classification model suddenly returns different number of classes
- **Extra fields leaking:** Internal model metadata exposed in response

### Standard Approaches

1. **JSON Schema validation** -- define expected request/response as JSON Schema, validate against it. Most portable.
2. **Pydantic models** -- define Python dataclasses, validate at runtime. Fastest in Python (3.5x faster than jsonschema in benchmarks). Native FastAPI integration.
3. **OpenAPI/Swagger specs** -- full API contract including endpoints, methods, headers. Tools: Dredd, Schemathesis, Specmatic.
4. **Consumer-Driven Contract Testing** -- consumers define what they expect, provider validates it meets all consumer contracts. Tools: Pact.

### Implementation Pattern for mltk

```python
import json

def assert_api_contract(
    func,                          # callable (inference function)
    input_data,                    # sample input
    input_schema: dict | None = None,   # JSON Schema for input
    output_schema: dict | None = None,  # JSON Schema for output
    input_type: type | None = None,     # Python type/Pydantic model for input
    output_type: type | None = None,    # Python type/Pydantic model for output
    allow_extra_fields: bool = True,    # strict or permissive
) -> TestResult:
    # 1. Validate input against schema (if provided)
    if input_schema:
        _validate_json_schema(input_data, input_schema)
    if input_type:
        _validate_type(input_data, input_type)

    # 2. Call the function
    output = func(input_data)

    # 3. Validate output against schema
    if output_schema:
        _validate_json_schema(output, output_schema)
    if output_type:
        _validate_type(output, output_type)

    # 4. Return TestResult with details about what was validated
```

### JSON Schema Approach (No Extra Dependencies)

```python
# User defines schema as plain dict (JSON Schema draft 2020-12)
input_schema = {
    "type": "object",
    "properties": {
        "features": {"type": "array", "items": {"type": "number"}},
        "model_version": {"type": "string"},
    },
    "required": ["features"],
}

output_schema = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "model_version": {"type": "string"},
    },
    "required": ["prediction"],
}
```

### Key Design Decisions

1. **Support both JSON Schema dicts and Python types.** JSON Schema is portable (language-agnostic). Python types/Pydantic are ergonomic. Support both.
2. **jsonschema library is optional.** If user passes a JSON Schema dict, import jsonschema lazily (like sklearn in model metrics). If they pass a Python type, use isinstance/dataclass checks with no extra deps.
3. **Validate both input AND output.** Input validation catches "did I send the right thing?" Output validation catches "did the model return what I expected?"
4. **allow_extra_fields default True.** ML APIs frequently add metadata fields (latency, model_version, request_id). Strict mode available for locked-down contracts.

---

## 4. Cold Start Testing (assert_cold_start)

### The Problem

The first inference after model load is dramatically slower:
- **CPU models:** 2-10x slower (cache population, memory allocation)
- **GPU models:** 10-100x slower (CUDA kernel compilation, weight transfer to GPU)
- **Serverless:** 5-20 seconds (container boot + model download + initialization)

Teams that don't test cold start get surprised in production when autoscaling spins up new instances under load.

### Standard Testing Pattern

```python
def assert_cold_start(
    setup_func,                    # function that loads/initializes the model
    inference_func,                # function that runs inference
    *args,
    max_cold_ms: float,            # max acceptable cold start latency
    max_warm_ms: float | None = None,  # optional warm comparison
    **kwargs,
) -> TestResult:
    # 1. Run setup (model loading)
    model = setup_func()

    # 2. Time the FIRST call (cold)
    start = time.perf_counter()
    inference_func(model, *args, **kwargs)
    cold_ms = (time.perf_counter() - start) * 1000

    # 3. Time a warm call for comparison
    inference_func(model, *args, **kwargs)  # throw away
    start = time.perf_counter()
    inference_func(model, *args, **kwargs)
    warm_ms = (time.perf_counter() - start) * 1000

    # 4. Assert cold start is within budget
    # 5. Report cold/warm ratio in details
```

### Industry Cold Start Budgets

| Deployment | Cold Start Budget | Notes |
|------------|-------------------|-------|
| Edge/Mobile | < 500ms | User is waiting |
| Real-time API | < 2s | Behind load balancer with warm instances |
| Batch pipeline | < 30s | Amortized over thousands of inferences |
| Serverless (Lambda/Modal) | < 10s | Container + model load |

---

## 5. Batch vs Single Inference Testing

### Two Distinct Patterns

**Online (single) inference:**
- One request at a time (or small batches)
- Optimized for latency (P95 < X ms)
- SLA: "every request responds within budget"

**Batch inference:**
- Process entire dataset at once
- Optimized for throughput (items/second)
- SLA: "process N items within T minutes"
- Typically 50% cheaper than real-time (Google Cloud pricing)

### Testing Strategy

`assert_latency` covers single inference. For batch, we need a separate check:

```python
def assert_batch_latency(
    func,                          # batch inference function
    batch_data,                    # list/array of inputs
    max_total_ms: float | None = None,    # max total time
    max_per_item_ms: float | None = None, # max average per item
    min_items_per_sec: float | None = None,
) -> TestResult:
    start = time.perf_counter()
    results = func(batch_data)
    elapsed_ms = (time.perf_counter() - start) * 1000

    n = len(batch_data)
    per_item = elapsed_ms / n
    items_per_sec = n / (elapsed_ms / 1000)
    # Assert against thresholds
```

### Key Insight

Batch inference should be **faster per item** than single inference due to GPU parallelism and framework-level batching (dynamic batching in Triton, batch inference in TorchServe). If batch is SLOWER per item, something is wrong (no vectorization, unnecessary per-item overhead).

---

## 6. Timeout and Graceful Degradation Testing

### Standard Patterns

Production ML systems implement multi-level fallback:

1. **Level 1:** Full model inference (primary path)
2. **Level 2:** Cached prediction from recent similar query
3. **Level 3:** Lightweight fallback model (e.g., logistic regression instead of deep network)
4. **Level 4:** Heuristic / rule-based default

### Circuit Breaker Pattern

Circuit breaker tracks failure rates, opens after threshold (typically 5 failures in 60 seconds), and automatically resets after cooldown. This is standard in microservice architectures and applies directly to ML serving.

### Testing Strategy for mltk

```python
def assert_timeout(
    func,
    *args,
    timeout_ms: float,             # max allowed response time
    fallback: callable | None = None,  # optional fallback function
    **kwargs,
) -> TestResult:
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > timeout_ms:
            # Timed out -- test if fallback works
            if fallback:
                fallback_result = fallback(*args, **kwargs)
                # Assert fallback produced valid output
    except Exception as e:
        # Function errored -- test if fallback works
        ...
```

### Recommendation for Sprint 5

Timeout testing is valuable but adds complexity. **Defer `assert_timeout` to Sprint 9 (Monitoring)** where it fits naturally with `assert_sla` and `drift_monitor`. Sprint 5 should focus on the core three: latency, throughput, contract.

---

## 7. Model Serving Framework Test Patterns

### Framework-Specific Considerations

**FastAPI + uvicorn (most common for Python ML):**
- Test via HTTP client (httpx, requests)
- Warm-up: first request triggers model load via `startup` event
- Contract: Pydantic models define request/response automatically

**TorchServe:**
- Health endpoint: `GET /ping`
- Prediction: `POST /predictions/{model_name}`
- Metrics endpoint: `GET /metrics` (Prometheus format)
- Dynamic batching configured in `config.properties`

**Triton Inference Server:**
- Health: `GET /v2/health/ready`
- Inference: `POST /v2/models/{name}/infer`
- Model config: `config.pbtxt` defines input/output tensors
- gRPC and HTTP support

**TF Serving:**
- Predict: `POST /v1/models/{name}:predict`
- Input/output: TensorFlow `SignatureDef` defines schema

### mltk's Approach: Framework-Agnostic

mltk should NOT depend on any serving framework. The user passes a callable:

```python
# Works with any serving framework
def my_inference(input_data):
    response = httpx.post("http://localhost:8080/predict", json=input_data)
    return response.json()

# Or with a local model
def my_inference(input_data):
    return model.predict(input_data)

# mltk doesn't care -- it just calls the function and measures
assert_latency(my_inference, sample_input, p95=50.0)
```

---

## 8. Summary: What mltk Should Implement in Sprint 5

### Priority 1 (MUST ship)

| Function | Description | Key Params |
|----------|-------------|------------|
| `assert_latency` | Percentile latency check | `p50`, `p95`, `p99`, `iterations`, `warmup` |
| `assert_throughput` | RPS / items-per-second check | `min_rps`, `duration`, `concurrency` |
| `assert_api_contract` | Input/output schema validation | `input_schema`, `output_schema` (JSON Schema dicts) |

### Priority 2 (SHOULD ship if time permits)

| Function | Description | Key Params |
|----------|-------------|------------|
| `assert_cold_start` | First-request-after-load latency | `max_cold_ms`, `setup_func` |
| `assert_batch_latency` | Batch processing performance | `max_per_item_ms`, `min_items_per_sec` |

### Priority 3 (Defer to later sprints)

| Function | Target Sprint | Reason |
|----------|---------------|--------|
| `assert_timeout` | Sprint 9 | Fits with monitoring/SLA module |
| `assert_error_rate` | Sprint 9 | Fits with production monitoring |
| Concurrent load testing | Sprint 9 | Complex (thread pools), fits monitoring |

### File Structure

```
src/mltk/inference/
    __init__.py          # re-export public API
    latency.py           # assert_latency, assert_cold_start
    throughput.py         # assert_throughput, assert_batch_latency
    contract.py          # assert_api_contract
```

### Test Structure

```
tests/test_inference/
    __init__.py
    test_latency.py      # ~15 tests
    test_throughput.py    # ~10 tests
    test_contract.py     # ~12 tests
```

### Documentation

```
docs/api/
    inference-latency.md
    inference-throughput.md
    inference-contract.md
```

### Dependencies

- **No new required dependencies.** All core functions use only numpy (already required) and stdlib `time`, `concurrent.futures`.
- **Optional:** `jsonschema` for JSON Schema validation (lazy import like sklearn). Without it, only Python type checking is available for contract testing.
- **Optional:** `httpx` for testing remote endpoints (not imported by mltk, user brings their own HTTP client).

### Design Principles (Consistent with Existing mltk Patterns)

1. Every function is decorated with `@timed_assertion` (matches drift.py, schema.py pattern)
2. Every function returns `TestResult` (never raw bool)
3. Details dict contains full measurement data for reports
4. Lazy imports for optional deps (matches metrics.py sklearn pattern)
5. Sensible defaults that work without configuration
6. `Severity.CRITICAL` by default (raises `MltkAssertionError` on failure)

---

## Sources

- [Measuring Inference Latency and Throughput](https://apxml.com/courses/quantized-llm-deployment/chapter-3-performance-evaluation-quantized-llms/measuring-inference-latency-throughput)
- [Key Metrics for LLM Inference (BentoML)](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [Load Testing AI Systems: Latency Under Pressure](https://thread-transfer.com/blog/2025-08-18-load-testing-ai/)
- [MLPerf Inference Benchmarks](https://docs.mlcommons.org/inference/)
- [MLPerf Inference v5.0 Results (MLCommons)](https://mlcommons.org/2025/04/llm-inference-v5/)
- [MLPerf Inference v5.1 Results (MLCommons)](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- [Understand LLM Latency and Throughput Metrics (Anyscale)](https://docs.anyscale.com/llm/serving/benchmarking/metrics)
- [Batch Inference vs Online Inference](https://mlinproduction.com/batch-inference-vs-online-inference/)
- [Cold Start Latency in LLM Inference](https://acecloud.ai/blog/cold-start-latency-llm-inference/)
- [Best Load Testing Tools 2026 (Vervali)](https://www.vervali.com/blog/best-load-testing-tools-in-2026-definitive-guide-to-jmeter-gatling-k6-loadrunner-locust-blazemeter-neoload-artillery-and-more/)
- [LLM Locust Benchmarking (TrueFoundry)](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [API Contract Testing with JSON Schema Validation (Medium)](https://medium.com/@abhishek.builds/api-contract-testing-with-restassured-and-json-schema-validation-2025-guide-64be23d6f765)
- [How API Schema Validation Boosts Contract Testing (Zuplo)](https://zuplo.com/learning-center/how-api-schema-validation-boosts-effective-contract-testing)
- [JSON Schema - Pydantic Validation](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Reducing Cold Start Latency (NVIDIA)](https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/)
- [AI Agent Retry Strategies: Exponential Backoff and Graceful Degradation](https://getathenic.com/blog/ai-agent-retry-strategies-exponential-backoff)
- [Model Serving Comparison 2026: TF Serving vs TorchServe vs Triton](https://reintech.io/blog/model-serving-comparison-tensorflow-serving-torchserve-triton-inference-server)
- [Best Tools for ML Model Serving (Neptune.ai)](https://neptune.ai/blog/ml-model-serving-best-tools)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/en/latest/)
- [numpy.percentile Documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
- [Serving ML Models at Scale (Sealos)](https://sealos.io/blog/serving-machine-learning-models-at-scale-a-guide-to-inference-optimization/)
