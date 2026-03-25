# LLM & GenAI Evaluation Patterns for mltk

Research completed: March 25, 2026

---

## Executive Summary

This research covers eight domains of LLM evaluation that mltk should implement to become a comprehensive ML testing toolkit -- not just for traditional ML, but also for generative AI. The backlog already references "LLM-as-judge evaluation patterns for generative AI" and Sprint 8 mentions `assert_no_prompt_injection`. This document expands those into a concrete implementation plan.

**Strategic note:** mltk's positioning is "pytest for ALL ML" -- not LLM-only like DeepEval. Adding LLM evaluation as a *domain kit* (alongside CV, NLP, Speech) preserves that positioning while closing the gap.

---

## 1. Hallucination Detection

### What It Is

Hallucination occurs when an LLM generates content that is factually incorrect, fabricated, or not grounded in the provided context. Two main types:

- **Intrinsic hallucination:** Contradicts the source/context (e.g., RAG system invents facts not in retrieved documents)
- **Extrinsic hallucination:** Cannot be verified from the source (neither supported nor contradicted)

### Detection Methods

| Method | How It Works | Cost | Accuracy |
|--------|-------------|------|----------|
| **Self-consistency** | Ask the same question N times, flag divergent answers | N x inference cost | Moderate |
| **Semantic entropy** | Measure entropy across semantically-clustered responses (Nature 2024) | N x inference cost | High |
| **QAG (Question-Answer Generation)** | Extract claims from output, generate questions, check if context answers them | 2-3 LLM calls | High |
| **Cross-reference** | Compare output against external knowledge base / search results | Variable | High |
| **NLI (Natural Language Inference)** | Use an NLI model to check if context entails each claim | Low (local model) | Moderate |
| **Calibration** | Measure model confidence vs actual accuracy | Low | Moderate |

### Key Insight

ROUGE-based evaluation dramatically overestimates method effectiveness. Detection methods show significant performance drops when evaluated against human-aligned metrics instead of ROUGE. Any mltk implementation should use semantic similarity or LLM-as-judge, never ROUGE alone.

### Proposed mltk API

```python
def assert_no_hallucination(
    output: str,
    context: list[str],           # source documents / retrieved chunks
    method: str = "nli",          # "nli" | "qag" | "self_consistency"
    threshold: float = 0.8,       # minimum faithfulness score (0-1)
    model: str | None = None,     # LLM for QAG method, NLI model for nli
) -> TestResult:
    """Asserts that the LLM output is grounded in the provided context."""
```

---

## 2. Semantic Similarity Scoring

### Methods Comparison

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **BERTScore** | Cosine similarity of contextual BERT embeddings (token-level greedy matching) | Low cost, interpretable (P/R/F1), captures paraphrases | Misses factual errors if phrasing is similar |
| **SemScore** | Single embedding cosine similarity (sentence-level) | Cheapest, fast | Too coarse for detailed evaluation |
| **ROUGE** | N-gram overlap | Fast, no model needed | Misses semantics, penalizes paraphrasing |
| **BLEU** | N-gram precision | Standard in MT | Poor for open-ended generation |
| **LLM-as-Judge** | LLM evaluates output quality against rubric | Most flexible, handles nuance | Expensive, judge can hallucinate |
| **G-Eval** | LLM + CoT + probability-weighted scoring | Highest human correlation (0.514 Spearman) | Requires API calls, non-deterministic |

### BERTScore Details

BERTScore computes three values:
1. **BERTPrecision:** For each token in candidate, find max cosine similarity with any reference token
2. **BERTRecall:** For each token in reference, find max cosine similarity with any candidate token
3. **BERTF1:** Harmonic mean of precision and recall

Advantages over ROUGE: captures synonyms and paraphrases, correlates better with human judgment, works across languages with multilingual models.

### G-Eval Details

G-Eval (NLG Evaluation using GPT-4 with Better Human Alignment) has three components:
1. **Prompt:** Task introduction + evaluation criteria
2. **Auto-CoT:** LLM generates evaluation steps using chain-of-thought
3. **Scoring:** Probability-weighted expected score (not a single sample)

G-Eval achieved 0.514 Spearman correlation with human judgments -- the highest of any automated metric.

### LLM-as-Judge Pattern

The most reliable evaluation method for open-ended tasks. Uses an LLM to grade outputs against natural-language rubrics. Variants:
- **Single-point grading:** Score one output on a rubric (1-5 Likert scale)
- **Pairwise comparison:** "Which response is better: A or B?"
- **Reference-graded:** Score output against a gold reference

### Proposed mltk API

```python
def assert_semantic_similarity(
    output: str,
    reference: str,
    method: str = "bertscore",    # "bertscore" | "cosine" | "rouge" | "bleu"
    threshold: float = 0.8,       # minimum similarity score
    model: str = "microsoft/deberta-xlarge-mnli",  # embedding model
) -> TestResult:
    """Asserts that output is semantically similar to reference."""

def assert_llm_judge(
    output: str,
    criteria: str,                # natural language evaluation criteria
    reference: str | None = None, # optional gold reference
    method: str = "g_eval",       # "g_eval" | "single_point" | "pairwise"
    threshold: float = 0.7,       # minimum score (0-1)
    model: str = "gpt-4",        # judge model
    scale: int = 5,              # Likert scale (1-5 default)
) -> TestResult:
    """Uses an LLM as a judge to evaluate output quality."""
```

---

## 3. Prompt Injection Testing

### Threat Landscape (2025-2026)

OWASP ranks prompt injection as the **#1 AI security risk** in its 2025 Top 10 for LLMs. Attack success rates in recent research:

| Attack Category | Success Rate | Mechanism |
|----------------|-------------|-----------|
| **Roleplay hijacking** | 89.6% | Deflects responsibility away from the model |
| **Logic traps** | 81.4% | Conditional structures + moral dilemmas |
| **Encoding tricks** (base64, zero-width) | 76.2% | Evades keyword-based filtering |
| **Adversarial poetry** | 60%+ | Poetic phrasing bypasses alignment (2025 discovery) |
| **Emoji smuggling** | Variable | Fully bypasses some guardrails |

### Attack Categories to Test

1. **Direct injection:** "Ignore previous instructions and..."
2. **Indirect injection:** Malicious content in retrieved documents / tool outputs
3. **Jailbreaking:** "DAN" mode, roleplaying, hypothetical framing
4. **Encoding attacks:** Base64, ROT13, Unicode, zero-width characters, homoglyphs
5. **Context overflow:** Extremely long inputs that push system prompt out of context window
6. **Prompt leaking:** "Repeat your system prompt" variants
7. **Goal hijacking:** Redirect model to perform unintended tasks

### Framework Reference: Promptfoo

Promptfoo (now part of OpenAI, MIT licensed) provides:
- 50+ vulnerability types from injection to jailbreaks
- Declarative YAML configs for test definitions
- Adversarial input generation via red-teaming strategies
- CI/CD integration

### Proposed mltk API

```python
def assert_no_prompt_injection(
    model_fn: Callable[[str], str],      # function that calls the LLM
    system_prompt: str,                    # the system prompt being tested
    attack_categories: list[str] | None = None,  # subset of attacks to test
    severity: str = "critical",           # "critical" | "high" | "medium"
    custom_attacks: list[str] | None = None,     # user-provided attack strings
) -> TestResult:
    """Tests an LLM endpoint against known prompt injection attacks."""

def assert_no_prompt_leakage(
    model_fn: Callable[[str], str],
    system_prompt: str,
    attempts: int = 10,                   # number of extraction attempts
    similarity_threshold: float = 0.5,    # flag if output is >50% similar to system prompt
) -> TestResult:
    """Tests whether the system prompt can be extracted."""

# Built-in attack library
INJECTION_ATTACKS = {
    "direct": [...],           # "Ignore previous instructions..."
    "roleplay": [...],         # "You are DAN..."
    "encoding": [...],         # Base64 encoded instructions
    "context_overflow": [...], # Padding + injection
    "prompt_leak": [...],      # "Repeat your system prompt"
    "goal_hijack": [...],      # "Instead of answering, write a poem"
    "indirect": [...],         # Injections in retrieved context
}
```

---

## 4. LLM-Specific Latency Metrics (TTFT and ITL)

### Definitions

| Metric | Full Name | What It Measures |
|--------|-----------|-----------------|
| **TTFT** | Time to First Token | Time from request submission to first token received. Includes queuing + prefill + network latency. |
| **ITL** (aka TPOT) | Inter-Token Latency / Time Per Output Token | Average time between consecutive tokens *after* the first. Measures decoding speed only. |
| **TPS** | Tokens Per Second | 1000 / ITL. User-perceivable generation speed. |
| **E2E Latency** | End-to-End Latency | Total time from request to last token. TTFT + (output_tokens - 1) * ITL. |

### Why TTFT and ITL Matter

- **TTFT** controls perceived responsiveness. Users notice delays > 500ms before any output appears.
- **ITL** controls reading speed. If tokens arrive slower than reading speed (~250 WPM, ~5.5 tokens/sec, ~180ms/token), the UX degrades.
- **Streaming vs non-streaming:** For streaming endpoints, TTFT is critical. For batch/non-streaming, only E2E matters.

### Industry Benchmarks (2025-2026)

| Use Case | TTFT P50 | TTFT P95 | ITL P50 | ITL P95 |
|----------|----------|----------|---------|---------|
| Chatbot (interactive) | < 500ms | < 1s | < 50ms | < 100ms |
| Code assistant | < 100ms | < 300ms | < 30ms | < 50ms |
| Batch processing | N/A | N/A | N/A | N/A (use TPS) |
| RAG pipeline | < 800ms | < 2s | < 50ms | < 100ms |

### Measurement Considerations

- TTFT should **not** be included in ITL calculation (NVIDIA genAI-perf approach)
- Warm-up requests should be excluded (cold start vs steady state)
- Report both streaming and non-streaming latency separately
- TTFT grows with prompt length (KV-cache computation scales with input)

### Proposed mltk API

```python
def assert_ttft(
    model_fn: Callable,           # streaming callable that yields tokens
    prompt: str,
    p50: float | None = None,     # max P50 TTFT in ms
    p95: float | None = None,     # max P95 TTFT in ms
    p99: float | None = None,     # max P99 TTFT in ms
    iterations: int = 50,
    warmup: int = 5,
) -> TestResult:
    """Asserts Time to First Token meets thresholds."""

def assert_itl(
    model_fn: Callable,           # streaming callable that yields tokens
    prompt: str,
    p50: float | None = None,     # max P50 ITL in ms
    p95: float | None = None,     # max P95 ITL in ms
    iterations: int = 50,
    warmup: int = 5,
) -> TestResult:
    """Asserts Inter-Token Latency (decoding speed) meets thresholds."""

def assert_tokens_per_second(
    model_fn: Callable,
    prompt: str,
    min_tps: float = 10.0,       # minimum tokens per second
    iterations: int = 50,
) -> TestResult:
    """Asserts generation speed meets minimum tokens/second."""
```

---

## 5. DeepEval Framework Analysis

### Overview

DeepEval (14.3K+ stars) is the fastest-growing LLM evaluation framework. Positioned as "pytest for LLMs" with 50+ metrics, multi-modal support, and CI/CD integration.

### Metrics Catalog

| Category | Metrics | Technique |
|----------|---------|-----------|
| **RAG** | Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall | QAG + LLM-as-Judge |
| **General** | G-Eval, Summarization, Coherence, Task Completion | LLM-as-Judge + CoT |
| **Safety** | Toxicity, Bias, Prompt Injection | LLM-as-Judge |
| **Hallucination** | Hallucination, Faithfulness | QAG + NLI |
| **Multi-turn** | Conversation Relevancy, Knowledge Retention, Role Adherence | ConversationalTestCase |
| **Non-LLM** | ExactMatch, PatternMatch, JsonSchema | Deterministic |
| **Custom** | G-Eval with user criteria | CoT + LLM-as-Judge |

### API Pattern

```python
# DeepEval's core pattern
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, FaithfulnessMetric

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France.",
    context=["France is a country in Europe. Paris is its capital."],
    expected_output="The capital of France is Paris.",
)

hallucination = HallucinationMetric(threshold=0.5)
faithfulness = FaithfulnessMetric(threshold=0.7)
evaluate(test_cases=[test_case], metrics=[hallucination, faithfulness])
```

### Key Design Decisions in DeepEval

1. **All scores are 0-1** with a threshold (default 0.5). Test passes if score >= threshold.
2. **Score reasoning** is always provided (human-readable explanation of the score).
3. **LLM-as-judge is default** for most metrics. Non-LLM metrics added in 2025.
4. **ConversationalTestCase** wraps multi-turn as a list of turns with a single evaluation.
5. **Model-agnostic:** Works with Ollama, OpenAI, Anthropic, Gemini via configuration.

### What mltk Should Adopt vs Differentiate

| Adopt | Differentiate |
|-------|---------------|
| 0-1 score + threshold pattern (matches mltk's TestResult) | mltk covers ALL ML, not just LLMs |
| Score reasoning in TestResult | mltk provides Rust-accelerated metrics where possible |
| ConversationalTestCase pattern for multi-turn | mltk integrates with pytest natively (DeepEval has its own runner) |
| G-Eval as the flexible LLM-as-judge method | mltk offers non-LLM fallbacks (BERTScore, NLI) for cost savings |

---

## 6. Evaluation Framework Landscape

### RAGAS (RAG Assessment)

Purpose-built for RAG pipeline evaluation. Five core metrics:

| Metric | What It Measures | Inputs Required |
|--------|-----------------|-----------------|
| **Faithfulness** | Are all claims in the answer supported by context? | response, contexts |
| **Answer Relevancy** | Is the answer relevant to the question? | question, response, contexts |
| **Context Precision** | Are relevant chunks ranked highest? | question, contexts, ground_truth |
| **Context Recall** | Does context contain all needed info? | contexts, ground_truth |
| **Answer Correctness** | Is the answer factually correct? | response, ground_truth |

**Key insight for mltk:** RAGAS is a library (no dashboard, no experiment tracking). Its metrics can be implemented directly in mltk without depending on RAGAS as a dependency.

### TruLens

Uses "feedback functions" injected after each LLM call. Three pillars:
1. **Groundedness:** Is the response grounded in retrieved context?
2. **Answer Relevance:** Does the response answer the question?
3. **Context Relevance:** Is the retrieved context relevant to the question?

Transparent and auditable -- every evaluation decision can be inspected.

### Arize Phoenix

OpenTelemetry-based tracing and observability for LLM apps. Strengths:
- Traces execution paths through multi-step LLM requests
- Dataset clustering and visualization via embeddings
- RAG-focused troubleshooting (correlating retrieval failures with generation errors)

**Key insight for mltk:** Phoenix is observability, not testing. mltk should focus on assertions, not tracing. But Phoenix's approach to embedding-based clustering could inspire drift detection for LLM outputs.

### Promptfoo

Red teaming and prompt testing framework. Now part of OpenAI, remains MIT licensed.
- 50+ vulnerability types
- Declarative YAML test configs
- Adversarial input generation
- Used by OpenAI and Anthropic

**Key insight for mltk:** Promptfoo's attack library and YAML test format are worth studying. mltk's planned YAML-driven test definitions could adopt similar patterns for LLM security testing.

---

## 7. Toxicity & Safety Testing

### Safety Evaluation Dimensions

| Dimension | What to Test | Tools/Benchmarks |
|-----------|-------------|-----------------|
| **Toxicity** | Hate speech, profanity, threats | Detoxify, Perspective API, ToxicChat |
| **Bias** | Gender, racial, religious, age bias in outputs | BBQ, BOLD, WinoBias |
| **PII leakage** | Model outputs personal information | Custom regex + NER |
| **Harmful instructions** | Weapons, drugs, self-harm content | HarmBench, JailbreakBench |
| **Overrefusal** | Model refuses benign requests (exaggerated safety) | XSTest |

### Detection Approaches

1. **Classifier-based:** Fine-tuned models (Detoxify, Perspective API) score toxicity per category. Cheap and fast.
2. **LLM-as-judge:** Ask an LLM to evaluate safety. More flexible but expensive.
3. **Guardrail models:** Dedicated safety models (Qwen3Guard, Llama Guard). Best accuracy but largest models.

### Guardrail Model Benchmarks (2025)

Qwen3Guard-8B achieved the highest overall accuracy of 85.3% across 1,445 test prompts and 21 attack categories. However, all models showed substantial degradation on unseen prompts (Qwen3Guard dropped from 91.0% to 33.8%).

**Key insight for mltk:** Do not rely solely on guardrail models. Combine a fast classifier (Detoxify) for screening with LLM-as-judge for nuanced cases.

### Proposed mltk API

```python
def assert_no_toxicity(
    output: str,
    categories: list[str] | None = None,  # "hate" | "threat" | "profanity" | "sexual" | "self_harm"
    threshold: float = 0.1,               # max toxicity score (0-1)
    model: str = "detoxify",              # "detoxify" | "perspective" | "llm_judge"
) -> TestResult:
    """Asserts that output contains no toxic content."""

def assert_no_bias_in_text(
    model_fn: Callable[[str], str],
    prompts: list[str],                    # same prompt with different demographic terms
    similarity_threshold: float = 0.9,     # outputs should be similar across demographics
    method: str = "bertscore",
) -> TestResult:
    """Asserts that LLM outputs do not vary significantly across demographic groups."""

def assert_no_pii_in_output(
    output: str,
    categories: list[str] | None = None,  # "email" | "phone" | "ssn" | "credit_card" | ...
) -> TestResult:
    """Asserts that LLM output does not contain personally identifiable information."""

def assert_no_overrefusal(
    model_fn: Callable[[str], str],
    benign_prompts: list[str],             # clearly safe prompts
    min_response_rate: float = 0.95,       # at least 95% should get substantive responses
) -> TestResult:
    """Asserts that the model does not refuse benign requests (XSTest pattern)."""
```

---

## 8. Multi-Turn Conversation Testing

### Why Single-Turn Testing Is Insufficient

Conversations build meaning across turns with references, pronouns, and implicit understanding. A model might perform well on isolated responses but fail to maintain consistent personality, facts, or logical threads across multiple exchanges.

### Evaluation Dimensions

| Dimension | What It Measures | Example Failure |
|-----------|-----------------|-----------------|
| **Coherence** | Responses connect to and build upon previous turns | Contradicts something said 3 turns ago |
| **Context retention** | Model remembers earlier information | Forgets user's name mentioned in turn 1 |
| **Reference resolution** | Correctly resolves pronouns and references | "it" refers to wrong entity |
| **Role adherence** | Maintains assigned persona consistently | Breaks character mid-conversation |
| **Task completion** | Fulfills the user's overall goal across turns | Loses track of the task after a follow-up |
| **Knowledge retention** | Retains facts introduced during conversation | Contradicts a fact it stated earlier |
| **Conversation relevancy** | Each response is relevant to the overall thread | Goes off-topic after topic shift |

### Evaluation Approaches

1. **MT-Bench:** Multi-turn benchmark with open-ended questions. LLM-as-judge scoring. Standard for chatbot evaluation.
2. **ConvBench:** NeurIPS 2024 multi-turn benchmark for structured evaluation.
3. **RAGAS multi-turn:** Evaluates conversations using same metrics as single-turn but across turn sequences.
4. **Automated reference resolution:** Tests pronoun and contextual reference accuracy.

### Proposed mltk API

```python
@dataclass
class ConversationTurn:
    role: str          # "user" | "assistant" | "system"
    content: str

@dataclass
class ConversationTestCase:
    turns: list[ConversationTurn]
    metadata: dict | None = None

def assert_conversation_coherence(
    test_case: ConversationTestCase,
    threshold: float = 0.7,
    model: str | None = None,     # LLM-as-judge model
) -> TestResult:
    """Asserts that assistant responses are coherent across all turns."""

def assert_knowledge_retention(
    model_fn: Callable[[list[dict]], str],  # takes message history, returns response
    facts: list[str],                        # facts to introduce and later test
    num_intervening_turns: int = 3,          # turns between fact introduction and test
    threshold: float = 0.8,
) -> TestResult:
    """Asserts that the model retains facts introduced earlier in conversation."""

def assert_role_adherence(
    test_case: ConversationTestCase,
    role_description: str,         # expected persona / behavior
    threshold: float = 0.8,
    model: str | None = None,
) -> TestResult:
    """Asserts that the assistant maintains its assigned role throughout."""

def assert_context_window_handling(
    model_fn: Callable,
    conversation_length: int = 50,  # number of turns
    fact_positions: list[int] | None = None,  # turns where facts are introduced
    threshold: float = 0.7,
) -> TestResult:
    """Tests model behavior as conversation approaches context window limits."""
```

---

## Implementation Roadmap for mltk

### Phase 1: LLM Domain Kit Foundation (Sprint 11 or 12)

**Priority:** Highest-impact, lowest-complexity assertions first.

| Assertion | Priority | Complexity | Dependencies |
|-----------|----------|------------|-------------|
| `assert_semantic_similarity` (BERTScore) | P0 | Low | sentence-transformers |
| `assert_no_toxicity` (Detoxify) | P0 | Low | detoxify |
| `assert_no_pii_in_output` | P0 | Low | reuse existing PII scanner |
| `assert_ttft` | P0 | Medium | streaming client support |
| `assert_itl` | P0 | Medium | streaming client support |

### Phase 2: LLM-as-Judge + Safety (Sprint 13)

| Assertion | Priority | Complexity | Dependencies |
|-----------|----------|------------|-------------|
| `assert_llm_judge` (G-Eval) | P1 | Medium | LLM API access |
| `assert_no_hallucination` (NLI) | P1 | Medium | NLI model |
| `assert_no_prompt_injection` | P1 | High | attack library |
| `assert_no_prompt_leakage` | P1 | Medium | similarity check |

### Phase 3: RAG + Multi-Turn (Sprint 14)

| Assertion | Priority | Complexity | Dependencies |
|-----------|----------|------------|-------------|
| `assert_faithfulness` (RAGAS-style) | P2 | Medium | LLM API |
| `assert_answer_relevancy` | P2 | Medium | LLM API |
| `assert_context_precision` | P2 | Medium | LLM API |
| `assert_conversation_coherence` | P2 | High | ConversationTestCase type |
| `assert_knowledge_retention` | P2 | High | multi-turn test harness |
| `assert_role_adherence` | P2 | Medium | LLM-as-judge |

### Phase 4: Advanced (Sprint 15+)

| Assertion | Priority | Complexity | Dependencies |
|-----------|----------|------------|-------------|
| `assert_no_bias_in_text` | P2 | Medium | BERTScore + demographic prompts |
| `assert_no_overrefusal` | P2 | Medium | benign prompt library |
| `assert_context_window_handling` | P3 | High | long conversation generation |
| `assert_tokens_per_second` | P3 | Low | streaming client |
| Red team YAML test format | P3 | High | parser + attack library |

### Dependencies to Add

```toml
# pyproject.toml extras
[project.optional-dependencies]
llm = [
    "sentence-transformers>=3.0",   # BERTScore, semantic similarity
    "detoxify>=0.5",                # toxicity classification
    "transformers>=4.40",           # NLI models for hallucination
]
llm-judge = [
    "openai>=1.0",                  # for G-Eval / LLM-as-judge
    "anthropic>=0.30",              # alternative judge model
]
```

### Rust Acceleration Opportunities

| Component | Rust Benefit | Approach |
|-----------|-------------|----------|
| BERTScore token matching | O(n*m) cosine similarity matrix | SIMD-accelerated cosine via simsimd |
| PII scanning in outputs | Regex matching at scale | reuse existing mltk-rs PII scanner |
| Prompt injection patterns | Pattern matching against attack library | Aho-Corasick multi-pattern matching |
| Token timing measurement | Sub-millisecond precision | Rust timing + PyO3 bridge |

---

## Competitive Positioning

| Feature | mltk (planned) | DeepEval | RAGAS | Promptfoo |
|---------|---------------|----------|-------|-----------|
| Traditional ML testing | Full suite | None | None | None |
| LLM evaluation | Domain kit | Core focus | RAG only | Red team only |
| Rust acceleration | Yes | No | No | No |
| pytest integration | Native | Custom runner | No | No |
| Cost (LLM-free metrics) | BERTScore, NLI, Detoxify | LLM-as-judge default | LLM required | LLM required |
| Multi-modal | Planned (CV kit) | Yes | No | Limited |
| Prompt injection | Built-in attack lib | Basic | No | 50+ attacks |
| YAML test definitions | Planned | No | No | Yes |

**mltk's unique advantage:** The only framework that tests the full ML lifecycle (data quality, training bugs, model quality, bias/fairness, inference, AND LLM evaluation) in a single toolkit with Rust acceleration and native pytest integration. DeepEval and RAGAS only cover LLM evaluation. Promptfoo only covers security testing.

---

## Sources

### Hallucination Detection
- [Lakera: Guide to Hallucinations in LLMs](https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models)
- [Comprehensive Survey of Hallucination in LLMs](https://arxiv.org/abs/2510.06265)
- [Detecting Hallucinations Using Semantic Entropy (Nature)](https://www.nature.com/articles/s41586-024-07421-0)
- [Deepchecks: LLM Hallucination Detection and Mitigation](https://deepchecks.com/llm-hallucination-detection-and-mitigation-best-techniques/)

### Semantic Similarity
- [BERTScore for LLM Evaluation (Comet)](https://www.comet.com/site/blog/bertscore-for-llm-evaluation/)
- [BERTScore: A Contextual Metric for LLM Evaluation](https://www.analyticsvidhya.com/blog/2025/04/bertscore-a-contextual-metric-for-llm-evaluation/)
- [Confident AI: LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [SemScore: Evaluating LLMs Using Semantic Similarity](https://www.emergentmind.com/papers/2401.17072)

### Prompt Injection
- [OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
- [OWASP Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
- [Adversarial Poetry: 2025 LLM Jailbreak Vulnerability](https://aviatrix.ai/threat-research-center/prompt-injection-poetry-2025-llm-vulnerability/)
- [Red Teaming the Mind of the Machine (arxiv)](https://arxiv.org/html/2505.04806v1)

### TTFT and ITL
- [NVIDIA NIM LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- [NVIDIA LLM Benchmarking Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [BentoML: Key Metrics for LLM Inference](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [Anyscale: LLM Latency and Throughput Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)

### DeepEval
- [DeepEval GitHub (confident-ai/deepeval)](https://github.com/confident-ai/deepeval)
- [DeepEval Metrics Introduction](https://deepeval.com/docs/metrics-introduction)
- [DeepEval Hallucination Metric](https://deepeval.com/docs/metrics-hallucination)
- [DeepEval G-Eval](https://deepeval.com/docs/metrics-llm-evals)

### Evaluation Frameworks
- [RAGAS: Faithfulness Metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [RAGAS: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [LLM Evaluation Frameworks Comparison (Comet)](https://www.comet.com/site/blog/llm-evaluation-frameworks/)
- [G-Eval: LLM-as-a-Judge Definitive Guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)

### Toxicity & Safety
- [LLM Guardrails: Strategies & Best Practices 2025](https://www.leanware.co/insights/llm-guardrails)
- [Evaluating Robustness of LLM Safety Guardrails (arxiv)](https://arxiv.org/html/2511.22047v1)
- [Guardrails AI: Toxic Language Validator](https://github.com/guardrails-ai/toxic_language)
- [Avidoai: LLM Guardrail Testing 2025](https://avidoai.com/blog/llm-guardrail-testing)

### Multi-Turn Conversation
- [RAGAS: Evaluating Multi-Turn Conversations](https://docs.ragas.io/en/stable/howtos/applications/evaluating_multi_turn_conversations/)
- [Survey: Evaluating LLM-based Agents for Multi-Turn Conversations](https://arxiv.org/html/2503.22458v1)
- [Confident AI: LLM Chatbot Evaluation Metrics](https://www.confident-ai.com/blog/llm-chatbot-evaluation-explained-top-chatbot-evaluation-metrics-and-testing-techniques)
- [ConvBench: Multi-Turn Conversation Evaluation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/b69396afc07a9ca3428d194f4db84c02-Paper-Datasets_and_Benchmarks_Track.pdf)

### Promptfoo
- [Promptfoo GitHub](https://github.com/promptfoo/promptfoo)
- [Promptfoo Red Teaming Guide](https://www.promptfoo.dev/docs/red-team/)
- [Promptfoo Red Team Strategies](https://www.promptfoo.dev/docs/red-team/strategies/)
