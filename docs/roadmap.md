# mltk Roadmap — Future Development

*This page is our honest assessment of where mltk stands, what we're missing, and where we're going. Items move from this list to the active sprint plan when resources and priorities align.*

---

## Honest Comparison Table

We believe transparency builds trust. Here's what we have, what we don't, and who does.

### Where mltk LEADS (no competitor matches)

| Capability | mltk | Nearest Competitor | Gap |
|-----------|:----:|-------------------|:---:|
| Behavioral consistency testing | **6 assertions + ParaphraseGenerator** | Nobody | Only provider |
| Multi-method dispatch (lexical→NLI→LLM) | **Unified API** | Promptfoo (partial) | Major |
| Full ML lifecycle coverage | **242 assertions** | Evidently (monitoring) + DeepEval (LLM) | Nobody covers all stages |
| Compliance test frameworks | **8 frameworks** | Giskard (platform certs) | Major |
| Rust acceleration (PyO3) | **Yes** | Nobody | Only provider |
| Training bug detection | **P0-P2 coverage** | Nobody | Only provider |
| Model scanning + test generation | **8 scanners** | Giskard (reports only) | We generate committable pytest files |
| VS Code extension | **Test Inspector** | Nobody (ML testing) | Only provider |
| YAML test definitions | **All assertion types** | Promptfoo (LLM-only) | Full lifecycle |
| ParaphraseGenerator | **Built-in** | Nobody | Only provider |

### Where mltk TRAILS (competitors are ahead)

| Capability | mltk Status | Who Leads | Their Approach | Gap Size | Feasibility |
|-----------|-------------|-----------|---------------|:--------:|:-----------:|
| **Synthetic test data generation** | `SyntheticQAGenerator` (template + optional `llm_fn`); residual gap vs RAGAS KG / Evol-Instruct product depth | RAGAS, DeepEval, Giskard | Evolutionary Q/A generation from documents | Medium | residual polish |
| **Dynamic red teaming** | `RedTeamSession` + 55 catalog payloads + mutations + optional `llm_attacker`; not full adaptive agent parity | Giskard (autonomous agents), Promptfoo (135 plugins) | Multi-turn adversarial attack chains that adapt mid-conversation | Large | residual vs GOAT/plugin scale |
| **MCP evaluation metrics** | **5** MCP-specific assertions shipped (`assert_mcp_*`) | DeepEval (first mover) | MCPUseMetric, MCPTaskCompletionMetric | Small | residual metrics only |
| **Multimodal evaluation** | Image-text alignment only | DeepEval (5 metrics), Kolena | Text-to-image quality, editing, coherence, helpfulness | Medium | 2 sprints |
| **LLM observability** | Basic OTLP export | Arize Phoenix, Langfuse (star counts: re-measure before citing) | Real-time trace visualization, span debugging | Large | Integrate, not build |
| **JSON schema validation** | **Shipped (S96):** `assert_valid_json`, `assert_json_schema`, `assert_pydantic_schema` (regex `assert_output_format` remains complementary) | DeepEval, Promptfoo | JSON Schema, Pydantic model validation | Closed | — |
| **Cost/token tracking** | **Shipped (S96):** `mltk.cost` + `assert_cost_within` / `assert_token_usage` | Promptfoo, Arize Phoenix | Token counts, dollar costs, budget alerts | Closed | — |
| **Data quality (standalone)** | ML-focused assertions | Great Expectations | ExpectAI, data profiling, 300+ expectations | N/A | Different market |

### Neutral Comparisons

| Capability | mltk | Competitors | Notes |
|-----------|:----:|:----------:|-------|
| RAG evaluation | 5 assertions + method dispatch | RAGAS (~10), DeepEval (similar) | Competitive; our method dispatch is unique |
| Agent evaluation | 10 agentic assertions | DeepEval (agent metrics), RAGAS (3) | Competitive |
| LLM safety | Toxicity + hallucination + refusal + taxonomy | Giskard (red team), Promptfoo (vuln scan) | Different approach: we test, they attack |
| Drift detection | 7 univariate + streaming (ADWIN/CUSUM) | Evidently (20+ methods) | We lead on streaming; they lead on variety |
| pytest integration | Native plugin | DeepEval (native), others (none) | Tied with DeepEval |

---

## Future Development Possibilities

These items are researched but **not committed**. Each includes an honest effort estimate and dependency analysis. Items move to active development based on team capacity and user demand.

### Tier 1: Closing Critical Gaps

#### Synthetic Test Data Generation
*Status: Shipped (S76+) — residual depth vs RAGAS/DeepEval*

Auto-generate Q/A evaluation datasets from document corpora. Teams evaluating RAG systems need labeled test data — today they build it manually or use RAGAS/DeepEval.

**What ships today:** `SyntheticQAGenerator` in `mltk.domains.llm.synthetic` with template mode (zero-dep, deterministic) and optional `llm_fn: Callable[[str], str]` (provider-agnostic). Multiple question types (factual, reasoning, multi-hop, counterfactual, out-of-scope, conversational, distracting). Output feeds existing RAG assertions (`assert_faithfulness`, `assert_answer_relevancy`, etc.).

**What competitors still lead on:** RAGAS knowledge-graph evolution (simple → multi-context), DeepEval multi-stage Synthesizer styling, Giskard RAGET component-targeted question types at product depth. mltk residual work is polish/coverage, not a greenfield build.

**Differentiator:** Provider-agnostic `llm_fn` / template base case without locking users into a vendor LLM client.

*Research brief: `docs/research/` — synthetic data generation (9 sources)*

#### Dynamic Red Teaming Framework
*Status: Shipped base (S77+) — residual vs autonomous GOAT / plugin-scale attackers*

Automated adversarial testing with static catalog + multi-turn sessions + encoding mutations. OWASP-oriented categories and YAML-driven suites.

**What ships today:**
- Attack catalog (~**55** static payloads across injection, jailbreak, data extraction, harmful content, agency, system-prompt theft, encoding bypass)
- `RedTeamSession` multi-turn chains with conversation state
- Encoding mutation techniques + `assert_encoding_mutation_resilience`
- Optional `llm_attacker` on `assert_red_team_resilient` (opt-in; base case needs no external API)
- Assertions: `assert_red_team_resilient`, `assert_no_session_jailbreak`, OWASP coverage helpers

**Honest residual gaps (still trails Giskard GOAT / Promptfoo plugin scale):**
1. Fully autonomous adaptive attacker agents (response-conditioned attack graphs at GOAT depth)
2. Plugin-scale payload generation (Promptfoo-class volume)
3. Indirect prompt injection via RAG/tools as a first-class product surface
4. Embedding/vector attacks (OWASP LLM08) as a dedicated suite
5. Mutation automation at Garak-scale (thousands of auto-generated variants)

**Design constraint:** Never require an external API for the base case. Multi-turn and encoding work with `model_fn` alone. LLM-as-attacker is opt-in.

*Research brief: `docs/research/` — dynamic red teaming (18 sources including Garak, PyRIT, DeepTeam, OWASP)*

#### MCP Evaluation Metrics
*Status: Shipped (S75) — 5 assertions; residual depth vs DeepEval metric suite*

Test Model Context Protocol tool use correctness — tool selection, argument schema conformance, resource access, context window utilization, error recovery.

**What ships today** (`mltk.domains.llm.mcp`):
- `assert_mcp_tool_schema_conformance`
- `assert_mcp_tool_selection`
- `assert_mcp_resource_access`
- `assert_mcp_context_window`
- `assert_mcp_error_recovery`
- Supporting types: `McpToolCall`, `McpResourceAccess`, `McpTrace`

**Residual vs DeepEval / full MCP surface:** sampling params, multi-server namespace routing depth, session-completion judges, OAuth scope modeling, and additional first-mover metrics. Generic agentic assertions remain complementary, not a substitute for the MCP-specific set above.

*Research brief: `docs/research/` — MCP evaluation (24 sources)*

#### Smart Dataset Importer / Test-Suite Mapper
*Status: In Progress — S97 + S98 (sprints 1–2 of 3) done: loading, column auto-mapping, task-type classification, suite generation, pytest emission, and the `mltk import` CLI (see [Smart Dataset Importer docs](api/dataset-importer.md))*

One-click onboarding: point mltk at a dataset (HuggingFace Hub or local CSV/Parquet/JSON) plus an optional golden set, and get a runnable mltk evaluation suite — no manual glue code.

**The workflow:** `DatasetImporter` loads and normalizes the source to a common schema (**done, S97**). A column auto-mapper infers field roles (input/prompt, golden/expected, context, label) from name heuristics and dtypes, with a preview the user can override (**done, S97**). A task-type classifier (classification, QA/RAG, summarization, generation, retrieval) generates the matching assertions as both a live suite and a committable pytest file, behind a one-command `mltk import <source>` CLI (**done, S98**, second sprint of the epic). The remaining sprint wires an MCP tool, golden-set binding with an LLM-judge fallback, and the versioned dataset registry (**S99**, third sprint of the epic).

**Differentiator:** competitors require you to hand-write test cases or adopt their dataset format. mltk would turn any public dataset into a runnable eval in one command (`mltk import <source>`) or one MCP call.

**Approach:** HuggingFace `datasets` adapter first (**done**); pluggable adapters for Kaggle/OpenML/local/object-storage later. Golden-set binding maps references to expected outputs; falls back to LLM-judge scorers when no exact golden exists (**S99**).

**Not in scope for this epic:** a URL-fetch adapter (raises `NotImplementedError` for now — download locally first), a SQL/database source, and HuggingFace's `streaming=True` mode (large-dataset streaming is deferred past S99; `DatasetImporter` always fully materializes the source).

**Effort:** 2-3 sprints | **Dependencies:** `datasets`, `pyarrow` (optional `mltk[importer]` extra) | **Priority:** High — lowers time-to-first-eval to near zero

#### Agent Trace Capture (Clearance-Tiered)
*Status: Proposed — new `mltk.trace` package owning a single capture record; the existing trace schemas become adapter targets*

mltk can **grade** an agent trace but cannot **record** one. Every trace-consuming assertion takes a structure the caller assembled by hand, so each real workflow begins with glue code that reconstructs what the model did from raw provider responses.

**Prerequisite finding — the trace surface is already fragmented into three incompatible schemas:**

| Schema | Location | Shape | Consumed by |
|--------|----------|-------|-------------|
| `AgentTrace` / `McpTrace` | `mltk.domains.llm.trace`, `.mcp` | Dataclass: `tool_calls`, `total_tokens`, `total_duration_ms`, `metadata` | 7 agentic assertions + 4 `assert_mcp_*` |
| `SpanTrace` / `Span` / `SpanKind` | `mltk.domains.llm.span` | Span tree, OTel-shaped, `.total_cost_usd` | 4 `assert_span_*` (`span_eval.py`) |
| plain `dict` | `mltk.integrations.trace_quality` | Flat keys: `latency_ms`, `cost_usd`, `score`, `output` | `assert_trace_quality` |

Nothing converts between them; `assert_trace_quality` does not import `AgentTrace` at all. There is therefore no single "existing trace surface" to extend — any capture layer that emits one of these three activates only that slice and leaves the other two untouched.

**The decision (2026-08-08): capture owns its own record type.** A new `mltk.trace` package defines one canonical capture record; the three schemas above become **outputs of an adapter step**, not the thing capture emits natively. This inverts the dependency — `domains/llm → mltk.trace` rather than capture living downstream of assertion code — and makes capture the forcing function that resolves the fragmentation, since it is the only component upstream of every consumer. Extending one existing schema instead would add a fourth de-facto shape under `domains/llm` and cap the concept at the LLM domain, when "record what the model did at a given clearance level" applies to any served inference.

**Clearance-tiered capture.** How much is observable depends on the access level, so capture is tiered and every field records the tier that produced it:

| Tier | Access level | What can be captured |
|------|-------------|----------------------|
| **T0 — closed inference** | Black-box provider API | Request/response pairs, tool-use blocks, tool-call count with names and arguments, total API-call/round-trip count, retries, token usage, stop reason, per-hop latency |
| **T1 — instrumented client** | Wrapping the SDK or agent loop | Nested sub-agent calls, additional/fan-out API calls, per-hop timing, error and retry chains, cache hits, ordered detailed action log |
| **T2 — open inference** | Self-hosted, weights or logits reachable | Logprobs and top-k alternatives, sampling parameters, refusal/guard internals, hidden-state hooks where the runtime exposes them |

**Honesty rule (non-negotiable):** a field the clearance tier could not observe is `None` with its tier recorded — never a zero, never a plausible default. This follows the S102 fairness precedent, where undefined per-group rates propagate as `None` and are excluded rather than coerced to `0.0`. A trace must never imply an observation the access level did not permit.

**Why the tier belongs in the schema, not in metadata.** This rule is the main structural argument for a dedicated record. Under any of the three existing schemas, tier provenance would land in `AgentTrace.metadata: dict[str, Any]` — an untyped catch-all — leaving the honesty guarantee as a convention nothing can enforce, in a codebase whose recent history is largely about eliminating exactly that class of silent default. A capture-owned record carries the tier as a typed field on every captured value. The same applies to T2 data: logprobs, top-k alternatives, and sampling parameters have no home in `AgentTrace`, `SpanTrace`, or the flat dict.

**Integration, unchanged:** adapters project the capture record into `AgentTrace` / `McpTrace`, `SpanTrace`, and the `trace_quality` dict, so all 16 existing trace assertions become reachable rather than only one family's worth. Export still goes through the shipped Phoenix / Langfuse / OTel adapters in `mltk.integrations` — no new backend. Token and cost fields reuse `mltk.cost`.

**Open questions for sprint planning:** provider coverage order; whether capture is a decorator, a context manager, or a client proxy; how T2 hooks degrade when the serving runtime does not expose them; and whether the three existing schemas are eventually deprecated in favor of the capture record or kept indefinitely as projections.

**Effort:** 3-4 sprints (larger than a single-schema extension — the adapter layer and the fragmentation it resolves are the added cost) | **Dependencies:** none for T0/T1 (provider SDKs optional); T2 depends on the serving runtime | **Priority:** High — turns a fixture-only assertion surface into a live one, and is the last natural moment to impose one trace record before a fourth consumer family ships

### Tier 2: Expanding Capabilities

#### Multimodal LLM Evaluation
*Status: Researched*

Test image generation quality, image-text alignment, visual reasoning, image editing accuracy. Needed for GPT-4o, Claude, Gemini multimodal outputs.

**What DeepEval has that we don't:** 4 LLM-as-Judge image metrics (TextToImage, ImageCoherence, ImageHelpfulness, ImageEditing). All use GPT-4o as the judge — no numerical metrics, purely subjective.

**Our approach (2-sprint plan):**
- Sprint A (LLM-judge path, zero new deps): `assert_prompt_faithfulness`, `assert_image_coherence`, `assert_image_helpfulness`, `assert_image_editing_score`, `assert_vqa_accuracy` — all reuse existing `judge_fn` pattern
- Sprint B (numerical path): `assert_clip_score` (embedding-in, zero dep + live CLIP with `open-clip-torch`), `assert_image_hallucination` (NLI, reuses existing dep), `assert_fid_score` (batch quality), `assert_edit_preservation` (SSIM)

**Key research finding:** CLIP score is reliable for regression testing but NOT for nuanced quality assessment (spatial reasoning blind, counting blind, style insensitive). PickScore/HPSv2 correlate 15% better with human preference. LLM-as-Judge is the right primary metric; CLIP is the fast CI gate.

**Effort:** 2 sprints | **Dependencies:** Pillow (Sprint A), open-clip-torch (Sprint B, optional) | **Priority:** Medium — growing fast

*Research brief: `docs/research/` — multimodal evaluation (15+ sources including T2I-CompBench)*

#### LLM Observability Integration
*Status: Researched, build-vs-buy decided*

**Decision: INTEGRATE, not build.** Building our own would cost 5-7 sprints for ~80% parity. Integrating takes 1 sprint for 100% parity, and Phoenix/Langfuse keep improving automatically.

**What's missing from our existing `otel.py`:** No LLM-specific attributes (input.value, output.value, token counts), no OpenInference semantic conventions (Phoenix won't render our spans as LLM traces), gRPC-only (Phoenix prefers HTTP), synchronous `SimpleSpanProcessor` (too slow for production), no eval score push-back to Phoenix/Langfuse.

**Our approach:**
- `PhoenixExporter`: HTTP OTLP export + OpenInference attributes + `BatchSpanProcessor` (~200 lines)
- `LangfuseLogger`: REST API adapter mirroring our existing `MlflowLogger` pattern (~100 lines)
- Attribute enrichment in `judge.py`, `rag.py`, `latency.py`, `agentic.py` (optional kwargs, backward-compat)

**Honest cost comparison:** Build = 5-7 sprints + ongoing maintenance of token pricing tables and trace visualization. Integrate = 1 sprint, Phoenix/Langfuse maintain the UI forever.

**Effort:** 1 sprint | **Dependencies:** arize-phoenix-otel (optional), langfuse (optional) | **Priority:** Medium

*Research brief: `docs/research/` — observability build vs integrate (15 sources including Phoenix, Langfuse, OTel GenAI semconv)*

#### Compliance Drift Auto-Sync
*Status: Proposed — extends the shipped `mltk.compliance` frameworks*

`mltk.compliance` ships EU AI Act, NIST AI RMF, ISO 42001, OWASP LLM, HIPAA, SR 11-7, FDA, and custom frameworks with `find_gaps`, `assert_*_coverage`, and `generate_compliance_report`. All of it is **point-in-time**: it answers "are we covered right now" and nothing watches that answer change.

**Two drift axes, both currently unmonitored:**

1. **Internal drift** — a control you previously covered silently loses coverage: the backing assertion was renamed, deleted, moved behind a skip, or started failing. Today this surfaces only if someone re-runs the report and reads it closely.
2. **External drift** — the framework itself moves underneath the mapping: renumbered articles, new controls, revised guidance. A mapping built against last year's text quietly stops meaning what it claims.

**The proposal:** a stored, versioned coverage baseline plus a re-check that diffs current coverage against it and reports what moved — delivered as a normal pytest gate (`assert_no_compliance_drift`) and a CI job, not as a dashboard.

**Design constraints:** the baseline is a committed, reviewable artifact (a snapshot test, diffable in a PR); drift reporting keeps "control lost coverage" and "framework definition changed" as separate categories and never merges them; and external-drift detection requires a human-confirmed framework version bump — mltk must not silently re-map controls against regulatory text it fetched on its own. An audit artifact that updates itself without review is worse than a stale one.

**Open questions for sprint planning:** baseline storage format and location, whether external drift ships at all in the first cut (it needs a maintained source of framework versions), and whether this reuses the `DatasetRegistry` versioning pattern.

**Effort:** 1-2 sprints for internal drift; external drift is a separate track | **Dependencies:** none for internal drift | **Priority:** Medium-High — compliance claims decay silently, the worst failure mode for an audit artifact

#### Metric Coverage Expansion (External-Library Backed)
*Status: Proposed — 6 named metrics, each an external dependency plus wiring*

Six metrics mltk does not implement today. Each is an optional-extra dependency wrapped in an assertion following the shipped pattern: import guard with an install hint, deterministic base case, `TestResult` with populated `.details`.

| Metric | Measures | Home | Library | Notes |
|--------|----------|------|---------|-------|
| **DER** | Diarization error rate — speaker attribution | `mltk.domains.speech`, joins `assert_wer` | `pyannote.metrics` | Cleanest fit of the six; mature reference implementation |
| **COMET** | Neural MT quality; correlates with human judgment far better than BLEU | `mltk.domains.nlp` | `unbabel-comet` | Heavy — pulls torch plus a model download; must be strictly optional and offline-testable |
| **chrF** | Character n-gram F-score; robust for morphologically rich languages | `mltk.domains.nlp`, joins `assert_bleu` / `assert_rouge` | `sacrebleu` | Lowest effort; `sacrebleu` also provides a canonical BLEU to cross-check `assert_bleu` against |
| **TEDS / GriTS** | Table structure recognition accuracy | New surface — document/table extraction | **No canonical PyPI package** | TEDS reference implementation lives in IBM's PubTabNet repo, GriTS in Microsoft's table-transformer repo. Pip options (`zss`, `easyted`) are generic tree-edit-distance libraries, not the table metric. Needs a vendored implementation over `zss` — highest effort, scope before scheduling |
| **CIDEr / SPICE / METEOR** | Caption and generation quality | `mltk.domains.multimodal` + `mltk.domains.nlp` | `pycocoevalcap` | Bundles all three, but **SPICE requires a Java runtime** — likely ship CIDEr + METEOR and gate SPICE behind an explicit availability check |
| **langdetect** | Output language identification | `mltk.domains.nlp` | `langdetect` (alternatives: `py3langid`, fastText LID) | Enables `assert_output_language` — a real gap for multilingual deployments |

**Sequencing note:** this is not one sprint. chrF and langdetect are near-trivial wiring; DER is contained; CIDEr/METEOR is moderate with a Java caveat; COMET and TEDS/GriTS each deserve their own scoping pass. Ship in that order rather than as a single batch.

**Effort:** 2-4 sprints depending on batching | **Dependencies:** `sacrebleu`, `langdetect`, `pyannote.metrics`, `pycocoevalcap`, `unbabel-comet` — all optional extras; TEDS/GriTS packaging unresolved | **Priority:** Medium — closes named-metric gaps that evaluators ask about by name

#### Meta-Evaluation — Testing the Eval Suite's Own Premises
*Status: Proposed — assertion family, not yet scoped*

Every eval suite rests on premises that are themselves untested: that the dataset is representative of production traffic, that the LLM judge correlates with human preference, that the chosen metric actually separates good outputs from bad ones. When one of those is false, the suite reports confidently and means nothing — a green run on an unrepresentative dataset is worse than no run, because it is trusted.

**Proposed assertions:** `assert_judge_correlates` (judge scores vs. a human-labeled subset, with a minimum correlation floor), `assert_dataset_representative` (distribution comparison between eval set and a production sample — reuses the shipped drift machinery), `assert_metric_discriminates` (does the metric actually separate known-good from known-deliberately-bad fixtures?).

**Why it fits mltk:** the library already treats "the test is wrong" as a first-class failure mode — dataset-quality gates in the importer, calibration validation, the empty-input contract. This extends that discipline from the data to the evaluation apparatus itself. It also composes with existing work rather than adding a new dependency surface: representativeness reuses drift detection, discrimination reuses fixture-based testing.

**Effort:** 1-2 sprints | **Dependencies:** none (reuses drift + judge infrastructure) | **Priority:** Medium — small surface, addresses a failure mode that silently invalidates everything downstream

#### Compliance Evidence from Captured Traces
*Status: Proposed — composition of two other roadmap items, not independently buildable*

Compliance frameworks want evidence that a control was exercised. Test results alone are weak evidence: they record that an assertion passed, not what was actually observable when it ran. A trace captured at a **known clearance tier** is strictly stronger — it records what was tested, at what access level, on what date, with what supporting data.

Combining Agent Trace Capture (Tier 1) with Compliance Drift Auto-Sync (above) produces auditable evidence as a byproduct of ordinary eval runs, rather than as a separate reporting exercise. The clearance tier is what makes it defensible: an auditor can distinguish "we verified this against model internals" from "we inferred it from black-box responses," and the honesty rule guarantees the distinction is recorded rather than assumed.

**Dependency note:** neither parent item delivers this alone, and it should not be scheduled before both have shipped. Listed here so the composition is not rediscovered later as a surprise.

**Effort:** 1 sprint on top of both parents | **Dependencies:** Agent Trace Capture + Compliance Drift Auto-Sync | **Priority:** Medium — highest-value combination of the two, but strictly downstream of them

#### JSON Schema Validation
*Status: Shipped (S96)*

Validate LLM structured outputs against JSON Schema and Pydantic models. `assert_valid_json`, `assert_json_schema` (jsonschema), and `assert_pydantic_schema` (pydantic v1/v2) in `mltk.domains.llm` complement the regex-based `assert_output_format`. XML/SQL validation remains future work.

**Shipped:** S96 | **Dependencies:** jsonschema + pydantic (optional, with install hints)

### Tier 3: Future Vision

#### Autonomous QA Agent ("mltk Agent")
*Status: Researched, architecture designed, Phase F in epic plan*

mltk as the test execution engine for autonomous coding agents. MCP server exposing scan/test/suggest tools. Agents run tests, detect issues, suggest fixes, create PRs.

**Approach:** MCP server (1 sprint) → Fix suggestion engine → Experiment runner → PR generator → Jira linker

**Monetization:** Separate product tier ($99/seat/month) — distinct from open-source core.

**Effort:** 6 sprints | **Priority:** Strategic — transforms mltk from tool to platform

#### Claude Code Skills (Persona-Based)
*Status: Designed, not started*

Role-specific agent behaviors for QA engineers, developers, PMs, and DevOps using mltk:
- `mltk-qa-skill` — scan, test, validate coverage
- `mltk-dev-skill` — TDD, fix failures, generate test suites
- `mltk-pm-skill` — read reports, compliance status, quality trends
- `mltk-devops-skill` — CI/CD integration, webhooks, quality gates

**Includes:**
- `CLAUDE.md` at repo root — project context, conventions, quality gates
- `.mcp.json` — MCP server config for mltk-as-tool
- 6 skills: `mltk-qa-skill`, `mltk-dev-skill`, `mltk-pm-skill`, `mltk-devops-skill`, `mltk-autoresearch`, `mltk-sprint-executor`
- 4 subagents: ml-test-engineer (Opus), ml-test-researcher (Sonnet), ml-test-reviewer (Opus), ml-test-qa (Opus)
- Workflow recipes: "add assertion", "red team scan", "audit docs", "release version"
- Memory seed: project overview, workflow rules, key decisions
- Model routing: Sonnet for research, Opus for code + review
- Hooks: auto-memory, pre-commit quality gates
- Prompt templates for common tasks
- Memory vault: Obsidian bridge (human-browsable) + ShrimPK vault (AI recall) — dual-vault strategy with auto-consolidation hooks

**Dependencies:** F-1 (MCP server mode) for `.mcp.json`

**Effort:** 2 sprints | **Priority:** Strategic — transforms mltk from toolkit to AI-native platform

#### Cost/Token Tracking
*Status: Shipped (S96)*

`mltk.cost` package: `MODEL_PRICING` table (Anthropic + OpenAI, 15 models), `estimate_cost`, runtime-overridable `register_pricing` / `get_pricing`, a `CostTracker` accumulator with per-model breakdown, and `assert_cost_within` / `assert_token_usage` suite-level budget assertions.

**Shipped:** S96 | **Dependencies:** none (stdlib)

---

## How Items Move to Active Development

1. **User demand** — if users/teams request a capability, it moves up
2. **Competitive pressure** — if a gap becomes a deal-breaker, it moves up
3. **Resource availability** — sprint capacity determines how many items are active
4. **Dependency readiness** — some items depend on others (MCP eval → MCP server → Agent)

Want to influence priorities? Open an issue on GitHub or reach out directly.

---

*Last updated: 2026-08-08 — added three proposed items (Agent Trace Capture, Compliance Drift Auto-Sync, Metric Coverage Expansion); previous pass 2026-07-22, TRAILS/body honesty reconciliation after the claim audit (S75/S76/S77/S96 shipped surfaces reconciled with code). Original research briefs remain under `docs/research/`*
