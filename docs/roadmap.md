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
| Full ML lifecycle coverage | **241 assertions** | Evidently (monitoring) + DeepEval (LLM) | Nobody covers all stages |
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

*Last updated: 2026-07-22 — TRAILS/body honesty pass after claim audit (S75/S76/S77/S96 shipped surfaces reconciled with code); original research briefs remain under `docs/research/`*
