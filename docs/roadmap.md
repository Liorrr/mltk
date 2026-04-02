# mltk Roadmap — Future Development

*This page is our honest assessment of where mltk stands, what we're missing, and where we're going. Items move from this list to the active sprint plan when resources and priorities align.*

---

## Honest Comparison Table

We believe transparency builds trust. Here's what we have, what we don't, and who does.

### Where mltk LEADS (no competitor matches)

| Capability | mltk | Nearest Competitor | Gap |
|-----------|:----:|-------------------|:---:|
| Behavioral consistency testing | **7 assertions** | Nobody | Only provider |
| Multi-method dispatch (lexical→NLI→LLM) | **Unified API** | Promptfoo (partial) | Major |
| Full ML lifecycle coverage | **224 assertions** | Evidently (monitoring) + DeepEval (LLM) | Nobody covers all stages |
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
| **Synthetic test data generation** | Not implemented | RAGAS, DeepEval, Giskard | Evolutionary Q/A generation from documents | Large | 2-3 sprints |
| **Dynamic red teaming** | 50 static payloads | Giskard (autonomous agents), Promptfoo (135 plugins) | Multi-turn adversarial attack chains that adapt mid-conversation | Very Large | 3-4 sprints |
| **MCP evaluation metrics** | Agentic assertions exist, no MCP-specific | DeepEval (first mover) | MCPUseMetric, MCPTaskCompletionMetric | Medium | 1 sprint |
| **Multimodal evaluation** | Image-text alignment only | DeepEval (5 metrics), Kolena | Text-to-image quality, editing, coherence, helpfulness | Medium | 2 sprints |
| **LLM observability** | Basic OTLP export | Arize Phoenix (13K stars), Langfuse (19K stars) | Real-time trace visualization, span debugging | Large | Integrate, not build |
| **JSON schema validation** | Regex-based format check | DeepEval, Promptfoo | JSON Schema, Pydantic model validation | Small | 1 sprint |
| **Cost/token tracking** | Duration only | Promptfoo, Arize Phoenix | Token counts, dollar costs, budget alerts | Small | 1 sprint |
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
*Status: Researched, not started*

Auto-generate Q/A evaluation datasets from document corpora. Teams evaluating RAG systems need labeled test data — today they build it manually or use RAGAS/DeepEval.

**What competitors do:** RAGAS builds a knowledge graph from documents and evolves questions through 4 depth levels (simple → reasoning → conditioning → multi-context). DeepEval's Synthesizer runs a 4-stage pipeline (input → filtration → evolution → styling) with multi-turn conversation support. Giskard's RAGET generates 6 question types each targeting a specific RAG component (retriever, generator, router, rewriter).

**Our approach:** `TestsetGenerator` class with `llm_fn: Callable | None` interface (matches our `judge_fn` pattern). `None` = template-based (zero-dep, deterministic). Provided = LLM-based evolution. Output feeds directly into existing `assert_faithfulness`, `assert_ragas_score`, `assert_answer_relevancy` — no glue code.

**Differentiator:** All three competitors require their own LLM client configuration. mltk would be the only provider-agnostic generator that works with any LLM the user already has.

**Sprint breakdown:**
- Sprint 1: `TestsetGenerator` + template backend (5 question types), `SyntheticTestCase` dataclass
- Sprint 2: LLM backend + Evol-Instruct evolution loop + filtration + CLI `mltk synthesize`
- Sprint 3: Knowledge graph for multi-hop + quality assertions (`assert_testset_coverage`)

**Effort:** 3 sprints | **Dependencies:** None (template) or user LLM (advanced) | **Priority:** High — unblocks RAG evaluation workflows

*Research brief: `docs/research/` — synthetic data generation (9 sources)*

#### Dynamic Red Teaming Framework
*Status: Researched, architecture designed*

Automated adversarial attack generation that goes beyond static payload lists. Multi-turn attack chains, OWASP LLM Top 10 coverage, YAML-driven attack configuration.

**Honest gap:** The gap is **architectural**, not quantitative. Our 50 static payloads are sent in isolation and never adapt. Giskard's GOAT framework reads the target's response and decides what to try next. Promptfoo's 135 plugins are fine-tuned models that generate payloads, not hardcoded strings. PyRIT's Crescendo outperforms static jailbreaks by 29-61% on GPT-4. We are a generation behind.

**5 specific capability gaps:**
1. No multi-turn attacks (no conversation state between payloads)
2. No LLM-as-attacker (no dynamic generation based on target response)
3. No indirect prompt injection (via RAG, tools, external data)
4. No embedding/vector attacks (OWASP LLM08, new in 2025)
5. No encoding mutation automation (8 manual vs Garak's 3,000+ auto-generated)

**Our approach (4-sprint roadmap):**
- Sprint A: `RedTeamSession` (multi-turn state) + encoding mutation engine (8→100+ payloads) + `mltk security-scan` CLI
- Sprint B: `assert_no_indirect_injection` (via RAG/tools) + `assert_embedding_privacy` (OWASP LLM08)
- Sprint C: `AttackerLLM` (optional LLM-as-attacker, falls back to static) + simplified PAIR/Crescendo
- Sprint D: OWASP ASI 2026 agentic coverage (tool injection, privilege escalation)

**Design constraint:** Never require an external API for the base case. Multi-turn and encoding work with `model_fn` alone. LLM-as-attacker is opt-in.

**Effort:** 4 sprints | **Dependencies:** None (base), optional LLM (advanced) | **Priority:** High — enterprise security requirement

*Research brief: `docs/research/` — dynamic red teaming (18 sources including Garak, PyRIT, DeepTeam, OWASP)*

#### MCP Evaluation Metrics
*Status: Researched, spec ready*

Test Model Context Protocol tool use correctness — tool selection accuracy, argument validation, resource access control, context window utilization.

**What we have vs what's needed:** Our `AgentTrace` only captures `tool_calls: list[ToolCall]` — no field for MCP Resources (read-only URIs), Sampling (server-initiated LLM calls), or server namespaces. MCP introduces 8 concepts with no equivalent in generic function-calling: tool namespacing (`server::tool`), Resources, Prompts, Sampling, context window management, multi-server routing, typed tool results, and OAuth scopes.

**Approach:** Extend `AgentTrace` dataclass (additive, backward-compat) + new `mcp.py` module with 8 assertions:
- `assert_mcp_tool_schema_conformance` — JSON Schema validation of tool arguments (biggest gap vs DeepEval)
- `assert_mcp_resource_access` — validate correct URIs accessed (completely unmodeled today)
- `assert_mcp_no_hallucinated_tools` — manifest-aware variant of existing assertion
- `assert_mcp_tool_namespace` — validate `server::tool` routing
- `assert_mcp_session_completion` — LLM-as-Judge wired to existing `judge.py`
- Plus: sampling params, server routing, resource over-fetch assertions

**Effort:** 1 sprint | **Dependencies:** jsonschema (lightweight) | **Priority:** High — agentic AI is mainstream

*Research brief: `docs/research/` — MCP evaluation (24 sources)*

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
*Status: Not started*

Validate LLM structured outputs against JSON Schema, Pydantic models, XML schemas, SQL syntax. Our `assert_output_format` uses regex — competitors use proper schema validation.

**Effort:** 1 sprint | **Dependencies:** jsonschema (lightweight) | **Priority:** Medium

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
*Status: Not started*

Track LLM token usage and dollar costs per assertion run. Budget alerts, cost-per-test-suite reports.

**Effort:** 1 sprint | **Dependencies:** Provider-specific token counting | **Priority:** Low — nice to have

---

## How Items Move to Active Development

1. **User demand** — if users/teams request a capability, it moves up
2. **Competitive pressure** — if a gap becomes a deal-breaker, it moves up
3. **Resource availability** — sprint capacity determines how many items are active
4. **Dependency readiness** — some items depend on others (MCP eval → MCP server → Agent)

Want to influence priorities? Open an issue on GitHub or reach out directly.

---

*Last updated: March 30, 2026 — based on 5 dedicated research briefs with 50+ sources*
