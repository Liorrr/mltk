# MCP Server

Connect any AI coding agent directly to mltk's ML testing toolkit via the
Model Context Protocol. Agents scan datasets, run assertion suites, browse
the assertion library, execute evaluation pipelines, and generate reports —
all through natural-language requests, with no CLI wrappers needed.

**Since:** v0.9.0

**Module:** `mltk.mcp`

**CLI:** `mltk serve --mcp` or `python -m mltk.mcp`

---

## Why an MCP Server for ML Testing?

No other ML testing toolkit exposes its capabilities as MCP tools. Most
frameworks require a separate CLI session, a Jupyter notebook, or a Python
script to run assertions. That breaks AI agent workflows: the agent must
switch contexts, write a script, and parse unstructured output.

The mltk MCP server makes ML testing **agent-native**:

| Without MCP | With MCP |
|-------------|----------|
| Agent writes a Python script to import mltk | Agent calls `mltk_scan` directly |
| Agent reads raw pytest output | Agent receives structured JSON with `status` and next-step hints |
| Agent guesses which assertions apply | Agent calls `mltk_list` to browse by domain |
| Agent re-invokes CLI after each fix | Agent chains `scan → test → report` natively |

This page documents the six tools the server exposes, how to configure it
in Claude Code, Cursor, and VS Code, and the agent workflow patterns that
make ML quality gating fully automated.

!!! note "MCP Protocol"
    The server uses stdio transport over JSON-RPC 2.0, managed by FastMCP.
    No network ports or authentication are required for local use.
    See the [MCP protocol spec research](../research/mcp-protocol-spec.md)
    for wire-format details.

---

## Quick Start

**Step 1 — Install mltk with MCP support:**

```bash
pip install "mltk[mcp]"
```

**Step 2 — Add `.mcp.json` to your project root:**

```json
{
  "mcpServers": {
    "mltk": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mltk.mcp"]
    }
  }
}
```

**Step 3 — Ask your agent to test your ML project:**

```
Scan the dataset at data/train.csv and run the assertion suite
at tests/model_quality.yaml. If any assertions fail, generate
a report.
```

The agent calls `mltk_scan`, `mltk_test`, and `mltk_report` in sequence —
no additional instructions needed.

!!! tip "Zero-install with uvx"
    If you have `uv` installed, use `uvx mltk-mcp` as the command instead
    of `python -m mltk.mcp`. No `pip install` required — `uvx` downloads
    and caches the package on first use.

---

## Setup

### Claude Code

Claude Code supports two scopes for MCP configuration:

**Project scope** (committed to git, shared with the team):

Create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "mltk": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mltk.mcp"],
      "env": {
        "MLTK_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**User scope** (personal, applies to all projects):

```bash
claude mcp add --scope user mltk python -m mltk.mcp
```

Verify the server is loaded:

```bash
claude mcp list
# mltk  stdio  python -m mltk.mcp  ✓ connected
```

!!! note "Tool Search"
    Claude Code 2.1.7+ uses lazy tool loading (Tool Search). The server's
    six tools are indexed at session start but schemas are only fetched when
    Claude determines they are relevant. This reduces context overhead by
    approximately 85% compared to eager loading.

**Recommended: add a workflow hint to your project's `CLAUDE.md`:**

```markdown
## ML Testing with mltk
- Run `mltk_scan` before running tests to detect data and model issues
- Use `mltk_test` to execute YAML assertion suites
- Run `mltk_list` to browse available assertions by domain
- Use `mltk_eval` for end-to-end solver/scorer evaluation pipelines
- Call `mltk_report` to generate a shareable HTML/JSON report
```

This ensures Claude follows the correct workflow without inferring it from
tool descriptions alone.

---

### Cursor

Create `.cursor/mcp.json` in the project root (project scope) or
`~/.cursor/mcp.json` (global scope):

```json
{
  "mcpServers": {
    "mltk": {
      "command": "python",
      "args": ["-m", "mltk.mcp"]
    }
  }
}
```

!!! warning "Cursor tool limit"
    Cursor enforces a hard limit of approximately 40 tools across all
    connected MCP servers. mltk exposes 6 tools, leaving 34 slots for
    other servers. Do not connect more than 5-6 tool-heavy servers
    simultaneously.

Toggle individual tools on or off in **Settings → Tools & MCP**.

!!! warning "Required key"
    The `"mcpServers"` root key is mandatory in Cursor config. If it is
    absent, Cursor silently ignores the file with no error or warning.

---

### VS Code

Create `.vscode/mcp.json` in the project root:

```json
{
  "servers": {
    "mltk": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mltk.mcp"]
    }
  }
}
```

---

### Windows path note

On Windows without WSL, if the MCP client spawns processes through `cmd`,
prefix the command:

```json
{
  "mcpServers": {
    "mltk": {
      "command": "cmd",
      "args": ["/c", "python", "-m", "mltk.mcp"]
    }
  }
}
```

This avoids `ENOENT` spawn errors when the Python executable is not on the
system PATH used by the client process.

---

## Tools Reference

The server exposes six tools. All tools return JSON with a top-level
`status` field and a `suggested_next_step` field when a natural follow-up
action exists.

---

### `mltk_scan`

Scan an ML project directory or specific paths for data quality issues,
model health problems, and configuration gaps. Returns a prioritized list
of findings with severity ratings.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `string` | Yes | Path to the project root, a dataset file, or a model file. Directories are scanned recursively. |
| `scanners` | `string[]` | No | Scanner names to run. Default: all. Values: `"data"`, `"model"`, `"drift"`, `"bias"`, `"security"`, `"schema"`, `"pii"`. |

**Example call:**

```json
{
  "name": "mltk_scan",
  "arguments": {
    "path": "data/train.csv",
    "scanners": ["data", "schema", "pii"]
  }
}
```

**Response schema:**

```json
{
  "status": "pass | warn | fail",
  "summary": "3 issues found: 1 high, 2 medium",
  "findings": [
    {
      "scanner": "data",
      "severity": "high | medium | low",
      "code": "DATA_NULL_RATE_HIGH",
      "message": "Column 'age' has 34.2% null values (threshold: 5%)",
      "path": "data/train.csv",
      "row_count": 10000
    }
  ],
  "scanners_run": ["data", "schema", "pii"],
  "duration_ms": 420,
  "suggested_next_step": "Run mltk_test to assert against findings, or mltk_list to browse assertions."
}
```

!!! tip "Scanner selection"
    Running all scanners on a large dataset can take 5-30 seconds. When you
    only need to check data quality, pass `scanners: ["data", "schema"]` to
    skip model and security analysis.

---

### `mltk_test`

Run an mltk YAML test suite against a model or dataset and return
pass/fail results with per-assertion detail. Suitable for CI gates and
pre-deployment checks.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `suite_path` | `string` | Yes | Path to a `.yaml` test suite file, or a directory (all `*.yaml` files are run). |
| `verbose` | `boolean` | No | Include full assertion output. Default: `false`. Set `true` when debugging failures. |

**Example call:**

```json
{
  "name": "mltk_test",
  "arguments": {
    "suite_path": "tests/model_quality.yaml",
    "verbose": false
  }
}
```

**Response schema:**

```json
{
  "status": "pass | fail | error",
  "summary": "7 passed, 2 failed, 0 errors",
  "passed": 7,
  "failed": 2,
  "errors": 0,
  "duration_ms": 1840,
  "assertions": [
    {
      "name": "assert_no_drift",
      "status": "fail",
      "metric": 0.34,
      "threshold": 0.20,
      "severity": "high",
      "message": "Feature drift score 0.34 exceeds threshold 0.20"
    },
    {
      "name": "assert_accuracy_above",
      "status": "pass",
      "metric": 0.91,
      "threshold": 0.85
    }
  ],
  "suggested_next_step": "2 failed. Call mltk_report to document findings, or mltk_list domain='drift'."
}
```

!!! note "YAML suite format"
    For test suite authoring, see [YAML Tests](yaml-tests.md). The MCP
    tool accepts the same suite format used by the pytest plugin.

---

### `mltk_list`

Browse the mltk assertion library filtered by text or domain. Returns
assertion names, signatures, and usage hints. Use this when you are not
sure which assertion to apply to a given ML problem.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filter_text` | `string` | No | Case-insensitive search across assertion names and descriptions. Example: `"drift"`, `"hallucination"`. |
| `domain` | `string` | No | Domain filter. Values: `"data"`, `"model"`, `"nlp"`, `"cv"`, `"rag"`, `"safety"`, `"drift"`, `"bias"`, `"performance"`, `"behavioral"`. |

**Example call:**

```json
{
  "name": "mltk_list",
  "arguments": {
    "domain": "behavioral",
    "filter_text": "consistency"
  }
}
```

**Response schema:**

```json
{
  "status": "ok",
  "total": 7,
  "assertions": [
    {
      "name": "assert_behavioral_consistency",
      "domain": "behavioral",
      "signature": "assert_behavioral_consistency(model_fn, prompts, threshold=0.9)",
      "description": "Assert semantically equivalent prompts produce consistent outputs.",
      "since": "v0.5.0",
      "example": "assert_behavioral_consistency(model, paraphrase_set, threshold=0.85)"
    }
  ],
  "suggested_next_step": "Add an assertion to a YAML suite and run mltk_test to validate."
}
```

!!! tip "Discovery workflow"
    Use `mltk_list` with no parameters to see the full assertion library
    organized by domain. Combine with `filter_text` to narrow results
    before writing a test suite.

---

### `mltk_eval`

Execute an evaluation pipeline that cleanly separates prompting strategy
(solver) from grading logic (scorer) against a dataset. Returns per-sample
scores and aggregate metrics.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_path` | `string` | Yes | Path to a CSV or JSON dataset. Must have an `input` column; `target` column is optional for reference-based scorers. |
| `scorer` | `string` | Yes | Scorer to apply. Values: `"exact_match"`, `"includes"`, `"pattern"`, `"llm_judge"`. |
| `solver` | `string` | No | Prompting strategy. Values: `"generate"` (direct), `"chain_of_thought"`, `"few_shot"`. Default: `"generate"`. |

**Example call:**

```json
{
  "name": "mltk_eval",
  "arguments": {
    "dataset_path": "data/qa_eval.csv",
    "scorer": "exact_match",
    "solver": "chain_of_thought"
  }
}
```

**Response schema:**

```json
{
  "status": "pass | fail",
  "summary": "ExactMatch accuracy: 0.82 (threshold: 0.80) — PASS",
  "metrics": {
    "ExactMatchScorer/accuracy": 0.82,
    "ExactMatchScorer/passed_samples": 41,
    "ExactMatchScorer/total_samples": 50
  },
  "solver": "chain_of_thought",
  "scorer": "exact_match",
  "dataset_path": "data/qa_eval.csv",
  "duration_ms": 3200,
  "failed_samples": [
    {
      "input": "What year was the model trained?",
      "target": "2024",
      "output": "2023",
      "score": 0.0
    }
  ],
  "suggested_next_step": "Review failed samples. Use mltk_list domain='nlp' for more assertion options."
}
```

!!! note "Eval pipeline architecture"
    For a deeper explanation of how solvers and scorers compose, see
    [Solver/Scorer Evaluation Pipeline](eval-pipeline.md).

---

### `mltk_dataset`

Retrieve metadata, quality metrics, and a sample preview for a named
dataset registered in the mltk dataset registry or discoverable at a given
path.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `string` | Yes | Registry dataset name or an absolute path to a CSV, JSON, or Parquet file. |
| `version` | `string` | No | Version tag for registry datasets. Defaults to the latest version. |

**Example call:**

```json
{
  "name": "mltk_dataset",
  "arguments": {
    "name": "mmlu",
    "version": "1.0.0"
  }
}
```

**Response schema:**

```json
{
  "status": "ok",
  "name": "mmlu",
  "version": "1.0.0",
  "row_count": 14042,
  "columns": ["input", "target", "subject", "split"],
  "quality": {
    "null_rate": 0.0,
    "duplicate_rate": 0.002,
    "schema_valid": true
  },
  "sample_rows": [
    {
      "input": "What is the derivative of sin(x)?",
      "target": "cos(x)",
      "subject": "calculus"
    }
  ],
  "source": "registry",
  "suggested_next_step": "Pass this dataset_path to mltk_eval to run an evaluation pipeline."
}
```

---

### `mltk_report`

Generate a structured test report in HTML or JSON format from raw
assertion results. Suitable for sharing with stakeholders, attaching to
PRs, or storing as CI artifacts.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `title` | `string` | Yes | Report heading. Example: `"Pre-deployment quality check — v2.4.1"`. |
| `description` | `string` | No | Free-text description of what was tested. Included in the report header. |
| `results_json` | `string` | Yes | Serialized JSON results from `mltk_test` or `mltk_eval`. |

**Example call:**

```json
{
  "name": "mltk_report",
  "arguments": {
    "title": "Drift check — train vs production",
    "description": "Weekly regression against the February production baseline.",
    "results_json": "{\"status\": \"fail\", \"passed\": 5, \"failed\": 2, ...}"
  }
}
```

**Response schema:**

```json
{
  "status": "ok",
  "report_path": "mltk-reports/drift-check-2026-04-02.html",
  "format": "html",
  "sections": ["summary", "passed_assertions", "failed_assertions", "metadata"],
  "assertion_count": 7,
  "generated_at": "2026-04-02T14:22:01Z",
  "suggested_next_step": "Open the report in a browser or attach it to your PR as a CI artifact."
}
```

!!! note "Report formats"
    By default, `mltk_report` generates HTML. To request JSON output,
    include `"format": "json"` in the `results_json` payload's `options`
    field. See [Report API](report.md) for the full format spec.

---

## Response Format

Every tool response follows a consistent envelope:

```json
{
  "status": "pass | fail | warn | ok | error",
  "summary": "One-sentence human-readable summary",
  "[domain-specific fields]": "...",
  "duration_ms": 420,
  "suggested_next_step": "Call X with Y to ..."
}
```

### Status values

| Value | Meaning |
|-------|---------|
| `"pass"` | All assertions passed or scan found no issues |
| `"fail"` | One or more assertions failed or findings exceed thresholds |
| `"warn"` | Issues found below failure thresholds — attention recommended |
| `"ok"` | Non-assertion tool completed successfully (list, dataset info, report) |
| `"error"` | Tool execution encountered a runtime error |

### `suggested_next_step`

When a natural follow-up action exists, the response includes a
`suggested_next_step` string. This field is designed to guide agent
chaining — the agent reads it and decides whether to invoke the next tool
without requiring external workflow instructions.

Example chain guided entirely by `suggested_next_step`:

```
mltk_scan → "Run mltk_test with tests/quality.yaml"
mltk_test → "2 assertions failed. Call mltk_report to document findings."
mltk_report → "Open mltk-reports/report.html or attach to PR."
```

### Error responses

Tool execution errors use `isError: true` in the MCP protocol response,
with a structured body:

```json
{
  "status": "error",
  "error": "File not found: data/train.csv — verify the path is correct and readable.",
  "recoverable": true,
  "suggested_action": "Check the path exists relative to the project root, or use an absolute path.",
  "example": "mltk_scan(path='/absolute/path/to/data/train.csv')"
}
```

!!! important "Error design principle"
    Error messages are injected back into the agent's context window. They
    are instructions for recovery, not diagnostics for humans. Stack traces
    and internal paths are never included. Every error that can be retried
    sets `recoverable: true` and includes `suggested_action`.

---

## Agent Workflows

### Workflow 1: Scan → Test → Report

The standard ML quality gate workflow. Use this before deploying a model
or merging a data pipeline change.

```
Agent: "Run a full quality check on the model training pipeline."

1. mltk_scan(path=".")
   → status: "warn", 2 medium findings

2. mltk_test(suite_path="tests/", verbose=false)
   → status: "fail", 5 passed, 2 failed

3. mltk_report(
     title="Pre-merge quality check",
     results_json=<output of step 2>
   )
   → report_path: "mltk-reports/pre-merge-2026-04-02.html"
```

The agent follows this chain naturally because each tool's
`suggested_next_step` points to the next action.

---

### Workflow 2: Discover → Evaluate

Use this when you are unsure which assertions to apply to a new model type.
The agent browses the library first, then runs a targeted evaluation.

```
Agent: "I have a RAG pipeline. What should I test?"

1. mltk_list(domain="rag")
   → 12 assertions found: assert_retrieval_precision,
     assert_answer_faithfulness, assert_citation_coverage, ...

2. mltk_eval(
     dataset_path="data/rag_eval.csv",
     scorer="llm_judge",
     solver="chain_of_thought"
   )
   → metrics: {"LLMJudgeScorer/faithfulness": 0.74}
   → status: "fail" (threshold: 0.80)
```

---

### Workflow 3: Dataset Quality Gate

Use before training to catch data problems early. Combines `mltk_dataset`
for a metadata snapshot with `mltk_scan` for deep quality analysis.

```
Agent: "Check data quality before we kick off training."

1. mltk_dataset(name="data/train.parquet")
   → quality: {null_rate: 0.12, duplicate_rate: 0.03}
   → suggested_next_step: "Run mltk_scan for detailed findings."

2. mltk_scan(path="data/train.parquet", scanners=["data", "schema"])
   → status: "fail"
   → findings: [
       {severity: "high", code: "DATA_NULL_RATE_HIGH",
        message: "Column 'label' has 12% null values"}
     ]
```

---

### Workflow 4: Behavioral Consistency Regression

Use after model updates to confirm outputs remain stable across
semantically equivalent prompts — mltk's first-mover behavioral assertion.

```
Agent: "Check that the model update didn't break behavioral consistency."

1. mltk_list(domain="behavioral")
   → assert_behavioral_consistency, assert_output_stability,
     assert_prompt_sensitivity, assert_semantic_equivalence, ...

2. mltk_test(suite_path="tests/behavioral_regression.yaml")
   → status: "pass", 4 assertions passed

3. mltk_report(title="Behavioral regression — post-update")
   → report_path: "mltk-reports/behavioral-regression.html"
```

---

### Agent Dialog Example

Here is a realistic end-to-end agent session using the MCP server in
Claude Code:

**User:** Scan my project and test it with our standard suite. If anything
fails, generate a report and explain what needs fixing.

**Agent (internal):** I'll start with a scan to identify any issues.

```
> mltk_scan(path=".")
```

```json
{
  "status": "warn",
  "summary": "1 medium finding detected",
  "findings": [{
    "scanner": "data",
    "severity": "medium",
    "code": "DATA_CLASS_IMBALANCE",
    "message": "Class 'negative' has only 8.3% representation (threshold: 15%)"
  }],
  "suggested_next_step": "Run mltk_test to validate assertions against these findings."
}
```

**Agent (internal):** Scan found a class imbalance warning. Now running the
test suite.

```
> mltk_test(suite_path="tests/", verbose=true)
```

```json
{
  "status": "fail",
  "summary": "6 passed, 1 failed",
  "assertions": [
    {
      "name": "assert_class_balance",
      "status": "fail",
      "metric": 0.083,
      "threshold": 0.15,
      "message": "Class imbalance ratio 0.083 below threshold 0.15"
    }
  ],
  "suggested_next_step": "Call mltk_report to document the failure for your team."
}
```

**Agent (internal):** One assertion failed. Generating a report.

```
> mltk_report(
    title="Pre-training data quality check",
    description="Automated scan + test before training run.",
    results_json="..."
  )
```

```json
{
  "status": "ok",
  "report_path": "mltk-reports/pre-training-2026-04-02.html"
}
```

**Agent:** The scan and test are complete. One assertion failed:

- `assert_class_balance` — the negative class is only 8.3% of your
  training data (minimum: 15%). This will likely cause the model to
  under-predict that class.

Recommended fix: oversample the negative class or apply class weights
during training. A report has been saved to
`mltk-reports/pre-training-2026-04-02.html`.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLTK_LOG_LEVEL` | `"WARNING"` | Log level for stderr output. Values: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`. |
| `MLTK_REPORT_DIR` | `"mltk-reports"` | Output directory for `mltk_report`. Created on first use. |
| `MLTK_SCAN_TIMEOUT` | `"60"` | Maximum seconds a single scanner may run before timing out. |
| `MLTK_DATASET_REGISTRY` | `""` | Path to a local dataset registry YAML for named lookups. |

Set these in the `env` block of your `.mcp.json`:

```json
{
  "mcpServers": {
    "mltk": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mltk.mcp"],
      "env": {
        "MLTK_LOG_LEVEL": "INFO",
        "MLTK_REPORT_DIR": "ci-reports"
      }
    }
  }
}
```

### Transport modes

The server defaults to **stdio** for local use. A Streamable HTTP mode is
available for remote deployments:

```bash
# Start as an HTTP server on port 8080
python -m mltk.mcp --transport http --port 8080
```

Remote HTTP config in `.mcp.json`:

```json
{
  "mcpServers": {
    "mltk-remote": {
      "type": "http",
      "url": "http://your-server:8080/mcp"
    }
  }
}
```

!!! note "Transport selection"
    Use stdio for all local development and CI environments. Streamable
    HTTP is intended for team-shared deployments where multiple users
    connect to a single mltk instance. SSE transport is deprecated as of
    MCP spec version 2025-03-26 and is not supported.

### Logging

All server logs go to **stderr**, never stdout. Stdout is the JSON-RPC
transport channel — any writes there corrupt the protocol stream. Log
output from the server is visible in the MCP client's server log panel
(Claude Code: **Output → MCP**; Cursor: **Settings → Tools & MCP**).

---

## Competitor Comparison

No other ML testing toolkit currently exposes an MCP server. The landscape
as of April 2026:

| Toolkit | MCP Server | Agent-native output | `suggested_next_step` |
|---------|-----------|---------------------|-----------------------|
| **mltk** | Yes (v0.9.0) | Yes (JSON) | Yes |
| DeepEval | No | No | No |
| Evidently AI | No | No | No |
| Great Expectations | No | No | No |
| Weights & Biases Weave | No | No | No |
| Ragas | No | No | No |

The absence of MCP support in competing frameworks means agents using those
libraries must write Python scripts, parse unstructured output, and manage
subprocess lifecycles — all of which mltk eliminates by design.

The agent-integration architecture follows best practices documented in
arXiv:2602.14878 ("MCP Tool Descriptions Are Smelly!", Feb 2026): tool
descriptions prioritize the three highest-impact components — purpose,
parameters, and returns — and avoid over-augmentation that degrades agent
performance.

---

## Research Citations

The design of this server is grounded in published MCP research and
production deployment patterns:

| Source | Finding Applied |
|--------|----------------|
| arXiv:2602.14878 — "MCP Tool Descriptions Are Smelly!" (Feb 2026) | Descriptions cover Purpose + Parameters + Returns only. Examples moved to this page. |
| arXiv:2506.01056 — "MCP-Zero: Active Tool Discovery" (Jun 2025) | ML-testing keywords in descriptions support semantic tool retrieval. |
| arXiv:2512.24565 — "MCPAgentBench" (Dec 2025) | Description clarity + flat schemas are primary agent-performance predictors. |
| [Block Engineering Playbook](https://engineering.block.xyz) | JSON over prose; `status` at response root; verb-first tool names. |
| [Alpic AI Engineering Blog](https://alpic.ai) | Error messages as agent instructions: specific + recoverable flag + action. |
| [MCP Spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) | `isError: true` for tool failures; JSON-RPC errors for protocol only. |
| [Claude Code MCP Docs](https://code.claude.com/docs/en/mcp) | `.mcp.json` scope, `--scope user` CLI, Tool Search lazy-loading behavior. |
| [Cursor MCP Docs](https://docs.cursor.com/context/model-context-protocol) | 40-tool hard limit; `"mcpServers"` root key required. |
| `docs/research/mcp-protocol-spec.md` | Wire format, lifecycle, FastMCP SDK, security model. |
| `docs/research/mcp-server-implementations.md` | FastMCP patterns, `uvx` packaging, stdio logging rules. |
| `docs/research/agent-mcp-usage-patterns.md` | `suggested_next_step` pattern, chaining, Cursor tool budget. |

---

## Related Pages

- [CLI Reference](cli.md) — `mltk serve --mcp` and other CLI flags
- [YAML Tests](yaml-tests.md) — authoring suites consumed by `mltk_test`
- [Solver/Scorer Evaluation Pipeline](eval-pipeline.md) — the pipeline
  behind `mltk_eval`
- [Report API](report.md) — full report format options
- [MCP Evaluation Research](mcp-evaluation.md) — how to test MCP servers
  using mltk assertions
- [Scan API](scan.md) — scanner documentation referenced by `mltk_scan`
- [Assertion Index](assertion-index.md) — full assertion catalog browsable
  via `mltk_list`
