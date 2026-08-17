# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`on_empty` policy parameter for empty-input gates** (adversarial-review batch 1) — ~25 assertions across `domains/llm/{rag,retrieval,conversation}`, `domains/recommendation`, `domains/codegen`, and `data/synthetic` gain `on_empty="fail" | "skip" | "pass"`. `"skip"` passes at INFO severity with `skipped: true` details; `"pass"` preserves the legacy behavior.
- **`MltkConfig.explicit_fields` provenance** — config records which fields the user actually set (yaml/pyproject/env), so consumers can distinguish "user configured X" from dataclass defaults. Env overrides are re-validated at load: `MLTK_DRIFT_THRESHOLD=2` now fails fast instead of producing an invalid config that breaks downstream.
- **Storage concurrency regression test** — 8 threads × 50 iterations of mixed reads/writes; verified to fail against the pre-fix shared-connection code before the fix landed.
- **Rust↔Python parity test suite** (`tests/test_rust/test_parity.py`) — all 10 dual-engine functions in `mltk._rust` (ks_test, psi, kl/js divergence, chi-squared, wasserstein, cosine, centroid distance, bertscore, scan_pii_fast) now run identical inputs through both engines and assert agreement. Statistics compared to 1e-9/1e-12; KS/chi² p-values loosely (the engines use different, bounded tail approximations). Previously only importability was tested, so a Rust/Python divergence silently changed assertion outcomes between machines with and without the compiled extension.
- **Shared empty-input + length-validation contracts** (adversarial-review batch 2) — `mltk.core.empty` (`ON_EMPTY_OPTIONS`, `empty_input_result`, `unknown_on_empty_result`) and `mltk.core.validation` (`require_same_length`) deduplicate six drifted per-module copies of the `on_empty` helpers (`data/synthetic`, `domains/codegen`, `domains/llm/{conversation,rag,retrieval}`, `domains/recommendation`) into one place. The unified `severity=Severity.CRITICAL` default reproduces both the parameterized callers (synthetic/rag) and the hardcoded-CRITICAL callers exactly, with zero call-site churn.
- **`seed` parameter on `assert_no_embedding_drift`** — MMD subsampling (both the median-heuristic bandwidth sample and the RBF-kernel sample) now draws a deterministic seeded random subset instead of the first N rows, removing order bias. Defaults to `seed=0`.
- **Asana and Linear issue trackers in `mltk_create_issue`** — the MCP tool's tracker allowlist now accepts `"asana"` and `"linear"` alongside `"github"`/`"jira"`; the adapters shipped but were unreachable. Config keys: Asana `{token, workspace_gid}`, Linear `{api_key, team_id}`.

### Changed
- **BREAKING: MCP `mltk_eval` refuses unknown `solver`/`scorer` instead of silently defaulting** (issue #17) — an unrecognized name resolved via `map.get(key, DEFAULT)` to `GenerateSolver`/`ExactMatchScorer` while the response echoed the *requested* name back, so `scorer="bleu"` returned ExactMatch numbers labelled `"scorer": "bleu"`. This matters most for MCP, where the caller is an agent that reports the metric onward as though the requested scorer ran. Unknown names now return a recoverable error listing the supported values, matching how `model_mode` already behaves. `scorer="llm_judge"` — which `docs/api/mcp-server.md` advertised as valid, including a workflow example showing `LLMJudgeScorer/faithfulness` metrics that were never produced — refuses with its own reason: `LLMJudgeScorer` takes a mandatory `judge_fn` callable and every MCP parameter is a string, so it is unreachable from this surface; the docs now say so and show the Python `EvalTask` equivalent. Blank/omitted values still resolve to the documented defaults. Only the second half of issue #17 remains open (an AST scanner for the general silent-fallback shape).
- **BREAKING: empty input now fails quality-gate assertions by default** — an empty LLM answer no longer scores faithfulness 1.0, zero queries no longer pass retrieval gates, and empty recommendation/conversation/codegen/synthetic inputs no longer pass. Restore old behavior per call with `on_empty="pass"`. Retrieval metrics additionally exclude zero-IDCG queries (no relevant docs) from nDCG/MRR/Recall/MAP means instead of scoring them 1.0.
- **BREAKING: MCP `mltk_scan` reports honestly** — Python file/directory targets return `scan_performed: false` with an explicit static-listing message (the tool never ran scanners on .py targets; it previously implied it had). JSON report targets return `scan_performed: true`. The dead, never-used `ScanEngine` construction is gone.
- **Assertion discovery is recursive and covers container/cost/eval** — `mltk list` now finds assertions in nested subpackages (`domains.llm.behavioral`: paraphrase/format invariance, semantic equivalence, output stability, retrieval consistency, directional expectation) and in `mltk.container`, `mltk.cost`, `mltk.eval` — 229 discoverable, up from 218. `mltk.server` and `mltk.integrations` stay excluded by design (heavy optional deps).
- **Documented env/yaml config fields are now consumed** — `drift_method`/`drift_threshold` (assert_no_drift defaults), `report_dir` (pytest-plugin HTML report), and `pii_patterns` (assert_no_pii default categories) apply when — and only when — the user actually set them; unset config leaves legacy defaults untouched (per-method drift threshold table, all PII patterns). (Batch 2 update: `report_format`/`baseline_dir` are now deleted as dead fields and `seed` is wired to embedding-drift MMD sampling.)

- **BREAKING (scale): `assert_no_drift(method="js")` now reports normalized JS in [0, 1]** — `_drift_js` delegates to `mltk._rust.js_divergence` (Rust-accelerated, ln-2-normalized) instead of its own unnormalized-nats histogram math, ending the two-scales-for-one-concept split flagged in the round-2 audit. Values are ~1.4427× larger than before: **custom `js` thresholds calibrated against the old nats scale must be multiplied by 1.4427** (the 0.1 default is unchanged and now reads on the documented [0, 1] scale). Also swaps the histogram floor from 1e-10 to `_rust`'s 1e-6. A regression test pins the reported statistic to `mltk._rust.js_divergence` exactly.
- **Test-honesty audit (round-2 Phase C)** — AST scan of all 4,788 tests for assertless bodies, tautological assertions (`assert True`, `assert len(x) >= 0`), assertions swallowed by try/except, and unconditional skips: zero tautologies, zero swallowed, zero dead skips (all 26 runtime skips are optional-dependency-conditional and exercised in CI). Of 9 assertless tests, 8 are documented no-crash contracts; the one weak test (`test_card_default_none` verified nothing) now pins the actual auto-generated-card contract.
- **`assert_no_drift` KL and Wasserstein paths route through `mltk._rust`** — `_drift_kl`'s inline histogram math was byte-identical to `mltk._rust.kl_divergence` and now delegates to it (Rust acceleration, zero behavior change); `_drift_wasserstein` imported scipy directly and hard-failed without it, now uses `mltk._rust.wasserstein` (Rust → scipy → numpy fallback chain). JS initially stayed inline to avoid a silent threshold rescale — superseded by the explicit BREAKING entry above.
- **`scan_pii` now uses the Rust engine when built** — the Rust-accelerated `scan_pii_fast` existed since the Rust extension landed but was only ever called by benchmarks; `scan_pii` (and everything above it: `assert_no_pii`, the PII scanner, hybrid NER mode's regex half) now dispatches to it with a transparent pure-Python fallback. Output is byte-identical across engines (parity-tested); checksum validators and allowlists still run in Python.
- **Multimodal judge helpers deduplicated** — `_parse_score` (3 copies) and `_cosine_similarity` (2 diverged copies) across `alignment`/`vlm`/`metrics` consolidated into `mltk.domains.multimodal._scoring`; the surviving `_cosine_similarity` is the `.ravel()` variant, so row vectors and 1-D arrays both work everywhere.
- **ADWIN drops dead variance tracking** (`monitor/streaming_drift`) — window-level Welford variance and bucket-variance merge math were maintained on every observation but never read by the mean-based Hoeffding bound; removed (detection behavior unchanged). Revert this commit's hunk if a variance-based ADWIN2 bound is ever implemented.
- **BREAKING: paired-sequence assertions reject length-mismatched inputs** (adversarial-review batch 2) — ~20 sites (cv detection/tracking, llm judge/long_context/similarity/retrieval, nlp generation/ner, recommendation, training distributed/numerical, testing golden) used `zip(strict=False)`, silently truncating to the shorter input and reporting a metric computed on partial data with no signal to the caller. Mismatched lengths are now a usage error: `require_same_length` raises `ValueError` naming each sequence and its length, at function entry. Genuine fixed-length internal invariants (`data/pii` NHS checksum, guaranteed 10 digits) use `zip(strict=True)` instead. Callers that previously received a truncated result or a failed `TestResult` — `assert_llm_judge_score`/`assert_llm_judge_pairwise` and `assert_gradient_sync` on mismatched lengths — now raise `ValueError`.
- **Fairness metrics no longer coerce undefined per-group rates to 0.0** (`model/bias.py`) — a group with no positives (undefined TPR/PPV) or no negatives (undefined FPR) was scored 0.0, inflating the max-across-groups disparity into false bias alarms. `_group_rates` now returns `None` for undefined rates; `equalized_odds`, `predictive_parity`, and `equal_opportunity` exclude undefined groups from the disparity (surfacing them under `undefined_groups`/`excluded_groups`) and return an INFO not-applicable result when fewer than two groups have the rate defined. `demographic_parity`/`disparate_impact` (selection-rate based, always defined when the group exists) are unchanged.
- **Scan-engine model probe is timeout-bounded** (`scan/engine.py`) — `_detect_model_type` ran `model_fn` on a sample with no time limit, *before* the per-scanner timeout took effect, so a hanging model blocked the entire scan. The probe now runs under a bounded timeout (`per_scanner_timeout`, default 5s) and returns `"unknown"` on timeout. Timed-out scanner daemon threads still run best-effort afterward — documented at the timeout site (Python has no safe thread kill).
- **Webhook delivery no longer blocks the async run-submit route** (`server/routes.py::submit_run`) — webhooks were sent with a synchronous `send_webhook` call in a loop on the event loop; each send is now scheduled via FastAPI `BackgroundTasks`, preserving the best-effort semantics without blocking the response.

### Fixed
- **Experiment sandbox validated importability, not fixes** (`experiment/sandbox.py`) — the generated subprocess script emitted `passed: True` whenever the fix snippet imported cleanly, so any syntactically valid no-op ranked as a winning fix and could drive PR creation. The script now re-runs the original assertion (imported by module + qualname with the finding's recorded args/kwargs); findings whose assertion can't be replayed produce an explicit `sandbox.unsupported` failure instead of a fabricated pass.
- **`assert_calibration` accepted impossible probabilities** (`model/slicing.py`) — values outside [0, 1] (logits, percentages) fell into no ECE bin, yielding ECE 0.0 and a "well-calibrated" pass; NaN and non-binary `y_true` were also unvalidated. All three now fail with explicit messages before either ECE method runs; the calibration scanner inherits the fix.
- **Server storage raced under concurrent requests** (`server/storage.py`) — one shared SQLite connection (`check_same_thread=False`) with per-call `row_factory` toggling and no lock; concurrent reads could interleave, fetch tuple rows, and crash on `dict(row)`. Row factory is now set once at connect and every connection touch is serialized behind a lock.

- **Pure-numpy Wasserstein fallback was off by one interval** (`mltk._rust.wasserstein`) — the scipy-less CDF integral weighted each inter-value gap by the CDF difference at the *right* endpoint (where both CDFs have already jumped) instead of the left, under-reporting distance and scoring completely disjoint point masses `0.0` (true W1 = 1.0) — a silent false "no drift" on installs with neither the compiled extension nor scipy. Unreachable before this PR (`_drift_wasserstein` hard-required scipy); exposed by the delegation above and caught in the round-2 review cycle. The parity suite now has a scipy-blocked leg (`TestWassersteinNumpyFallback`) so the numpy path is actually exercised.
- **SmoothECE reflected kernel missed the boundary-1 mirror image** (`model/slicing.py::_reflected_gaussian_kernel`) — the four-term kernel had the reflection at 0 (`p+q`) but used period-2 translations instead of the boundary-1 mirror (`p+q−2`), losing up to **half the kernel mass** for probabilities near 1 (mass 0.50 at q=1.0 instead of 1.0) — i.e. biased `smooth_ece` exactly where confident classifiers concentrate (measured ~5% relative error on an overconfident model). Now the full |k|≤1 method-of-images truncation, verified to 9 decimals against the untruncated series; smECE is now invariant under the label-flip mirror (f→1−f, y→1−y), with regression tests for both properties.
- **`assert_no_drift(method="auto")` silently discarded a user-supplied `threshold`** — the auto path passed hardcoded defaults to the selected method. A custom threshold now applies to whichever method auto picks.
- **Rust `ks_test` inflated the KS statistic on tied values** — the two-pointer ECDF merge measured the CDF gap mid-jump whenever a value appeared in both samples, adding up to 1/n per tie: *identical* samples scored D=1/n instead of 0, and heavily tied data (integer/categorical-coded features) scored wildly high (e.g. 0.70 where scipy gives 0.20) → false drift positives. Both pointers now advance past each shared value before the gap is measured; the pure-Python/scipy path was always correct, so this was also a Rust-vs-Python divergence. The Rust unit test that asserted `stat < 0.3` for identical data (loose enough to hide the bug) now asserts exact zero, plus two tie-regression tests. Found by the new parity suite on its first run.
- **LLM-judge JSON score parsing** (`mltk.domains.llm.judge`) — `_parse_score` grabbed the first number in the raw response, so a judge returning `{"reasoning_steps": 3, "score": 8}` was scored 3, not 8. Now honors a JSON `"score"` key first (matching the multimodal parsers), with the first-number fallback for prose responses; an explicit but unconvertible score (`null`, `"N/A"`) returns None rather than scraping a number from the raw JSON.
- **Scan findings lost `name`/`message` through the create-PR/issue chain** (adversarial-review batch 2) — `ScanReport.to_json()` emits flat finding fields, but the MCP `_create_pr`/`_create_issue` consumers read a nested `result` object, so the advertised scan → ticket round-trip silently dropped each finding's identity. `_finding_from_json` now accepts the flat shape (falling back from the nested `result` key when it is absent) and carries `details`/`duration_ms` through; a round-trip test builds a real `ScanReport`, serializes it, and asserts name/message survive into the consumer.
- **Overfit scanner broadcast shape-mismatched predictions** (`scan/scanners/overfit.py::_accuracy`) — mismatched `y_true`/`y_pred` shapes silently numpy-broadcast into a bogus accuracy instead of erroring; now raises `ValueError` naming both shapes before the comparison.
- **`embedding_drift` and `bertscore` swallowed every exception on the accelerated path** — `except (ImportError, Exception)` treated any failure (not just a missing Rust/native backend) as a silent fallback, masking genuine computation bugs; narrowed to `except ImportError` so real errors propagate.

### Removed
- **Dead code from the S100 audit** — `scan/codegen._indent` (never called since introduction), `multimodal/_image._validate_image_pillow` (never wired into any assertion), the unused `slice_col` parameter of `SliceScanner._gen_fix`, and the private `_RED_TEAM_CATALOG`/`_generate_mutations` back-compat shims in `cli/security_scan` (only mltk's own tests imported them; tests migrated to the canonical catalog + `mutate_payloads`).

## [0.13.0] — 2026-07-07

**S100 milestone release.** First tagged release since v0.12.7, bundling sprints S95–S99: the complete three-sprint **Smart Dataset Importer** epic (`mltk.importer` — HuggingFace Hub / CSV / JSON / Parquet → deterministic column-role mapping → committable pytest scaffold, with golden-set binding, an LLM-judge fallback, quality-gated registry save, and the `mltk_import` MCP tool #13), nine net-new assertions (First-Mover S95 + Structured Output & Cost Tracking S96), and the repo-wide CI bring-up (first fully-green CI, Python 3.10 support, `mlspec` dist-name resolution). No new required dependencies — `import mltk` stays numpy/pandas-only.

### Fixed
- **RUSTSEC-2026-0204 — `crossbeam-epoch` 0.9.18 → 0.9.20** (transitive via `crossbeam-deque`) — clears an invalid-pointer-dereference advisory in the `fmt::Pointer` impl for `Atomic`/`Shared`; `cargo audit` clean. Advisory published 2026-07-06; caught by the CI security gate on this release PR. Lockfile also rolled `mltk-rust` 0.12.4 → 0.13.0 to match the version bump.
- **`assert_pipeline_stages_compatible` dtype canonicalization** — dtype comparison was exact-string match, so equivalent spellings (`int64` vs `Int64` vs `int`, `float64` vs `float`) produced false FAILs. Now canonicalized via an alias map (platform-deterministic, checked before numpy) + `np.dtype(x).name` + lowercase fallback for pandas extension dtypes; mismatch reports keep the original spellings. Closes an S95 Opus-review deferred P2 item. First change implemented via the codex-worker dispatch loop.

### Added

#### First-Mover Assertions (S95)
- **`assert_no_unicode_attacks(text, *, checks)`** (`mltk.domains.llm`) — detects zero-width/invisible characters (Unicode category `Cf` + explicit set), bidirectional override controls (Trojan Source / CVE-2021-42574), and mixed-script homoglyph tokens (Latin + Cyrillic/Greek). Ships with `detect_unicode_attacks()` raw detector. Legitimate Arabic/Syriac/Kaithi `Cf` format marks are allowlisted to avoid false positives on real RTL text. 28 tests.
- **`assert_pipeline_stages_compatible(stages, *, check_dtypes)`** (`mltk.pipeline`) — validates inter-stage schema flow via `StageSpec(name, produces, requires)`; a stage may consume any column produced by any upstream stage. Distinct from single-dataset data contracts. 17 tests.
- **`assert_pipeline_resilient(pipeline_fn, baseline_input, *, faults, max_failure_rate)`** (`mltk.pipeline`) — ML chaos engineering: injects a 7-fault catalog (`null_injection`, `empty_input`, `dropped_column`, `dtype_corruption`, `scale_shift`, `duplicate_rows`, `single_row`) and asserts graceful degradation within a crash-rate budget. Never mutates `baseline_input`. Ships with `apply_fault()` + `DEFAULT_FAULTS`. 22 tests.
- **`assert_combinatorial_coverage(test_cases, parameters, *, strength, min_coverage)`** (`mltk.testing`) — NIST-style t-way (pairwise default) covering-array coverage measurement over a parameter space. Ships with `combinatorial_coverage()` helper. Missing test-case keys are distinguished from explicit `None` values (sentinel), duplicate parameter values are deduped, and empty value lists raise `ValueError`. 21 tests.
- 88 new tests (4379 total), incl. 8 regression tests from Opus review fixes; all four are pure stdlib/numpy/pandas — no new dependencies.

#### CI bring-up & repo-wide fixes (S97 review follow-through)
- **Fixed the two historical leakage-scanner test failures** — `LeakageScanner` indexed `details["leaky_features"]` as a list, but `assert_no_target_leakage` produces a `dict[str, float]` (`leaky[0]` → `KeyError: 0`); it also read a nonexistent `"correlation"` detail key. The suite is now fully green (no "known failures" caveat).
- **CI actually runs now, and is fully green (15/15 jobs, first time ever)** — the workflow triggered on `main` but the default branch is `master`, so no CI job had ever executed. Also: test matrix installs the `sklearn`/`importer`/`cli`/`server`/`report`/`multimodal` extras (their tests hard-fail rather than skip when deps are absent), `fail-fast` disabled, the never-adopted `ruff format --check` step removed (style enforcement is `ruff check`, per convention), and the uncalibrated coverage gate dropped pending calibration.
- **Python 3.10 support actually works** — `monitor/{aws,azure,gcp}.py` used `datetime.UTC` (3.11+ only; every cloud latency/error-rate assertion raised `AttributeError` on 3.10), and two test sites orphaned modules from `sys.modules` (the mcp conftest's blanket `patch.dict` snapshot-restore and a bare `sys.modules.pop`), which broke Python 3.10's `unittest.mock` dotted-target resolution (it resolves via parent-package attributes; 3.11+ uses importlib) so patches silently no-opped. The 3.10 matrix had never run before.
- **`mltk.__version__`/`mltk.server` work on fresh installs** — version metadata lookups hardcoded the `mltk` dist name, but the distribution is `mlspec` (PyPI squatting workaround): `mltk.server` was unimportable on any clean install and `__version__` fell back to a stale literal. Now resolves `mltk` → `mlspec` → source-tree fallback, single-sourced from `mltk.__version__`.
- **Test-suite honesty fixes** — the report "with plotly" test asserted a plotly embed the generator never emits (charts are dependency-free inline SVG; nothing in `src/` imports plotly) and had only ever skipped; a footer test hardcoded the authoring machine's stale dist version; the CLI `--help` test now neutralizes rich's CI-detection styling; a "nondeterministic" helper now uses a counter instead of a clock (timer-resolution flake).
- **`ruff check src/ tests/` clean repo-wide** — 152 pre-existing errors fixed across 45 files (57× `from __future__ import annotations` placed before module docstrings, unsorted/unused imports, `zip(strict=False)`, lambda assignments, etc.). Missing `Span`/`SpanKind`/`SpanTrace` + 4 `assert_span_*` re-exports added to `mltk.domains.llm.__all__`.
- **pyo3 0.28 → 0.29** (RUSTSEC-2026-0177); `extension-module` is now injected only at maturin wheel-build time so `cargo test`/`clippy` link libpython normally on all platforms. rustfmt drift fixed.
- **`scripts/bump.py verify --skip-test-count`** — CI's docs-freshness job verifies only environment-independent AST-based counts; the collected-test count stays owned by the local pre-commit hook.

#### Smart Dataset Importer — MCP tool, golden-set binding, registry integration (S99)
- **`mltk_import` MCP tool (tool #13)** (`mltk.mcp.server`) — exposes the whole import pipeline to AI agents. **Return-only by default**: returns `mapping_preview` + `task_type` + `generated_code` (the pytest scaffold as a string) and writes a file only when `output_path` is set (MCP tools avoid surprise filesystem side effects). Supports `golden_path`/`golden_target_column`/`golden_key`/`golden_key_column`/`judge` for golden binding and `register` for quality-gated registry save; response carries `golden_binding`/`registration` summaries and a `workflow_hint` → `mltk_dataset`/`mltk_eval`/`mltk_test`. Lazy imports inside the closure (patched at source per the MCP test rule); added to `_WORKFLOW_HINTS`; `EXPECTED_TOOLS` and the registered-tool-count tests bumped 12 → 13.
- **`bind_golden(dataset, golden, *, target_column, key=None, golden_key=None)`** (`mltk.importer.golden`) — binds a user-provided golden/reference file (`load_golden()` reads CSV/TSV/JSON/JSONL/Parquet → `list[dict]`) onto an imported `EvalDataset`, returning a **new** dataset (input never mutated; card/provenance preserved, fingerprint recomputed) plus a `GoldenBindingReport` (matched/unmatched/`match_rate`). Join by an explicit key column (`key="input"` or a metadata field, matched against `golden_key`) or by row order when no key is given. A golden value overrides a pre-existing target; when the golden file has none, an existing target is kept. Every sample is stamped `metadata["scoring"]` = `"exact"` (has a reference) or `"judge"` (none). Binding never scores at import time — it only selects the per-sample Tier-2 scorer.
- **Golden judge fallback in generated tests** — `generate_pytest(..., golden_spec=GoldenSpec(...))` (and CLI `--judge`) emits a golden-aware `dataset` fixture (re-binds the golden file at test time), a `judge_fn` fixture (skips until `MLTK_JUDGE_FN=module:callable`, mirroring `predict_fn`), and a `test_judge_scored_samples` Tier-2 test that scores `metadata["scoring"] == "judge"` samples reference-free via `assert_llm_judge_score`. The importer's golden judge contract is `assert_llm_judge_score`'s `judge_fn: (prompt) -> float` (3–5 scale) — deliberately kept in a separate lane from `build_suite`'s RAG `(a, b) -> float` dataset-side shape, so the importer never leaks two incompatible `judge_fn` shapes.
- **`register_dataset(dataset, *, registry_dir=None, overwrite=False, ...thresholds)`** (`mltk.importer.registry`) — runs a **blocking** `assert_dataset_quality` gate (catches the CRITICAL-severity raise) and saves to `DatasetRegistry` only if it passes, returning a `RegistrationResult` (`saved`/`quality_passed`/`quality_detail`/`reason`/`path`). Gate defaults are import-oriented and lenient on shape (`min_samples=1`, `min_target_coverage=0.0` — small/unlabeled eval sets are legitimate); the load-bearing guard is `max_duplicate_rate=0.5`. Provenance in `DatasetCard.source` preserved; existing `name/version` refused unless `overwrite=True` (non-destructive default).
- **`mltk import` CLI** gains `--golden`/`--golden-target-column`/`--golden-key`/`--golden-key-column`/`--judge` (bind + judge fallback) and `--register` (quality-gated save; exits non-zero if the gate blocks). Emit still always happens (unless `--no-emit`); registration runs after.
- Facade re-exports: `load_golden`, `bind_golden`, `GoldenSpec`, `GoldenBindingReport`, `register_dataset`, `RegistrationResult` from `mltk.importer`.
- `load_golden` reads delimited golden files with `utf-8-sig`, so a spreadsheet-exported BOM is stripped instead of corrupting the first header (which would break a key/target join on that column).
- 45 new tests (4737 total): golden module + codegen emission (incl. a subprocess acceptance gate — an emitted golden+judge scaffold runs `3 passed, 2 skipped`), registry integration (tmp-dir only, never `~/.mltk/`), CLI golden/register flags with a blocked-gate path, and the `mltk_import` MCP tool (registers behind `MLTK_DATASET_DIR`). Zero network; no new dependencies. **Closes the 3-sprint Smart Dataset Importer epic.**

#### Smart Dataset Importer — classifier, suite generator, pytest emitter, CLI (S98)
- **`classify_task(mapping) -> TaskType`** (`mltk.importer.classify`) — deterministic, role-presence-based task-type classification into a 5-type taxonomy (`classification` / `qa_rag` / `summarization` / `generation` / `retrieval`): CONTEXT+GOLDEN → QA-RAG, CONTEXT-only → retrieval, LABEL without GOLDEN → classification, summary-named GOLDEN (whole-token match: `summary_text` yes, `summarylike` no) → summarization, otherwise generation. The `dataset` argument is reserved for future data-peek heuristics.
- **`build_suite(eval_dataset, mapping, task_type, *, judge_fn=None) -> MltkSuite`** (`mltk.importer.suite_gen`) — two-tier semantics: Tier 1 always adds `assert_dataset_quality` with **baseline thresholds** computed from the dataset itself via `compute_baseline_thresholds()` (sample count; coverage floored / duplicate rate ceiled to readable 0.05 buckets — "passes today, gates regressions" starting points in the Great Expectations profiler tradition); Tier 2 judge-scored dataset-side checks (golden-answer faithfulness / answer relevancy / context relevancy for QA-RAG, context relevancy for retrieval) are added only when a `judge_fn` is provided. Samples missing a target/context are skipped for the checks that need them — never invented.
- **`generate_pytest(import_result, task_type, *, dataset_name=None, output_path=None)`** (`mltk.importer.codegen`) — emits a self-contained, committable, `ast.parse`-gated pytest scaffold: `TestDataSanity` (schema snapshot; no-nulls on INPUT/GOLDEN, honestly omitted with a comment when the imported data already has missing cells there; dataset-quality baselines marked `# baseline from import — tighten as needed`) runs immediately, while `TestModelQuality` (task-type assertions looped over samples through a `predict_fn` fixture) stays skipped until a model is wired — `MLTK_PREDICT_FN=module:callable` or a one-fixture edit un-skips everything at once (mirrors the `scan/codegen.py` model-fixture pattern). Assertions needing data an import can't provide (`assert_no_bias` protected attributes, `assert_json_schema` schemas) are emitted as commented lines with `requires ...` notes, not un-runnable tests. Output is byte-deterministic (no timestamps); a fresh emission runs green out of the box (Tier 1 passes, Tier 2 skips). Deliberately does NOT reuse `scan/codegen.py` (joblib-model+dataframe-shaped fixtures — verified mismatch); mirrors its structure only.
- **`mltk import <source>` CLI** (`mltk.cli.importer`) — one command: load → mapping preview → validation warnings → task type → in-memory suite summary → pytest file (emit-by-default to `./test_<stem>_<task_type>.py`). `--split`/`--input-column`/`--target-column` forward to the loader; `--name`, `--output`, `--no-emit`; `--force` required to overwrite an existing output file (it is a user-edited scaffold, never clobbered silently). Exits non-zero with the validation problems when no column maps to INPUT. Lazy imports keep `mltk.cli.app` importable without the `importer` extra, and `import mltk` still never loads `mltk.importer`/`datasets` (subprocess-guarded in tests).
- Facade re-exports: `TaskType`, `classify_task`, `build_suite`, `compute_baseline_thresholds`, `generate_pytest` from `mltk.importer`.
- 46 new tests (16 classify, 12 suite_gen, 9 codegen, 9 CLI); zero network; acceptance gate = the emitted file for the bundled fixture runs green via subprocess (`3 passed, 3 skipped`).
- Second sprint of the 3-sprint epic (MCP tool + golden-set binding + registry integration remain in S99).

#### Smart Dataset Importer (S97)
- **`mltk.importer` package** — `DatasetImporter.load(source)` normalizes a local CSV/JSON/Parquet file or a HuggingFace Hub dataset id into an `ImportResult` (columns/dtypes/rows). `auto_map_columns()` infers a `ColumnRole` (INPUT/GOLDEN/CONTEXT/LABEL/METADATA/UNKNOWN) per column via deterministic, **token-based** name + dtype heuristics — keywords match whole word-tokens (split on non-alphanumerics and camelCase), never substrings, so `question_id` no longer false-matches `question`. A column whose last token is `id`/`idx`/`index`/`uid`/`uuid`/`timestamp`/`date`/`split` is now the highest-priority rule and always maps to `METADATA` (e.g. `question_id`, `created_date`). Never guesses when ambiguous — surfaces `UNKNOWN` for user review via `ColumnMapping.preview()`/`.override()`, and multiple `INPUT`-keyword candidates now demote all but the first (by column order) to `UNKNOWN` instead of silently overwriting one another.
- **Exclusive overrides** — `DatasetImporter.load(input_column=..., target_column=...)` and `ColumnMapping.override(column, role, exclusive=True)` demote any other column currently holding that role to `UNKNOWN`, guaranteeing the named column lands on `EvalSample.input`/`.target`. The default `override()` call remains non-exclusive.
- **Numeric-answer `GOLDEN` support** — a column whose whole name is exactly a `GOLDEN` keyword (`answer`/`target`/`expected`/`golden`/`reference`/`output`/`completion`/`response`) may now be numeric (e.g. a numeric `answer` column in a math-QA dataset); every other `GOLDEN` keyword match still requires string dtype.
- **`LABEL` → `metadata["category"]` mirroring** — a single `LABEL` column is now written to both `metadata["label"]` and `metadata["category"]`, so `EvalDataset.categories` and `assert_dataset_quality(min_categories=...)` work without extra plumbing; multiple `LABEL` columns still collapse to a `dict` under `metadata["label"]` only.
- **Missing-cell normalization** — missing cells (`None`/NaN/blank) in `LABEL`, `METADATA`/`UNKNOWN` passthrough, and extra-`INPUT` passthrough columns now normalize to `None` (never NaN, never the string `"None"`); non-missing values keep their original type. `INPUT`/`GOLDEN`/`context`/`references` behavior is unchanged.
- Standalone package (mirrors `mltk.cost`) — not imported by `mltk`'s top-level `__init__.py`, so `import mltk` never requires the optional `datasets`/`pyarrow` dependencies. New `mltk[importer]` extra, now floors `datasets>=4.0` (drops script-based/remote-code dataset loading).
- First sprint of a 3-sprint epic (task-type classification + suite/pytest generation + CLI in S98; MCP tool + golden-set binding + registry integration in S99).
- 200 new tests (incl. 41 regression tests from the 4-perspective PR review: token-based heuristics, exclusive overrides, category mirroring, missing-cell normalization, subprocess import-isolation guard); all HuggingFace/network calls mocked, zero live calls in CI.

#### Structured Output & Cost Tracking (S96)
- **`assert_valid_json` / `assert_json_schema` / `assert_pydantic_schema`** (`mltk.domains.llm`) — validate LLM outputs as JSON, against a JSON Schema (optional `jsonschema`), or against a Pydantic model (optional `pydantic`, v1/v2 auto-detected). Replaces the regex-only `assert_output_format` gap. The wrappers fail cleanly rather than leak a raw `SchemaError`/`TypeError`/`AttributeError` on a malformed schema, non-str input, or a non-`BaseModel` `model` argument.
- **`mltk.cost` package** — token/dollar cost tracking: `MODEL_PRICING` table (Anthropic + OpenAI, 15 models, prices as of 2026-06-30), `estimate_cost()`, runtime-overridable `register_pricing()` / `get_pricing()` (raises on unknown model, rejects negative tokens — never silently returns 0), `CostTracker` accumulator with `by_model()` breakdown, and `assert_cost_within()` / `assert_token_usage()` suite-level budget assertions (distinct from agentic's per-trace `assert_cost_budget`).
- 76 new tests; `jsonschema` and `pydantic` are optional deps surfaced with `pip install` hints. 5 new assertions (241 total).

### Changed

### Fixed

## [0.12.7] — 2026-04-30

### Fixed
- Add `ubuntu-24.04-arm` to release matrix so arm64 Linux wheels are published to PyPI — Docker arm64 builds were failing due to missing wheel (falling back to sdist + Rust compile inside slim container)

## [0.12.6] — 2026-04-30

### Added
- Homebrew tap: `brew tap Liorrr/mltk && brew install mltk` (custom tap at github.com/Liorrr/homebrew-mltk)

### Changed
- `release.yml`: add production PyPI publish job (`pypi` env, OIDC Trusted Publisher) — runs after TestPyPI succeeds
- README: add `## Install` section with Homebrew and pip instructions

## [0.12.5] — 2026-04-29

### Fixed
- Fix `LICENSE-COMMERCIAL` missing from sdist: add `license-files` (PEP 639) to `[project]` and `include` to `[tool.maturin]` — maturin was declaring it in metadata but not packaging it, causing TestPyPI 400 rejection

## [0.12.4] — 2026-04-29

### Fixed
- Add `MANIFEST.in` to explicitly include `LICENSE-COMMERCIAL` in sdist archive (maturin auto-discovers it for metadata but doesn't package it, causing TestPyPI 400 rejection)

## [0.12.3] — 2026-04-29

### Changed
- Repository is now public on GitHub
- PyPI distribution name is `mlspec` (`pip install mlspec`); Python import, CLI, and module names remain `mltk`

### Fixed
- CI: pin Python 3.12 in wheel build jobs (previously picked up runner default, causing cp312/cp314 mismatch across platforms)
- CI: add `skip-existing: true` to TestPyPI publish step so re-runs don't fail on already-uploaded files
- CI/Docker: update `MLTK_PIP_TARGET` from `mltk[all]` to `mlspec[all]` in Dockerfile and docker-publish workflow
- Rename `LICENSE-COMMERCIAL.md` → `LICENSE-COMMERCIAL` (TestPyPI rejected sdist when referenced license file had `.md` extension)

## [0.12.2] — 2026-04-28

### Changed
- Distribution name changed to `mlspec` on PyPI/TestPyPI (PyPI name `mltk` pending transfer claim)
- Added temporary installation note to README and docs pointing to `pip install git+https://github.com/Liorrr/mltk`

## [0.12.0] — 2026-04-25
### Added

#### Container & Kubernetes Friendliness (S93)
- **`mltk.container` module** — Trivy-backed container image security scanning
  - `assert_container_vulnerabilities(image, max_critical=0, max_high=0)` — pytest-native CVE threshold assertion
  - `assert_no_secrets_in_image(image)` — pytest-native exposed-secrets assertion
  - `ContainerScanner` — sibling scanner returning `ScanFinding` objects (not a `Scanner` ABC subclass)
  - `TrivyAdapter` — subprocess wrapper for Trivy JSON SchemaVersion 2 output; supports `scan_image` and `scan_fs`
  - `_binary.py` — Trivy binary auto-discovery: `PATH` → `trivy-py` installed binary → `ImportError` with install hint
- **MCP tool #12**: `mltk_container_scan(image, max_critical, max_high)` — scans image and returns structured JSON with pass/fail + CVE + secret details
- **CLI**: `mltk container scan <image>` — with `--max-critical`, `--max-high`, `--severity-floor`, `--json`, `--junit-xml` flags; exit codes 0/1/2
- **`/metrics` endpoint** on FastAPI server — Prometheus exposition format (opt-in: `pip install mltk[metrics]`); returns HTTP 404 if `prometheus_client` not installed
  - Counters: `mltk_assertions_total{status,category}`, `mltk_container_scan_vulnerabilities_total{severity}`
  - Histogram: `mltk_assertion_duration_seconds{category}`
- **Multi-architecture Docker images** on Docker Hub (`liorrr/mltk`) and GHCR (`ghcr.io/liorrr/mltk`) (published on `v*` tags via `docker-publish.yml`):
  - `:latest` / `:<version>` — `python:3.12-slim` + `mltk[all]`, `linux/amd64` + `linux/arm64`
  - `:full` / `:<version>-full` — `:latest` + Trivy 0.60.0 bundled at `/usr/local/bin/trivy`
- New docs: `guides/container-scanning.md`, `guides/container-deployment.md`
- New pyproject extras: `mltk[container]` (`trivy-py>=0.70`), `mltk[metrics]` (`prometheus-client>=0.20`); both included in `mltk[all]`
- 61 new tests (4273+ total); 2 known pre-existing leakage scanner failures unchanged

### Changed

### Fixed

## [0.11.1] — 2026-04-24
### Added

#### Persona Skills (S92)
- `mltk-qa` skill — QA engineer persona: scan → triage → assert → report workflow (174 lines)
- `mltk-dev` skill — Developer persona: TDD, failure fixes, test generation from scans (176 lines)
- `mltk-pm` skill — PM persona: ML Test Score, compliance, risk assessment, stakeholder reports (171 lines)
- `mltk-devops` skill — DevOps persona: CI/CD gates, server setup, monitoring, MCP config (221 lines)
- Updated `scripts/generate_skill_index.py` to install all `mltk-*.md` skills from repo

### Changed

#### License
- **License changed from Apache 2.0 to Elastic License 2.0 (ELv2).** Free for internal, non-commercial, and evaluation use. Redistribution as a hosted/managed service requires a commercial license. See [LICENSE](LICENSE) and [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL). Prior releases (v0.9.0 and earlier) remain under Apache 2.0.

### Fixed

#### S90-S92 Audit Fixes
- **SEC-2**: `similarity.py` now uses `_backends.embedding_cosine_pairs()` instead of direct `SentenceTransformer` — restores supply-chain revision pinning
- **SEC-2**: `_backends.py` uses `revision=` kwarg directly (not `model_kwargs`) for correct SentenceTransformer API
- `server.py`: `issue_url` field now properly stringified (was passing raw object for Jira)
- `jira_adapter.py`: `add_remote_link` and `update_issue` now log warnings on failure instead of silent swallow
- `github_adapter.py`: narrowed `except Exception` to `json.JSONDecodeError/ValueError`; moved `urllib.parse` to top-level import
- `judge.py`: empty-prompts pass case uses `Severity.INFO` (was `CRITICAL`)
- `similarity.py`: removed unused `numpy` import after refactor
- `docs/api/otel.md`: fixed all 3 `MltkTracer` method signatures (were documenting wrong API)
- `docs/api/llm.md`: fixed code-gen section (wrong module `code_gen` → `codegen`, 4 wrong function names)
- `docs/api/llm.md`: `assert_summary_conciseness` → `assert_summary_compression` (function didn't exist)
- `docs/api/llm.md`: extraction payload categories 8 → 9
- `BACKLOG.md`: header S91→S92, 228→230 assertions; footer updated
- `CHANGELOG.md`: source file count 9→7

### Added

#### OTLP / OpenInference (S92, IP-6)
- OpenInference span attributes on all assertion spans (`openinference.span.kind=EVALUATION`, `eval.name`, `eval.score`, `eval.label`)
- Phoenix displays mltk assertions in native Evaluations tab (not generic spans)
- Added to both live `trace_result()` and JSON `export_json()` code paths
- Environment variable documentation (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
- End-to-end workflow examples: local Phoenix, CI/CD JSON export, Langfuse scoring
- 2 new tests for OpenInference attribute verification

#### Embedding Model Upgrade (S92)
- Default embedding model upgraded from `all-MiniLM-L6-v2` (84-85% STS) to `all-mpnet-base-v2` (87-88% STS)
- Validated by SemScore paper (Jan 2024) as best sentence-transformer for LLM evaluation
- Pinned revision `e8c3b32edf5434bc` for supply-chain defense (SEC-2)
- MiniLM still supported — pass `embedding_model="all-MiniLM-L6-v2"` for lightweight mode
- Updated all 7 source files, 4 doc files, 2 test files, regenerated API index

#### Quick Wins (S92)
- `"semantic_equivalence"` criterion in LLM-as-Judge `DEFAULT_CRITERIA` — rubric for meaning-preserving evaluation
- Per-module coverage thresholds in `pyproject.toml` (`[tool.coverage.run]` + `[tool.coverage.report]`)
- Marked E-6 (semantic similarity for prompt leak detection) as done — already implemented in S85

#### Agent Protocol + E2E Pipeline Tests (S91, F-7)
- `mltk_workflow` MCP tool (11th) — canonical agent workflow with 5 pipeline paths and severity-based decision tree
- `workflow_hint` metadata in all tool success responses — `position` (start/middle/late/end/info) + `next_tools` list for agent routing
- Severity-conditional `suggested_next_step` in `mltk_scan` JSON-report path — critical→suggest, warning→issue, info→report
- `fallback_parameters` in `_error()` responses — mid-chain recovery guidance (e.g., PR failure → issue creation)
- `.mcp.json` sample config for Claude Code / Cursor / VS Code / Cline / OpenClaw
- ~55 new tests: 34 E2E agent simulation tests + 23 workflow/response enhancement tests

#### PR Generator + Issue Linker (S90, F-5+F-6)
- `PullRequestGenerator` — create GitHub PRs from scan findings + fix suggestions via isolated git worktrees
- `PullRequestResult` dataclass — PR URL, branch name, number, draft status
- `render_pr_body()` — structured Markdown PR body (finding/fix/code sections)
- `IssueLinker` — create tracker tickets from scan findings with dedup + template rendering
- `GitHubIssuesAdapter.create_pull_request()` — GitHub REST API PR creation with draft support and label attachment
- `JiraAdapter.add_remote_link()` — link external URLs (e.g., PRs) to Jira issues
- `mltk_create_pr` MCP tool (9th) — end-to-end PR creation from finding + fix JSON
- `mltk_create_issue` MCP tool (10th) — issue creation with GitHub/Jira backends, dedup, and optional PR linking
- `"finding_issue"` ticket template for scan-finding-based issues
- ~54 new tests across 4 test files (PR generator, issue linker, MCP tools, tool registration)

#### Sandboxed Execution (S89, F-4)
- `GitWorktree` context manager — create/cleanup git worktrees for isolated experiment execution
- `SandboxedExperimentRunner(ExperimentRunner)` — runs hypotheses in isolated git worktrees via subprocess
- `git_available()` / `find_git_root()` — git CLI detection and repo root discovery
- Path traversal protection in `write_file()` — validates relative paths stay inside worktree
- Code injection prevention in assertion scripts — scanner names escaped via `json.dumps()`
- `mltk_experiment` MCP tool gains `sandbox: bool = False` parameter for worktree-based execution
- Proper `ScanFinding` construction with baseline `TestResult` in MCP sandbox path
- ~97 new tests across 4 test files (worktree, sandbox, MCP sandbox, integration)

#### Experiment Runner (S88, F-3)
- `ExperimentRunner` — test fix hypotheses against scan findings: baseline → apply fix → re-run assertion → rank results
- `Hypothesis` / `HypothesisResult` dataclasses — pair fix suggestions with apply functions, track improvement and ranking
- `ExperimentResult` — aggregated results with `selected_fix`, `any_fix_works`, `best_result` properties
- `rank_hypotheses()` — 3 ranking strategies: `passed` (binary pass/fail), `delta` (metric improvement), `composite` (weighted score)
- `mltk_experiment` MCP tool — 8th tool, heuristic ranking of fixes by confidence/category/snippet availability
- Per-hypothesis timeout with daemon thread isolation (matches ScanEngine pattern)
- `run_batch()` for testing fixes across multiple findings with `apply_fns_map` lookup
- 58 new tests (14 dataclass + 10 runner + 10 ranking + 10 integration + 14 MCP)

#### Fix Suggestion Engine (S87, F-2)
- `FixSuggestion` dataclass — category (code/config/data/process), title, description, confidence (high/medium/low), code_snippet
- `ScanFinding.suggested_fixes` — 1-3 ranked fix suggestions per finding
- `_gen_fix()` / `_gen_null_fix()` / `_gen_pii_fix()` on all 8 scanners (drift, bias, overfit, calibration, data, leakage, robustness, slice)
- `mltk_suggest` MCP tool — 7th tool, parses finding JSON, returns ranked fixes with category/confidence filtering
- `format_fixes()` console formatter with confidence tags (+++/++/+) and code snippet display
- `format_console_output(verbose=True)` shows inline fix suggestions per finding
- `ScanReport.to_json()` serializes `suggested_fixes` array per finding
- `ScanReport.summary()` shows fix count footer
- `__post_init__` validation on FixSuggestion category and confidence values
- 51 new tests (12 dataclass + 10 engine + 15 integration + 14 MCP)

### Fixed

#### MCP Server Test Debt (S86)
- Rewrote 86 MCP server tests (77 were failing due to wrong mock targets and missing `create_server()` calls)
- Split monolithic `test_server.py` into 8 focused files with shared conftest/helpers
- Fixed mock targets: patch lazy imports at source modules (`mltk.scan.*`, `mltk.eval.task.*`, etc.) instead of non-existent module-level functions
- Added autouse fixture that creates mock server and populates tool registry before every test
- Added 7 hardening tests from Opus code review (YAML-not-dict, .yml extension, verbose .py, list error path, report FAIL items, dict results_json, 50-file cap)
- Total: 93 MCP server tests, all passing (was 77 failing / 9 passing)

#### MCP Evaluation (S75)
- `assert_mcp_tool_schema_conformance` — validate tool args against JSON Schema (first-mover, no LLM needed)
- `assert_mcp_tool_selection` — server-namespace-aware tool selection (precision/recall/F1)
- `assert_mcp_resource_access` — expected/forbidden URI access patterns (unique to mltk)
- `assert_mcp_context_window` — model-aware context utilization check
- `assert_mcp_error_recovery` — detect same-tool retry loops
- `McpTrace`, `McpToolCall`, `McpResourceAccess` dataclasses (extend AgentTrace)
- New `mcp` optional dependency group: `pip install mltk[mcp]`

#### LLM-as-Judge Defaults (S80, IP-1)
- `configure_default_judge()` — set a default judge_fn for all subjective assertions
- `resolve_judge()` — priority chain: explicit > module default > fallback method
- `assert_with_judge()` — convenience wrapper with auto-fallback to lexical
- Thread-safe module-level configuration

#### LLM Observability Adapters (S80, CG-5)
- `PhoenixAdapter` — wrap any mltk assertion as a Phoenix evaluator callable
- `register_phoenix()` — one-line OTLP endpoint configuration
- `LangfuseAdapter` — wrap mltk assertion as Langfuse score function
- `assert_trace_quality` — unified CI/CD quality gate (latency + cost + score)
- New `phoenix` and `langfuse` optional dependency groups

#### Multimodal Evaluation v2 (S79)
- `assert_clip_score` — CLIPScore via open-clip or pre-computed embeddings (dual-path, zero-dep option)
- `assert_object_hallucination` — POPE-style binary probing for VLM object hallucination
- `assert_edit_preservation` — SSIM structural similarity + pixel_diff fallback
- `assert_ocr_accuracy` — CER/WER for OCR quality (pure Python Levenshtein, zero deps)

#### Multimodal Evaluation v1 (S78)
- `assert_prompt_faithfulness` — text-to-image semantic alignment via LLM judge
- `assert_image_coherence` — image-text document coherence
- `assert_image_helpfulness` — image utility for comprehension
- `assert_vqa_accuracy` — VQA correctness (judge + exact match modes)
- `ImageInput` unified type (str path, Path, bytes) with `image_description` escape hatch
- New `multimodal` and `clip` optional dependency groups

#### Red Team v2 Enhancements (S78)
- `RedTeamSession` — stateful multi-turn attack management
- 3 built-in attack chains (trust building, roleplay escalation, context poisoning)
- Confidence tiers (COMPROMISED/LIKELY/AMBIGUOUS/RESILIENT) with indicator tracking
- `llm_attacker` parameter for LLM-generated payload variants

#### Red Team Framework (S77)
- `assert_red_team_resilient` — run 55+ attack payloads across 7 OWASP categories (closes CG-2)
- `assert_no_session_jailbreak` — multi-turn conversation attack detection
- `assert_owasp_llm_coverage` — meta-assertion for OWASP category coverage
- `assert_encoding_mutation_resilience` — 8 encoding bypass techniques (Base64, ROT13, leetspeak, etc.)
- `mltk security-scan` CLI command — run red team catalog against any model function
- 55 built-in educational attack payloads across 7 categories

#### Synthetic QA v2 Enhancements (S77)
- `generate_multi_hop()` — questions requiring cross-chunk reasoning
- `generate_conversational()` — multi-turn dialogue generation
- `generate_distracting()` — questions with misleading elements from different contexts
- New QuestionType values: CONVERSATIONAL, DISTRACTING

#### Synthetic QA Generation (S76)
- `SyntheticQAGenerator` — generate synthetic QA pairs from documents (closes CG-1)
- Template mode (zero-dep, CI-safe) + LLM mode (any `Callable[[str], str]`)
- 5 question types: factual, reasoning, multi-hop, counterfactual, out-of-scope
- `QAPair` dataclass integrates directly with RAG assertions
- `QualityFilter` for LLM-generated pair scoring
- `split_text()` zero-dep word-count text splitter

#### Test Hardening (S75-S76)
- +25 tests across behavioral stability, retrieval, paraphrase generator

#### Research (S75)
- Synthetic data generation research (RAGAS, DeepEval, Giskard comparison)

## [0.9.0] — 2026-03-31

### Added

#### NER PII Detection (S73)
- `assert_no_pii(method="ner")` — Microsoft Presidio + spaCy NER for contextual PII (names, orgs, locations)
- `assert_no_pii(method="gliner")` — GLiNER zero-shot NER for domain-specific PII (healthcare MRN, legal case numbers)
- `assert_no_pii(method="hybrid")` — regex + NER union with intelligent span deduplication
- `scan_pii_dispatch()` — unified routing function for all 4 methods
- `scan_pii_ner()`, `scan_pii_gliner()`, `scan_pii_hybrid()` — standalone NER scanning functions
- New `ner` optional dependency group: `pip install mltk[ner]`

#### Test Hardening (S73)
- +22 tests across drift (MMD), calibration (SmoothECE), fairness (intersectional), behavioral (invariance)
- High-dimensional MMD, perfectly calibrated ECE, three-attribute intersectionality, all 6 paraphrase methods

#### Research (S73)
- NER PII detection research brief (Presidio architecture, GLiNER zero-shot, hybrid approach)
- Red teaming architecture research (Promptfoo 135 plugins, Giskard GOAT, hybrid recommendation)
- MCP evaluation research (JSON Schema validation, resource access, DeepEval comparison)

## [0.8.0] — 2026-03-27

### Added

#### Integrations (S56-S57)
- GitHub App — webhook HMAC-SHA256 verification, check run creation, app auth (JWT → installation token)
- OpenTelemetry — `MltkTracer` (real/no-op modes), `trace_result`, `trace_suite`, `export_json`
- Weights & Biases — `WandbLogger` (log_result, log_suite, W&B Tables)
- DVC — `assert_dvc_file_tracked`, `assert_dvc_data_version`
- Kubeflow — `assert_kubeflow_pipeline_success`, `assert_kubeflow_step_outputs`
- SageMaker — `assert_sagemaker_pipeline_success`, `assert_sagemaker_step_status`
- Grafana — dashboard JSON export, provisioning YAML, 4-panel dashboard template

#### Enterprise (S58)
- RBAC — role-based access control (admin/writer/reader) for mltk server
- Audit log — SOC 2 compliant action logging with CSV export + `assert_audit_log_complete`
- HIPAA compliance mapping (4 rules) with `assert_hipaa_coverage`
- Custom compliance framework builder (YAML-driven)

#### Advanced ML Testing (S59)
- `assert_counterfactual_fairness` — per-sample fairness via attribute perturbation
- `assert_ate_significant` — Average Treatment Effect significance (pure numpy t-test)
- `assert_no_confounding` — detect treatment-feature correlations
- `assert_image_text_alignment` — multimodal CLIP-style alignment check
- `assert_cross_modal_consistency` — cross-modality prediction agreement
- `assert_reward_bounded`, `assert_cumulative_reward` — RL reward validation

#### Observability (S60)
- `assert_no_test_anomaly` — Z-score/IQR/percentile anomaly detection on test metrics
- `assert_impact_coverage` — verify all impacted tests were executed
- `analyze_impact` — import dependency graph for test impact analysis
- `TestScheduler` — periodic test run scheduling with webhook notifications
- Live monitoring portal — self-contained HTML with real-time polling (no CDN deps)

#### Retrieval Metrics + Developer Experience (S61)
- `assert_ndcg`, `assert_mrr`, `assert_recall_at_k`, `assert_map_at_k` — retrieval ranking metrics completing the RAG story
- `mltk list` CLI — assertion discovery with filter and JSON output (27th CLI command)
- JUnit XML export for Jenkins, GitLab CI, Azure DevOps integration

#### LLM-as-Judge + Summarization (S62)
- `assert_llm_judge_score` — score model outputs via any LLM (vendor-neutral judge_fn callable)
- `assert_llm_judge_pairwise` — A/B comparison via LLM judge (pairwise win rate)
- `assert_summary_coverage` — key information preservation (token recall)
- `assert_summary_compression` — compression ratio bounds
- `assert_summary_faithfulness` — no hallucinated content (token precision)
- `DEFAULT_CRITERIA` — 5 built-in rubrics (helpfulness, correctness, coherence, relevance, harmlessness)

#### Recommendation Systems (S63) — FIRST-MOVER
- `assert_hit_rate`, `assert_diversity`, `assert_novelty`, `assert_coverage`, `assert_serendipity`
- Zero competitors offer recommendation system assertions as pytest assertions

#### Long-Context LLM Testing (S63)
- `assert_needle_in_haystack` — fact retrieval at configurable context positions
- `assert_context_utilization` — verify model uses multiple facts from full window
- `assert_no_lost_in_middle` — detect accuracy degradation in middle of context

#### Composable TestSuite API (S64)
- `MltkSuite` — run assertions without pytest (notebooks, scripts, CI)
- `SuiteResult` — structured results with pass_rate, duration, counts
- Export to JSON, HTML, JUnit XML via `to_json()`, `to_html()`, `to_junit()`
- Method chaining: `suite.add(...).add(...).run()`

#### Code Generation Testing (S64)
- `assert_code_executes` — subprocess isolation with timeout
- `assert_code_passes_tests` — run generated code against test cases
- `assert_no_code_vulnerabilities` — AST scan for eval/exec/shell=True/hardcoded creds
- `assert_code_complexity` — cyclomatic complexity + line count bounds

#### Test Hardening (S57, S61, S62, S63, S64)
- 208 new parametrized + edge-case tests across safety, drift, synthetic, conformal, attribution, agentic, multi-agent, GitHub App, OTEL, kubeflow, sagemaker, dvc, hipaa, counterfactual, multimodal, anomaly, audit

## [0.7.0] — 2026-03-27

### Added

#### LLM Safety & Security (S47, S53)
- `assert_no_system_prompt_leakage` — 34 extraction payloads across 8 categories
- `assert_refusal_consistency` — phrasing-dependent safety gap detection
- `assert_safety_taxonomy` — per-category safety coverage
- Prompt injection payloads expanded 8 → 50 (6 categories, backward compatible)

#### Compliance (S48)
- NIST AI RMF mapping (Govern, Map, Measure, Manage) with `assert_nist_rmf_coverage`
- ISO 42001 mapping (8 Annex A controls) with `assert_iso_42001_coverage`
- `mltk compliance-gap` CLI — unified gap analysis across 5 frameworks

#### Agent Trace Testing (S49, S54)
- `AgentTrace`/`ToolCall` dataclasses with `from_dict()` (3 input formats)
- 9 agentic assertions: tool_chain, no_forbidden_actions, step_efficiency, no_redundant_calls, no_hallucinated_tools, cost_budget, error_recovery
- 2 multi-agent assertions: no_agent_loop, agent_handoff

#### Conformal Prediction (S50, S55)
- `assert_interval_coverage`, `assert_prediction_set_size`
- `assert_conformal_calibration` — two-sided calibration check
- `assert_conditional_coverage` — per-group fairness (Mondrian)

#### Distributed Training (S50)
- `assert_n_rank_gradient_sync`, `assert_gradient_alignment`
- `assert_weight_divergence`, `assert_gradient_clipped`

#### Drift Detection (S51)
- `assert_no_streaming_drift` with ADWIN and CUSUM detectors
- `assert_no_concept_drift` — P(Y|X) drift via chi2/fisher/proportion
- Completes drift story: P(X), P(Ŷ), streaming, P(Y|X)

#### Synthetic Data & NLP Robustness (S52)
- `assert_marginal_fidelity`, `assert_correlation_preserved`, `assert_synthetic_novelty`, `assert_dcr_safe`
- `TextPerturber` (4 methods) + `assert_text_robust`

#### Attribution Stability (S53)
- `assert_top_k_stable`, `assert_attribution_cosine_stability`

#### Infrastructure
- HTML report: pass/fail donut chart + module bar chart (pure CSS/SVG)
- TestPyPI step in release workflow
- OWASP LLM02/LLM06/LLM07/LLM08 mappings updated
- NIST AI RMF function mappings wired to new assertions
- 20 new MkDocs documentation pages

## [0.6.0] — 2026-03-26

### Added
- Server platform: FastAPI + SQLite + dashboard + Docker deployment
- Rust SIMD cosine similarity and BERTScore assertion
- PII expansion: international phones, MAC addresses, crypto wallets, allowlists
- Bias report generator with demographic breakdown
- RAGAS composite score, coherence check, OWASP LLM Top 10 mapping
- Multi-turn conversation evaluation (knowledge retention, turn relevancy)
- Data quality preset (one-call bundle) and sentiment analysis
- Benchmarks vs competitors, feature-label correlation shift, output drift detection

### Changed
- Embedding drift now uses Rust cosine when available
- pytest integration supports `--mltk-server` flag for auto-push

## [0.5.0] — 2026-03-26

### Added
- Data statistics: assert_column_mean, assert_column_median, assert_column_stdev, assert_quantiles
- Data validation: assert_datetime_format, assert_values_in_set, assert_no_conflicting_labels
- ML quality: assert_no_overfitting, assert_label_drift
- RAG evaluation (faithfulness, context precision/recall, answer relevancy)
- Agentic evaluation (task completion, tool selection, tool call correctness)
- Text quality assertions and training-serving skew detection

## [0.4.0] — 2026-03-25

### Added
- PII Tier 4: France NIR, Italy Codice Fiscale, Spain DNI
- Chat interface (ChatEngine, `mltk chat` CLI command)
- GitHub Issues adapter, Slack webhook notifications, plugin system
- Test resource registry with push/pull/list CLI commands
- Testing patterns: flaky detection, golden baselines, retry with confidence, smart test selection
- Local docs server with hot reload

## [0.3.0] — 2026-03-25

### Added
- PII Tier 3: UK NHS, UK NINO, Germany Steuer-ID, India Aadhaar, India PAN
- Training bug detection P2: augmentation, checkpoint, distributed, memory
- Rust acceleration: KL, chi-squared, Jensen-Shannon, Wasserstein, PII scanning
- Cloud monitoring: AWS SageMaker, GCP Vertex AI, Azure ML, Prometheus/Triton
- MLflow integration with `--mltk-mlflow` flag, Jupyter rich display, model card generator

## [0.2.0] — 2026-03-25

### Added
- Israel PII: Teudat Zehut (Luhn checksum), Israel phone numbers, IBAN MOD-97
- YAML test definitions with `mltk test` runner
- EU AI Act compliance report with article mapping and evidence HTML
- `mltk doctor` with 9 diagnostic checks and fix hints
- Environment variable config (MLTK_* prefix)
- CV tracking: assert_mota, assert_motp, assert_idf1
- Training bug P1: gradient and numerical stability checks

## [0.1.0] — 2026-03-25

### Added
- 60+ assertion functions across 6 domain kits (data, model, NLP, CV, speech, inference)
- Rust-accelerated drift (KS, PSI), PII scanning (24 patterns + Luhn)
- pytest plugin with `--mltk-report` HTML report generation
- CLI with 8 commands (run, report, config, doctor, etc.)
- Production monitoring: degradation detection, SLA compliance
- Tabular domain kit: feature drift, importance stability, class balance
- LLM evaluation: semantic similarity, toxicity, hallucination, latency (TTFT/ITL)
- Data contracts engine (YAML to pytest), drift expansion (JS, Wasserstein, embedding)
- Jira integration with ML ticket templates
- Face recognition: assert_face_far
- MkDocs documentation site
