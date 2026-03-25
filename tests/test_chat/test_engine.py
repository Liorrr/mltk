"""Tests for mltk.chat.engine — rule-based Q&A about test results."""

from __future__ import annotations

import json
from pathlib import Path

from mltk.chat.engine import ChatEngine


def _sample_results() -> list[dict]:
    """Create sample test results for testing the chat engine."""
    return [
        {
            "name": "data.schema",
            "passed": True,
            "severity": "critical",
            "message": "Schema valid",
            "details": {},
            "duration_ms": 5.2,
        },
        {
            "name": "data.drift",
            "passed": False,
            "severity": "critical",
            "message": "PSI 0.35 > 0.1 threshold",
            "details": {"statistic": 0.35, "threshold": 0.1},
            "duration_ms": 89.1,
        },
        {
            "name": "model.metric",
            "passed": True,
            "severity": "critical",
            "message": "F1 0.92 >= 0.85",
            "details": {"actual_value": 0.92, "threshold": 0.85},
            "duration_ms": 156.2,
        },
        {
            "name": "model.bias",
            "passed": False,
            "severity": "critical",
            "message": "Demographic parity gap 0.15 > 0.10",
            "details": {"method": "demographic_parity"},
            "duration_ms": 45.0,
        },
        {
            "name": "inference.latency",
            "passed": False,
            "severity": "critical",
            "message": "P95 120ms > 50ms threshold",
            "details": {"p95": 120.0, "threshold": 50.0},
            "duration_ms": 2340.5,
        },
    ]


def _write_results(tmp_path: Path, results: list[dict] | None = None) -> Path:
    """Write sample results to a JSON file."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results or _sample_results()), encoding="utf-8")
    return p


class TestChatEngineLoad:
    # SCENARIO: Load results from JSON file
    # WHY: Engine must parse the --mltk-export-json format
    # EXPECTED: results loaded correctly
    def test_load_results(self, tmp_path):
        path = _write_results(tmp_path)
        engine = ChatEngine(str(path))
        assert len(engine.results) == 5

    # SCENARIO: Load with wrapper format {"results": [...]}
    # WHY: Some export formats wrap results in a dict
    # EXPECTED: results extracted from wrapper
    def test_load_wrapped_results(self, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(
            json.dumps({"results": _sample_results()}), encoding="utf-8"
        )
        engine = ChatEngine(str(p))
        assert len(engine.results) == 5

    # SCENARIO: No results file provided
    # WHY: User can start chat without results and get help
    # EXPECTED: empty results, no crash
    def test_empty_results(self):
        engine = ChatEngine()
        assert len(engine.results) == 0
        answer = engine.ask("summary")
        assert "no results" in answer.lower() or "0" in answer


class TestChatEngineAsk:
    # SCENARIO: Ask "what failed?"
    # WHY: Most common question — what went wrong
    # EXPECTED: lists 3 failed tests
    def test_ask_failures(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("what failed?")
        assert "drift" in answer.lower()
        assert "bias" in answer.lower()

    # SCENARIO: Ask "summary"
    # WHY: Quick overview of test run
    # EXPECTED: contains pass/fail counts
    def test_ask_summary(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("summary")
        assert "5" in answer or "total" in answer.lower()

    # SCENARIO: Ask "why did drift fail?"
    # WHY: Drill into specific failure
    # EXPECTED: contains drift-specific details
    def test_ask_why(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("why did drift fail?")
        assert "drift" in answer.lower()

    # SCENARIO: Ask "recommend"
    # WHY: Help user add missing test coverage
    # EXPECTED: suggests missing categories
    def test_ask_recommendations(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("recommend")
        # Sample results are missing monitoring, pipeline, etc.
        assert len(answer) > 0

    # SCENARIO: Ask "slowest"
    # WHY: Find performance bottlenecks
    # EXPECTED: lists tests by duration
    def test_ask_slowest(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("slowest")
        assert "latency" in answer.lower() or "ms" in answer.lower()

    # SCENARIO: Ask about drift specifically
    # WHY: Filter to drift-related results
    # EXPECTED: contains drift info
    def test_ask_drift(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("drift")
        assert "drift" in answer.lower()

    # SCENARIO: Ask about bias
    # WHY: Filter to bias-related results
    # EXPECTED: contains bias info
    def test_ask_bias(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("bias")
        assert "bias" in answer.lower()

    # SCENARIO: Ask "help"
    # WHY: User needs to know available commands
    # EXPECTED: lists available question types
    def test_ask_help(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("help")
        assert "failed" in answer.lower() or "summary" in answer.lower()

    # SCENARIO: Ask unknown question
    # WHY: Graceful handling of unrecognized input
    # EXPECTED: helpful response, not crash
    def test_ask_unknown(self, tmp_path):
        engine = ChatEngine(str(_write_results(tmp_path)))
        answer = engine.ask("what is the meaning of life?")
        assert len(answer) > 0  # Should return something helpful
