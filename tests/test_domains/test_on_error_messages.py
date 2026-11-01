"""on_error policy: failure messages and run-level (not per-category) scope.

When errors cause the failure, the message must lead with the error reason
instead of a misleading score ("resilience too low: 1.0000").  For the CLI
scan, ``fail_if_all`` applies to the whole run, so one fully erroring
category in an otherwise healthy run still counts as resisted.
"""

from __future__ import annotations

import pytest

from mltk.cli.security_scan import _run_scan
from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.red_team import assert_red_team_resilient
from mltk.domains.llm.red_team.catalog import AttackCategory, AttackPayload
from mltk.domains.nlp.security import assert_no_prompt_injection


def _dead(prompt: str) -> str:
    raise ConnectionError("endpoint down")


def _safe(prompt: str) -> str:
    return "I cannot help with that."


class TestErrorFailureMessages:
    def test_red_team_all_errors_message_leads_with_reason(self) -> None:
        with pytest.raises(MltkAssertionError) as exc:
            assert_red_team_resilient(_dead)
        result = exc.value.result
        assert result.passed is False
        assert result.message.startswith("all ")
        assert "endpoint unreachable" in result.message
        assert "too low" not in result.message
        assert result.details["error_rate"] == 1.0

    def test_red_team_partial_errors_pass_with_count(self) -> None:
        calls = {"n": 0}

        def flaky(prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("blip")
            return _safe(prompt)

        result = assert_red_team_resilient(flaky)
        assert result.passed is True
        assert result.details["errored_count"] == 1
        assert "calls errored)" in result.message

    def test_prompt_injection_all_errors_message(self) -> None:
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_prompt_injection(_dead, payloads=["a", "b"])
        message = exc.value.result.message
        assert message.startswith("all 2 model calls raised")
        assert "No prompt injection vulnerabilities" in message


class TestScanFailIfAllIsRunLevel:
    def _catalog(self) -> list[AttackPayload]:
        return [
            AttackPayload(
                category=AttackCategory.JAILBREAK,
                payload_text="jb-1",
                description="jb",
                owasp_id="LLM01",
            ),
            AttackPayload(
                category=AttackCategory.PROMPT_INJECTION,
                payload_text="pi-1",
                description="pi",
                owasp_id="LLM01",
            ),
        ]

    def test_one_dead_category_still_resisted(self) -> None:
        def half_dead(prompt: str) -> str:
            if prompt.startswith("jb"):
                raise ConnectionError("guardrail raised")
            return _safe(prompt)

        results = _run_scan(half_dead, self._catalog())
        assert all(c["resisted"] == c["tested"] for c in results.values())
        assert sum(c["errored_count"] for c in results.values()) == 1

    def test_all_dead_run_resists_nothing(self) -> None:
        results = _run_scan(_dead, self._catalog())
        assert all(c["resisted"] == 0 for c in results.values())

    def test_resist_mode_keeps_legacy(self) -> None:
        results = _run_scan(_dead, self._catalog(), on_error="resist")
        assert all(c["resisted"] == c["tested"] for c in results.values())

    def test_strict_mode_never_resists_errors(self) -> None:
        def half_dead(prompt: str) -> str:
            if prompt.startswith("jb"):
                raise ConnectionError("guardrail raised")
            return _safe(prompt)

        results = _run_scan(half_dead, self._catalog(), on_error="strict")
        resisted = sum(c["resisted"] for c in results.values())
        assert resisted == 1


class TestReviewRegressions:
    """Gate-review findings on the on_error rollout."""

    def _catalog(self, n: int) -> list[AttackPayload]:
        return [
            AttackPayload(
                category=AttackCategory.JAILBREAK,
                payload_text=f"jb-{i}",
                description="jb",
                owasp_id="LLM01",
            )
            for i in range(n)
        ]

    def test_cli_strict_fails_on_one_error_despite_rate(self) -> None:
        from mltk.cli.security_scan import _scan_passed

        calls = {"n": 0}

        def one_error(prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("blip")
            return _safe(prompt)

        cats = self._catalog(20)
        strict = _run_scan(one_error, cats, on_error="strict")
        assert _scan_passed(strict, 0.8, "strict") is False
        calls["n"] = 0
        default = _run_scan(one_error, cats)
        assert _scan_passed(default, 0.8, "fail_if_all") is True

    def test_cli_json_carries_error_fields(self) -> None:
        import json

        from mltk.cli.security_scan import _format_json

        results = _run_scan(_dead, self._catalog(2), on_error="resist")
        report = json.loads(
            _format_json("t", results, 0.8, on_error="resist"),
        )
        cat = next(iter(report["categories"].values()))
        assert cat["errored_count"] == 2
        assert cat["error_rate"] == 1.0
        assert report["on_error"] == "resist"

    def test_defaults_on_error_not_merged_into_plugin_params(self) -> None:
        from mltk.testdefs.runner import _merge_defaults
        from mltk.testdefs.schema import RedTeamDefaults

        merged = _merge_defaults(RedTeamDefaults(on_error="strict"), {})
        assert "on_error" not in merged
        explicit = _merge_defaults(
            RedTeamDefaults(), {"on_error": "resist"},
        )
        assert explicit["on_error"] == "resist"

    def test_reused_session_fail_if_all_scoped_to_chain(self) -> None:
        from mltk.domains.llm.red_team.session import RedTeamSession

        state = {"up": True}

        def model(prompt: str) -> str:
            if not state["up"]:
                raise ConnectionError("down")
            return _safe(prompt)

        session = RedTeamSession(model_fn=model, max_turns=10)
        session.send("hello")  # earlier successful turn
        state["up"] = False
        with pytest.raises(RuntimeError, match="all 2 turns raised"):
            session.run_chain(["a", "b"])
