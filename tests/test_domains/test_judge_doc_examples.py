"""Executable examples copied from the judge-defaults documentation."""

from __future__ import annotations

import json
import urllib.request

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm import assert_faithfulness
from mltk.domains.llm.judge import assert_llm_judge_score
from mltk.domains.llm.judge_defaults import (
    assert_with_judge,
    configure_default_judge,
)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _ollama_judge(prompt: str) -> str:
    """The documented Ollama callback, with the network supplied by a test."""
    payload = json.dumps({
        "model": "llama3.2",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


@pytest.fixture(autouse=True)
def _clear_default_judge():
    configure_default_judge(None)
    yield
    configure_default_judge(None)


def test_documented_ollama_callback_returns_raw_reply(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req: _FakeResponse(
            json.dumps({"message": {"content": "Score: 0.8"}}).encode()
        ),
    )
    configure_default_judge(_ollama_judge)

    result = assert_with_judge(
        "quality", "answer", "reference", min_score=0.5,
    )

    assert result.passed
    assert result.details["score"] == 0.8


def test_documented_mock_default_judge_uses_zero_to_one_scale() -> None:
    configure_default_judge(lambda prompt: "0.8")
    result = assert_with_judge("quality", "answer", "reference")
    assert result.passed
    assert result.details["score"] == 0.8


def test_documented_default_judge_out_of_range_reply_is_an_error() -> None:
    configure_default_judge(lambda prompt: "4.0")
    with pytest.raises(MltkAssertionError) as exc:
        assert_with_judge("quality", "answer", "reference")
    assert "score 4.0 outside [0, 1.0]" in exc.value.result.message


@pytest.mark.parametrize(("score", "should_pass"), [(0.9, True), (0.2, False)])
def test_documented_parametrized_judge_thresholds(score, should_pass) -> None:
    configure_default_judge(lambda prompt: score)
    if should_pass:
        result = assert_with_judge(
            "faithfulness", "The sky is blue.",
            "Scientific studies confirm the sky appears blue.",
            min_score=0.75,
        )
        assert result.passed
    else:
        with pytest.raises(MltkAssertionError):
            assert_with_judge(
                "faithfulness", "The sky is blue.",
                "Scientific studies confirm the sky appears blue.",
                min_score=0.75,
            )


def test_documented_repetition_example_uses_real_judge_signatures() -> None:
    def judge(prompt: str) -> float:
        return 4.0

    def faithfulness_judge(answer: str, context: str) -> float:
        return 0.9

    faithfulness = assert_faithfulness(
        "The answer is grounded.",
        "The context supports the answer.",
        method="llm",
        judge_fn=faithfulness_judge,
    )
    scored = assert_llm_judge_score(
        judge_fn=judge,
        prompts=["question"],
        responses=["answer"],
        criterion="helpfulness",
    )

    assert faithfulness.passed
    assert scored.passed


def test_faithfulness_llm_method_without_judge_fails() -> None:
    with pytest.raises(MltkAssertionError, match="requires a judge_fn"):
        assert_faithfulness("answer", "context", method="llm")


def test_faithfulness_llm_method_uses_explicit_judge() -> None:
    result = assert_faithfulness(
        "answer", "context", method="llm",
        judge_fn=lambda answer, context: 0.9,
        min_score=0.7,
    )
    assert result.passed

