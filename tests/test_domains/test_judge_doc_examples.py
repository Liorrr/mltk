"""Executable examples copied from the judge-defaults documentation."""

from __future__ import annotations

import json
import urllib.request

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm import assert_faithfulness
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

