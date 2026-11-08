"""Tests for shared judge-score parsing."""

from __future__ import annotations

import math

import pytest

from mltk.core.judge_parse import parse_judge_score


def test_parse_judge_score_json_and_text() -> None:
    assert parse_judge_score('{"reasoning_steps": 3, "score": 8}') == 8.0
    assert parse_judge_score('{"score": null}') is None
    assert parse_judge_score('{"score": "N/A"}') is None
    assert parse_judge_score("I give this a 4/5") == 4.0
    assert parse_judge_score("no number") is None
    assert parse_judge_score("[3, 9]") is None
    assert parse_judge_score("[3, 9]", legacy=True) == 3.0


def test_parse_judge_score_json_nan_requires_legacy() -> None:
    assert parse_judge_score('{"score": NaN}') is None
    assert math.isnan(parse_judge_score('{"score": NaN}', legacy=True))


@pytest.mark.parametrize("raw", [
    '{"reasoning_steps": 3, "score": 4}',
    '```json\n{"reasoning_steps": 3, "score": 4}\n```',
    'I checked 3 claims; overall score: 4/5',
    '<think>The answer cites 2 sources; 1 claim is unsupported, '
    'so maybe 3 of 5.</think>\n{"score": 4}',
])
def test_requirement_probes(raw: str) -> None:
    assert parse_judge_score(raw, expected_max_score=5) / 5 == 0.8


@pytest.mark.parametrize("raw", [
    'There are 3 issues and 2 strengths.', 'I rate this 3 out of 5',
    '{"score": true}', '{"score": false}', '{"score": null}',
    '{"score": "NaN"}', '{"score": Infinity}', '{"score": 1e999}',
    '<think>score: 4', '</think>score: 4', 'score: 4/0',
    'score: 4/-5', 'score: 4/10', 'score: 4 then score: 5',
    '{"score": nope} then {"score": 4}',
    '{"score": 4} then {"score": "invalid"}',
    'score: 4/NaN', 'score: 4foo',
])
def test_fail_closed(raw: str) -> None:
    assert parse_judge_score(raw, expected_max_score=5) is None


def test_last_json_and_nested_strings() -> None:
    raw = '{"score": 2} revised: {"reason": "a } brace", "score": 4} {"other": 9}'
    assert parse_judge_score(raw) == 4


def test_custom_reasoning_and_unclosed_blocks() -> None:
    delimiters = (("[reason]", "[/reason]"),)
    assert parse_judge_score(
        '[reason]score: 1 <think>score: 2</think>[/reason] score: 4',
        reasoning_delimiters=delimiters,
    ) == 4
    assert parse_judge_score(
        '[reason]score: 1', reasoning_delimiters=delimiters,
    ) is None
    with pytest.raises(ValueError, match="distinct nonempty strings"):
        parse_judge_score('4', reasoning_delimiters=(("", "end"),))


def test_fraction_policy_and_numeric_callbacks() -> None:
    assert parse_judge_score('4/10', expected_max_score=5) is None
    assert parse_judge_score('4/10', expected_max_score=5, legacy=True) == 4
    assert parse_judge_score('4/10') == 4
    assert parse_judge_score(-1.0, legacy=True) == -1.0
    assert parse_judge_score(-1.0) == -1.0
    assert parse_judge_score(True) is None
    assert parse_judge_score(float('inf')) is None


def test_parse_score_aliases() -> None:
    from mltk.domains.llm.judge import _parse_score as llm_parse_score
    from mltk.domains.multimodal._scoring import _parse_score as multimodal_parse_score

    assert llm_parse_score is parse_judge_score
    assert multimodal_parse_score is parse_judge_score


@pytest.mark.parametrize(
    "surface", ["faithfulness", "coherence", "helpfulness", "vqa", "default"],
)
def test_numeric_adapters_share_strict_parser(surface):
    from mltk.core.assertion import MltkAssertionError
    from mltk.domains.llm.judge_defaults import assert_with_judge
    from mltk.domains.multimodal.alignment import (
        assert_image_coherence,
        assert_prompt_faithfulness,
    )
    from mltk.domains.multimodal.vlm import assert_image_helpfulness, assert_vqa_accuracy

    def invoke(raw, **options):
        def judge(*args):
            return raw
        image = {"image": None, "image_description": "A tree", "judge_fn": judge}
        if surface == "faithfulness":
            return assert_prompt_faithfulness(prompt="tree", **image, **options)
        if surface == "coherence":
            return assert_image_coherence(text="tree", **image, **options)
        if surface == "helpfulness":
            return assert_image_helpfulness(question="what?", answer="tree", **image, **options)
        if surface == "vqa":
            return assert_vqa_accuracy(
                question="what?", expected_answer="tree", actual_answer="tree", **image, **options,
            )
        return assert_with_judge("quality", "tree", "tree", judge_fn=judge, **options)

    result = invoke(
        '[r]score: 0.2[/r] ```json\n{"steps": 3, "score": 0.8}\n```',
        reasoning_delimiters=(("[r]", "[/r]"),),
    )
    assert result.details["score"] == 0.8
    assert invoke('0.8/1').details["score"] == 0.8
    for invalid in (
        '0.8/5', '2 issues 3 strengths', '<think>score: 1', True, float('nan'),
    ):
        with pytest.raises(MltkAssertionError) as exc:
            invoke(invalid, min_score=0)
        assert exc.value.result.details["score"] is None
    assert invoke('invalid', on_error="zero", min_score=0).details["score"] == 0
    assert invoke('there are 0.8 points', legacy_parse=True).details["score"] == 0.8
    assert invoke(-1.0, legacy_parse=True, min_score=-2).details["score"] == -1


def test_configured_default_judge_uses_shared_parser(monkeypatch):
    from mltk.domains.llm import judge_defaults

    monkeypatch.setattr(
        judge_defaults, "_default_judge",
        lambda _: '<think>score: 0.1</think> {"score": 0.8}',
    )
    assert judge_defaults.assert_with_judge("quality", "a", "b").details["score"] == 0.8
