"""Tests for shared judge-score parsing."""

from __future__ import annotations

import math

from mltk.core.judge_parse import parse_judge_score


def test_parse_judge_score_json_and_text() -> None:
    assert parse_judge_score('{"reasoning_steps": 3, "score": 8}') == 8.0
    assert parse_judge_score('{"score": null}') is None
    assert parse_judge_score('{"score": "N/A"}') is None
    assert parse_judge_score("I give this a 4/5") == 4.0
    assert parse_judge_score("no number") is None
    assert parse_judge_score("[3, 9]") == 3.0


def test_parse_judge_score_json_nan_is_preserved() -> None:
    assert math.isnan(parse_judge_score('{"score": NaN}'))


def test_parse_score_aliases() -> None:
    from mltk.domains.llm.judge import _parse_score as llm_parse_score
    from mltk.domains.multimodal._scoring import _parse_score as multimodal_parse_score

    assert llm_parse_score is parse_judge_score
    assert multimodal_parse_score is parse_judge_score
