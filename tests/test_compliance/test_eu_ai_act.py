"""Tests for EU AI Act compliance module (mltk.compliance).

Covers: risk classification, article mapping, gap detection,
report generation, HTML structure, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mltk.compliance.eu_ai_act import (
    RISK_LEVELS,
    classify_risk,
    find_gaps,
    map_results_to_articles,
)
from mltk.compliance.generator import generate_compliance_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results_json(tmp_path: Path, results: list[dict]) -> Path:
    """Write a results list to a temporary JSON file and return its path."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results), encoding="utf-8")
    return p


SAMPLE_RESULTS = [
    {"name": "model.metric.accuracy", "passed": True,  "message": "accuracy=0.94"},
    {"name": "model.metric.f1",       "passed": False, "message": "f1=0.61 < 0.70"},
    {"name": "data.schema.columns",   "passed": True,  "message": "schema ok"},
    {"name": "data.no_nulls.train",   "passed": True,  "message": "0 nulls"},
    {"name": "model.bias.demographic_parity", "passed": False, "message": "disparity=0.18 > 0.10"},
    {"name": "model.calibration.ece", "passed": True,  "message": "ECE=0.04"},
    {"name": "monitor.degradation.week", "passed": True, "message": "no degradation"},
]


# ---------------------------------------------------------------------------
# Test 1 — classify_risk returns correct classification for "high"
# ---------------------------------------------------------------------------


def test_classify_risk_high():
    # SCENARIO: caller requests classification for the "high" risk level
    # WHY:      high-risk AI systems have the most comprehensive requirements
    #           under EU AI Act Title III; the returned dict must be correct
    # EXPECTED: label, color, badge_class, and non-empty articles_required list

    info = classify_risk("high")

    assert info["label"] == "High Risk"
    assert info["badge_class"] == "warn"
    assert isinstance(info["articles_required"], list)
    assert len(info["articles_required"]) >= 4, (
        "High-risk systems should require at least 4 articles"
    )
    assert "Art. 10"    in info["articles_required"]
    assert "Art. 15"    in info["articles_required"]
    assert "Art. 72"    in info["articles_required"]
    assert "description" in info
    assert "color"       in info


def test_classify_risk_all_levels():
    # SCENARIO: all four risk levels are requested in sequence
    # WHY:      every level in RISK_LEVELS must be a valid key without raising
    # EXPECTED: each level returns a dict with required keys

    required_keys = {"label", "description", "color", "badge_class", "articles_required"}
    for level in RISK_LEVELS:
        info = classify_risk(level)
        assert required_keys.issubset(info.keys()), (
            f"Missing keys for level {level!r}"
        )


def test_classify_risk_unknown_raises():
    # SCENARIO: an unrecognised risk level string is passed
    # WHY:      callers must receive a clear ValueError rather than a KeyError
    # EXPECTED: ValueError is raised containing the bad level name

    with pytest.raises(ValueError, match="Unknown risk level"):
        classify_risk("extreme")


# ---------------------------------------------------------------------------
# Test 2 — map_results_to_articles groups by article correctly
# ---------------------------------------------------------------------------


def test_map_results_to_articles():
    # SCENARIO: a mixed list of results covering several assertion prefixes
    # WHY:      the grouped output drives the per-article HTML sections;
    #           results must land in the right article bucket
    # EXPECTED: Art. 15 bucket contains metric results; Art. 10(2f) has bias result

    grouped = map_results_to_articles(SAMPLE_RESULTS)

    # Art. 15 — Accuracy & Robustness (model.metric.*)
    assert "Art. 15" in grouped
    art15_names = [r["name"] for r in grouped["Art. 15"]]
    assert "model.metric.accuracy" in art15_names
    assert "model.metric.f1"       in art15_names

    # Art. 10 — Data Governance (data.schema.*, data.no_nulls.*)
    assert "Art. 10" in grouped
    art10_names = [r["name"] for r in grouped["Art. 10"]]
    assert "data.schema.columns" in art10_names
    assert "data.no_nulls.train" in art10_names

    # Art. 10(2f) — Bias Detection (model.bias.*)
    assert "Art. 10(2f)" in grouped
    bias_names = [r["name"] for r in grouped["Art. 10(2f)"]]
    assert "model.bias.demographic_parity" in bias_names

    # Art. 14 — Human Oversight (model.calibration.*)
    assert "Art. 14" in grouped
    art14_names = [r["name"] for r in grouped["Art. 14"]]
    assert "model.calibration.ece" in art14_names

    # Art. 72 — Post-market Monitoring (monitor.degradation.*)
    assert "Art. 72" in grouped
    art72_names = [r["name"] for r in grouped["Art. 72"]]
    assert "monitor.degradation.week" in art72_names


def test_map_results_to_articles_enriches_article_key():
    # SCENARIO: mapped results are checked for the injected "article" key
    # WHY:      generator.py reads r["article"] when building article_sections
    # EXPECTED: every result dict in a named bucket has an "article" key

    grouped = map_results_to_articles(SAMPLE_RESULTS)
    for article, items in grouped.items():
        if article == "uncategorised":
            continue
        for item in items:
            assert item.get("article") == article, (
                f"Result {item['name']!r} in bucket {article!r} "
                f"has article={item.get('article')!r}"
            )


# ---------------------------------------------------------------------------
# Test 3 — find_gaps flags missing bias tests
# ---------------------------------------------------------------------------


def test_find_gaps_missing_bias():
    # SCENARIO: results list has no model.bias.* assertions for a high-risk system
    # WHY:      Art. 10(2f) is mandatory for high-risk; missing coverage is a gap
    # EXPECTED: "Art. 10(2f)" appears in the returned gaps list

    results_no_bias = [
        r for r in SAMPLE_RESULTS if not r["name"].startswith("model.bias")
    ]
    gaps = find_gaps(results_no_bias, "high")

    assert "Art. 10(2f)" in gaps, (
        "Expected Art. 10(2f) to be flagged as a gap when no bias tests are present"
    )


def test_find_gaps_no_gaps_when_all_covered():
    # SCENARIO: results cover all required articles for "high" risk level
    # WHY:      a fully covered system should report zero gaps
    # EXPECTED: gaps list is empty

    full_results = [
        {"name": "data.schema.x",                   "passed": True,  "message": ""},
        {"name": "model.bias.x",                     "passed": True,  "message": ""},
        {"name": "model.calibration.x",              "passed": True,  "message": ""},
        {"name": "model.metric.x",                   "passed": True,  "message": ""},
        {"name": "monitor.degradation.x",            "passed": True,  "message": ""},
    ]
    gaps = find_gaps(full_results, "high")
    assert gaps == [], f"Expected no gaps but got: {gaps}"


def test_find_gaps_minimal_risk_no_requirements():
    # SCENARIO: risk level is "minimal" which has no mandatory articles
    # WHY:      minimal-risk systems have no mandatory testing requirements;
    #           gaps should always be empty regardless of results
    # EXPECTED: empty gaps list even with empty results

    gaps = find_gaps([], "minimal")
    assert gaps == []


# ---------------------------------------------------------------------------
# Test 4 — generate_report_creates_file
# ---------------------------------------------------------------------------


def test_generate_report_creates_file(tmp_path: Path):
    # SCENARIO: generate_compliance_report is called with a valid results JSON
    # WHY:      the core contract is that an HTML file is written to disk
    # EXPECTED: returned Path exists and has .html suffix

    results_file = _make_results_json(tmp_path, SAMPLE_RESULTS)
    out_dir = tmp_path / "reports"

    out_path = generate_compliance_report(
        results_path=results_file,
        risk_level="high",
        system_name="Test Classifier",
        output_dir=out_dir,
    )

    assert out_path.exists(), f"Expected HTML file at {out_path}"
    assert out_path.suffix == ".html"
    assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 5 — report contains expected HTML sections
# ---------------------------------------------------------------------------


def test_report_contains_sections(tmp_path: Path):
    # SCENARIO: the generated HTML is inspected for mandatory structural elements
    # WHY:      template sections must render without error and contain expected
    #           article references, system name, risk label, and mltk branding
    # EXPECTED: HTML string contains key section markers

    results_file = _make_results_json(tmp_path, SAMPLE_RESULTS)
    out_path = generate_compliance_report(
        results_path=results_file,
        risk_level="high",
        system_name="My Test System",
        output_dir=tmp_path / "reports",
    )

    html = out_path.read_text(encoding="utf-8")

    # System name and risk label appear in header
    assert "My Test System" in html
    assert "High Risk"       in html

    # All five mandatory articles for high risk are present
    assert "Art. 10"    in html
    assert "Art. 10(2f)" in html
    assert "Art. 14"    in html
    assert "Art. 15"    in html
    assert "Art. 72"    in html

    # Section titles
    assert "Data Governance"        in html
    assert "Bias Detection"         in html
    assert "Accuracy" in html
    assert "Robustness" in html

    # Compliance score section
    assert "Compliance Score" in html

    # Coverage gaps section
    assert "Coverage Gaps" in html

    # mltk branding footer
    assert "mltk" in html


# ---------------------------------------------------------------------------
# Test 6 — empty results do not crash
# ---------------------------------------------------------------------------


def test_empty_results(tmp_path: Path):
    # SCENARIO: results JSON is an empty list
    # WHY:      a system that has run zero assertions should still produce a
    #           valid (all-gaps) report rather than crashing
    # EXPECTED: HTML file is created, compliance score is 0, all articles are gaps

    results_file = _make_results_json(tmp_path, [])
    out_path = generate_compliance_report(
        results_path=results_file,
        risk_level="high",
        system_name="Empty System",
        output_dir=tmp_path / "reports",
    )

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")

    # Score should be 0
    assert "0.0%" in html or "0%" in html

    # All required articles should show "NO TESTS"
    assert "NO TESTS" in html
