"""Tests for mltk.report.model_card — Google Model Card generator.

Each test follows the pattern:
  # SCENARIO: <what situation is being tested>
  # WHY: <why this matters / what could go wrong>
  # EXPECTED: <the concrete assertion>
"""

from __future__ import annotations

import json
from pathlib import Path

from mltk.report.model_card import generate_model_card

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_results(tmp_path: Path, results: list[dict]) -> Path:
    """Write a list of result dicts to a JSON file and return its path."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results), encoding="utf-8")
    return p


def _make_result(
    name: str,
    passed: bool = True,
    message: str = "",
    severity: str = "error",
    details: dict | None = None,
    duration_ms: float = 1.0,
) -> dict:
    return {
        "name": name,
        "passed": passed,
        "message": message,
        "severity": severity,
        "details": details or {},
        "duration_ms": duration_ms,
    }


# ---------------------------------------------------------------------------
# Test 1 — file creation
# ---------------------------------------------------------------------------

def test_generate_card_creates_file(tmp_path: Path) -> None:
    # SCENARIO: generate_model_card is called with a valid results JSON
    # WHY: The function must produce a file at output_path; if it doesn't,
    #       nothing downstream can consume the card
    # EXPECTED: The output path exists and has a .md suffix

    results_path = _write_results(tmp_path, [
        _make_result("model.metric.accuracy", passed=True, details={"actual_value": 0.92}),
    ])
    out_path = tmp_path / "my-card.md"

    returned = generate_model_card(
        results_path=results_path,
        model_name="Test Model",
        model_version="1.0",
        output_path=out_path,
    )

    assert returned == out_path
    assert out_path.exists(), "model card file was not created"
    assert out_path.suffix == ".md"


# ---------------------------------------------------------------------------
# Test 2 — model details section
# ---------------------------------------------------------------------------

def test_card_has_model_details(tmp_path: Path) -> None:
    # SCENARIO: Model name and version are passed as arguments
    # WHY: The Model Details section is the primary identity block of the card;
    #       missing name/version makes the card useless for attribution
    # EXPECTED: The generated Markdown contains both the model name and version

    results_path = _write_results(tmp_path, [
        _make_result("model.metric.f1", passed=True, details={"actual_value": 0.88}),
    ])
    out_path = tmp_path / "card.md"

    generate_model_card(
        results_path=results_path,
        model_name="FraudNet",
        model_version="3.1.4",
        output_path=out_path,
    )

    content = out_path.read_text(encoding="utf-8")
    assert "FraudNet" in content, "model name not found in card"
    assert "3.1.4" in content, "model version not found in card"
    assert "## 1. Model Details" in content, "Model Details section header missing"


# ---------------------------------------------------------------------------
# Test 3 — metrics section with actual values
# ---------------------------------------------------------------------------

def test_card_has_metrics(tmp_path: Path) -> None:
    # SCENARIO: Results contain model.metric.* entries with actual_value in details
    # WHY: Metrics are the core quantitative output — consumers need to see
    #       accuracy, F1, etc. pulled through from test details
    # EXPECTED: The Metrics section exists and contains the numeric values
    #            from the details dict

    results_path = _write_results(tmp_path, [
        _make_result(
            "model.metric.accuracy",
            passed=True,
            details={"actual_value": 0.95, "threshold": 0.90},
        ),
        _make_result(
            "model.metric.f1",
            passed=False,
            message="F1 below threshold",
            details={"actual_value": 0.72, "threshold": 0.80},
        ),
    ])
    out_path = tmp_path / "card.md"

    generate_model_card(results_path=results_path, output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "## 3. Metrics" in content, "Metrics section missing"
    assert "0.95" in content, "accuracy actual_value not in card"
    assert "0.72" in content, "f1 actual_value not in card"
    # Python serialises 0.90 → "0.9" and 0.80 → "0.8" when rendering floats
    assert "0.9" in content, "accuracy threshold not in card"
    assert "0.8" in content, "f1 threshold not in card"


# ---------------------------------------------------------------------------
# Test 4 — limitations section lists failed tests
# ---------------------------------------------------------------------------

def test_card_has_limitations(tmp_path: Path) -> None:
    # SCENARIO: Some tests have passed=False with descriptive messages
    # WHY: The Limitations section is the transparency mechanism — readers
    #       must be able to identify what the model can't do reliably
    # EXPECTED: Every failed test's name and message appears under
    #            "Limitations & Known Issues"

    results_path = _write_results(tmp_path, [
        _make_result("model.metric.accuracy", passed=True),
        _make_result(
            "model.bias.demographic_parity",
            passed=False,
            message="Disparity ratio 0.74 exceeds threshold 0.80",
        ),
        _make_result(
            "data.schema.feature_types",
            passed=False,
            message="Column 'age' has wrong dtype",
        ),
    ])
    out_path = tmp_path / "card.md"

    generate_model_card(results_path=results_path, output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "## 9. Limitations" in content, "Limitations section missing"
    assert "model.bias.demographic_parity" in content
    assert "Disparity ratio 0.74 exceeds threshold 0.80" in content
    assert "data.schema.feature_types" in content
    assert "Column 'age' has wrong dtype" in content


# ---------------------------------------------------------------------------
# Test 5 — empty results do not crash
# ---------------------------------------------------------------------------

def test_card_empty_results(tmp_path: Path) -> None:
    # SCENARIO: The results JSON is a valid but empty list []
    # WHY: Empty runs happen during CI setup or when no tests match a marker;
    #       the generator must be robust and produce a complete card regardless
    # EXPECTED: No exception is raised and the output file is created with
    #            the expected section headers

    results_path = _write_results(tmp_path, [])
    out_path = tmp_path / "empty-card.md"

    # Must not raise
    generate_model_card(
        results_path=results_path,
        model_name="EmptyModel",
        model_version="0.0.0",
        output_path=out_path,
    )

    assert out_path.exists(), "card not created for empty results"
    content = out_path.read_text(encoding="utf-8")
    # All mandatory sections must still appear
    for section in [
        "## 1. Model Details",
        "## 2. Intended Use",
        "## 3. Metrics",
        "## 4. Fairness Analysis",
        "## 5. Subgroup Performance",
        "## 6. Calibration",
        "## 7. Robustness",
        "## 8. Data Quality Summary",
        "## 9. Limitations",
        "## 10. Generated By",
    ]:
        assert section in content, f"Section '{section}' missing from empty-results card"


# ---------------------------------------------------------------------------
# Test 6 — data quality summary shows pass/fail counts
# ---------------------------------------------------------------------------

def test_card_data_quality_summary(tmp_path: Path) -> None:
    # SCENARIO: Results include a mix of passing and failing data.* tests
    # WHY: Data quality is a distinct section; consumers need a quick count
    #       of how many data checks passed vs. failed without reading each row
    # EXPECTED: The Data Quality Summary section reflects the correct totals
    #            and individual sub-categories are listed when more than one
    #            data prefix is present

    results_path = _write_results(tmp_path, [
        _make_result("data.schema.dtypes", passed=True),
        _make_result("data.schema.nulls", passed=True),
        _make_result("data.drift.ks_test", passed=False, message="KS drift detected"),
        _make_result("data.pii.email_scan", passed=True),
        _make_result("model.metric.accuracy", passed=True),
    ])
    out_path = tmp_path / "card.md"

    generate_model_card(results_path=results_path, output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "## 8. Data Quality Summary" in content
    # 3 passed, 1 failed out of 4 data tests
    assert "3" in content, "passed count (3) missing from data quality section"
    assert "1" in content, "failed count (1) missing from data quality section"
    # Sub-categories must be visible
    assert "data.schema" in content
    assert "data.drift" in content
    assert "data.pii" in content


# ---------------------------------------------------------------------------
# Test 7 — fairness section renders bias results
# ---------------------------------------------------------------------------

def test_card_fairness_section(tmp_path: Path) -> None:
    # SCENARIO: Results contain model.bias.* entries with method and group_metrics
    # WHY: Fairness is a top-level Google Model Cards section; auditors look here
    #       specifically to assess protected-attribute disparities
    # EXPECTED: The Fairness Analysis section contains the bias test name,
    #            method, and per-group metric values

    results_path = _write_results(tmp_path, [
        _make_result(
            "model.bias.equalized_odds",
            passed=False,
            message="TPR gap 0.18 exceeds limit 0.10",
            details={
                "method": "equalized_odds",
                "group_metrics": {"male": 0.91, "female": 0.73},
            },
        ),
    ])
    out_path = tmp_path / "card.md"

    generate_model_card(results_path=results_path, output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "## 4. Fairness Analysis" in content
    assert "equalized_odds" in content
    assert "male" in content
    assert "female" in content
    assert "0.91" in content
    assert "0.73" in content
