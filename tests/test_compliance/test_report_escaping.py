"""Compliance report HTML escaping tests."""

from __future__ import annotations

import json

from mltk.compliance.generator import generate_compliance_report


def _results_file(tmp_path, results):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def test_system_name_is_escaped(tmp_path):
    html = generate_compliance_report(
        _results_file(tmp_path, []),
        system_name="<script>alert(1)</script>",
        output_dir=tmp_path / "reports",
    ).read_text(encoding="utf-8")

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_result_name_and_message_are_escaped(tmp_path):
    html = generate_compliance_report(
        _results_file(
            tmp_path,
            [{
                "name": "model.metric.<img src=x onerror=alert(1)>",
                "passed": False,
                "message": "<b>x</b>",
            }],
        ),
        output_dir=tmp_path / "reports",
    ).read_text(encoding="utf-8")

    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "<b>x</b>" not in html


def test_plain_report_renders_unchanged(tmp_path):
    html = generate_compliance_report(
        _results_file(
            tmp_path,
            [{
                "name": "model.metric.accuracy",
                "passed": True,
                "message": "A & B",
            }],
        ),
        system_name="Normal System",
        output_dir=tmp_path / "reports",
    ).read_text(encoding="utf-8")

    assert "Normal System" in html
    assert "model.metric.accuracy" in html
    assert "A &amp; B" in html
    assert "Art. 15" in html
    assert "Total Tests" in html
    assert "Passed" in html
    assert "Compliance Score" in html
