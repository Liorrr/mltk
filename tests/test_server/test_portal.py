"""Tests for mltk.server.portal -- live test monitoring portal.

The portal generates a self-contained HTML page that polls the mltk server
for real-time test results.  These tests verify the HTML structure, security
properties (no external CDN), and the data extraction function.

No network access or running server is required.

Each test follows the pattern:

    # SCENARIO: <what situation is being tested>
    # WHY: <reason this behaviour matters>
    # EXPECTED: <what the test asserts>

Test coverage:
    1. create_portal_html returns a complete, valid HTML document
    2. Generated HTML contains no external CDN references
    3. refresh_seconds value is correctly embedded in the output
    4. get_portal_data returns the expected dictionary keys
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mltk.server.portal import create_portal_html, get_portal_data

# ---------------------------------------------------------------------------
# Tests: HTML generation
# ---------------------------------------------------------------------------


class TestCreatePortalHtml:
    # SCENARIO: generate portal HTML with default settings
    # WHY: the portal is the entry point for live monitoring.  If the HTML is
    #      malformed, browsers will render a blank page with no error message.
    # EXPECTED: returns a string that starts with <!DOCTYPE html> and contains
    #           the essential structural elements (head, body, script)
    def test_returns_valid_html_string(self):
        html = create_portal_html()

        assert isinstance(html, str), "must return a string"
        assert html.strip().startswith("<!DOCTYPE html>"), "must start with DOCTYPE"
        assert "<html" in html, "must contain <html> tag"
        assert "<head>" in html, "must contain <head> section"
        assert "<body>" in html, "must contain <body> section"
        assert "</html>" in html, "must close the <html> tag"
        assert "<script>" in html, "must contain inline JavaScript"
        assert "mltk" in html.lower(), "must reference mltk"

    # SCENARIO: verify the HTML has no external CDN references
    # WHY: external CDN dependencies break behind corporate firewalls, in air-
    #      gapped environments, and when CDNs have outages.  The portal must be
    #      100% self-contained -- all CSS inline, no <link> to external sheets,
    #      no <script src="https://...">.
    # EXPECTED: no href/src pointing to external domains
    def test_no_external_cdn_references(self):
        html = create_portal_html()

        # Must not load any external JavaScript
        assert 'src="http' not in html, "must not reference external JS via http"
        assert "src='http" not in html, "must not reference external JS via http (single quotes)"
        # Must not load external CSS
        assert 'href="http' not in html, "must not reference external CSS via http"
        assert "href='http" not in html, "must not reference external CSS via http (single quotes)"
        # Common CDN domains that should never appear
        for cdn in ["cdn.jsdelivr.net", "cdnjs.cloudflare.com", "unpkg.com", "cdn.tailwindcss.com"]:
            assert cdn not in html, f"must not reference CDN: {cdn}"

    # SCENARIO: custom refresh_seconds is embedded in the generated HTML
    # WHY: if the refresh interval is not correctly baked into the HTML,
    #      the portal will poll at the wrong frequency -- either hammering
    #      the server (too fast) or appearing stale (too slow).
    # EXPECTED: the JS code contains the custom interval value
    def test_refresh_seconds_embedded(self):
        html_15 = create_portal_html(refresh_seconds=15)
        html_60 = create_portal_html(refresh_seconds=60)

        # The interval should appear in the JavaScript (as milliseconds or seconds)
        # Our implementation uses: const INTERVAL = {safe_interval} * 1000;
        assert "15" in html_15, "15-second interval must appear in HTML"
        assert "60" in html_60, "60-second interval must appear in HTML"

        # Also verify the human-readable display in the meta section
        assert "every 15s" in html_15 or "every 15 s" in html_15, (
            "15-second refresh must be shown to the user"
        )
        assert "every 60s" in html_60 or "every 60 s" in html_60, (
            "60-second refresh must be shown to the user"
        )


# ---------------------------------------------------------------------------
# Tests: portal data extraction
# ---------------------------------------------------------------------------


class TestGetPortalData:
    # SCENARIO: get_portal_data with a mock storage that has runs
    # WHY: the portal data function bridges the storage layer and the frontend.
    #      It must return a consistent shape regardless of what storage contains,
    #      so the JavaScript can always find the keys it expects.
    # EXPECTED: result has 'runs', 'summary', and 'health' keys with correct types
    def test_returns_expected_keys(self):
        # Build a mock storage with sample run data
        mock_storage = MagicMock()
        mock_storage.get_runs.return_value = [
            {"id": 2, "project": "default", "timestamp": "2025-01-02T00:00:00",
             "total": 10, "passed": 9, "failed": 1, "score": 90.0, "duration_ms": 500.0},
            {"id": 1, "project": "default", "timestamp": "2025-01-01T00:00:00",
             "total": 10, "passed": 10, "failed": 0, "score": 100.0, "duration_ms": 400.0},
        ]

        data = get_portal_data(mock_storage)

        # Top-level keys
        assert "runs" in data, "must contain 'runs' key"
        assert "summary" in data, "must contain 'summary' key"
        assert "health" in data, "must contain 'health' key"

        # Summary structure
        summary = data["summary"]
        assert "total_runs" in summary, "summary must have 'total_runs'"
        assert "pass_rate" in summary, "summary must have 'pass_rate'"
        assert "avg_duration_ms" in summary, "summary must have 'avg_duration_ms'"
        assert "latest_score" in summary, "summary must have 'latest_score'"

        # Health is a string label
        assert data["health"] in {"healthy", "degraded", "failing", "unknown"}, (
            "health must be one of the defined labels"
        )

        # Values make sense for the mock data
        assert summary["total_runs"] == 2
        assert summary["pass_rate"] == 50.0  # 1 of 2 runs was fully passing
        assert data["runs"] == mock_storage.get_runs.return_value

    # SCENARIO: get_portal_data with None storage
    # WHY: during server startup or in test environments, storage may not be
    #      initialized yet.  The function must return a safe empty response.
    # EXPECTED: empty runs, zero summary values, "unknown" health
    def test_none_storage_returns_safe_defaults(self):
        data = get_portal_data(None)

        assert data["runs"] == []
        assert data["summary"]["total_runs"] == 0
        assert data["summary"]["pass_rate"] == 0.0
        assert data["health"] == "unknown"
