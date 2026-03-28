"""Tests for GitHub App integration — webhook verification, check runs, and App auth.

Covers:
- Webhook signature verification (valid, tampered, wrong secret, malformed)
- Check run output formatting (all pass, mixed, empty, annotations, truncation)
- Check run API creation (request body structure, error handling)
- GitHubAppAuth (JWT structure, token caching, token refresh, API errors)

All HTTP calls are mocked — no real network requests.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import MagicMock, patch

import pytest

from mltk.integrations.github_app import (
    _MAX_ANNOTATIONS_PER_REQUEST,
    _MAX_SUMMARY_LENGTH,
    GitHubAppAuth,
    create_check_run,
    format_check_run_output,
    verify_webhook_signature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status: int, body: dict) -> MagicMock:
    """Build a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _compute_signature(payload: bytes, secret: str) -> str:
    """Compute the HMAC-SHA256 signature the way GitHub does it.

    This helper is intentionally separate from the code under test so we
    have an independent reference implementation for comparison.
    """
    digest = hmac_mod.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def _make_http_error(code: int, body: dict):
    """Build a mock urllib HTTPError with a JSON body."""
    import io
    import urllib.error

    return urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=code,
        msg="Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(json.dumps(body).encode()),
    )


# ---------------------------------------------------------------------------
# Webhook Signature Verification
# ---------------------------------------------------------------------------


class TestVerifyWebhookSignature:
    # SCENARIO: payload signed with the correct secret
    # WHY: the happy path — a legitimate GitHub webhook delivery
    # EXPECTED: returns True
    def test_valid_signature_passes(self):
        payload = b'{"action": "opened", "number": 42}'
        secret = "my-webhook-secret"
        signature = _compute_signature(payload, secret)

        assert verify_webhook_signature(payload, signature, secret) is True

    # SCENARIO: payload has been modified after signing (man-in-the-middle)
    # WHY: signature verification must reject tampered payloads — this is
    #      the core security guarantee of HMAC
    # EXPECTED: returns False
    def test_tampered_payload_fails(self):
        original_payload = b'{"action": "opened", "number": 42}'
        secret = "my-webhook-secret"
        signature = _compute_signature(original_payload, secret)

        tampered_payload = b'{"action": "opened", "number": 999}'

        assert verify_webhook_signature(tampered_payload, signature, secret) is False

    # SCENARIO: signature was computed with a different secret
    # WHY: if an attacker guesses your endpoint but not your secret, their
    #      signatures won't match
    # EXPECTED: returns False
    def test_wrong_secret_fails(self):
        payload = b'{"action": "opened", "number": 42}'
        correct_secret = "correct-secret"
        wrong_secret = "attacker-guess"
        signature = _compute_signature(payload, correct_secret)

        assert verify_webhook_signature(payload, signature, wrong_secret) is False

    # SCENARIO: signature header is missing the "sha256=" prefix
    # WHY: malformed signatures should be rejected immediately, not cause
    #      parsing errors downstream
    # EXPECTED: returns False (no crash)
    def test_malformed_signature_prefix_fails(self):
        payload = b'{"action": "opened"}'
        # Missing "sha256=" prefix — should be rejected
        assert verify_webhook_signature(payload, "not-a-valid-signature", "secret") is False

    # SCENARIO: empty payload with valid signature
    # WHY: edge case — empty bodies are technically valid and have valid HMACs
    # EXPECTED: returns True (the HMAC of empty bytes is still deterministic)
    def test_empty_payload_with_valid_signature(self):
        payload = b""
        secret = "test-secret"
        signature = _compute_signature(payload, secret)

        assert verify_webhook_signature(payload, signature, secret) is True


# ---------------------------------------------------------------------------
# format_check_run_output
# ---------------------------------------------------------------------------


class TestFormatCheckRunOutput:
    # SCENARIO: all tests passed
    # WHY: the happy path — check run should be "success" with 100% pass rate
    # EXPECTED: conclusion is "success", summary contains pass count
    def test_all_tests_passed(self):
        results = [
            {"name": "drift_psi", "passed": True, "duration": 0.5, "message": "PSI=0.05"},
            {"name": "bias_dpd", "passed": True, "duration": 1.2, "message": "DPD=0.02"},
            {"name": "accuracy", "passed": True, "duration": 0.3, "message": "ACC=0.95"},
        ]
        output = format_check_run_output(results)

        assert output["conclusion"] == "success"
        assert "3/3" in output["summary"]
        assert "100.0%" in output["summary"]
        assert output["annotations"] == []

    # SCENARIO: mix of passed and failed tests
    # WHY: check run should be "failure" when any test fails, with annotations
    #      pointing to the exact files/lines that failed
    # EXPECTED: conclusion is "failure", failed tests have annotations
    def test_mixed_results(self):
        results = [
            {"name": "drift_psi", "passed": True, "duration": 0.5},
            {
                "name": "bias_dpd",
                "passed": False,
                "duration": 1.2,
                "message": "DPD=0.15 exceeds threshold 0.1",
                "file": "models/predict.py",
                "line": 42,
            },
            {"name": "accuracy", "passed": True, "duration": 0.3},
        ]
        output = format_check_run_output(results)

        assert output["conclusion"] == "failure"
        assert "2/3" in output["summary"]
        assert len(output["annotations"]) == 1
        assert output["annotations"][0]["path"] == "models/predict.py"
        assert output["annotations"][0]["start_line"] == 42
        assert output["annotations"][0]["annotation_level"] == "failure"
        assert "DPD=0.15" in output["annotations"][0]["message"]

    # SCENARIO: empty results list
    # WHY: edge case — no tests ran (maybe config error). Should return
    #      "neutral" (informational), not crash or report "success"
    # EXPECTED: conclusion is "neutral", summary says no results
    def test_empty_results(self):
        output = format_check_run_output([])

        assert output["conclusion"] == "neutral"
        assert "No test results" in output["summary"]
        assert output["annotations"] == []

    # SCENARIO: failed test without file/line info
    # WHY: not all tests are associated with source files (e.g., data quality
    #      tests). They should still fail the check but not create annotations
    # EXPECTED: conclusion is "failure", but no annotations generated
    def test_failed_test_without_file_no_annotation(self):
        results = [
            {"name": "data_quality", "passed": False, "message": "Missing column"},
        ]
        output = format_check_run_output(results)

        assert output["conclusion"] == "failure"
        assert "0/1" in output["summary"]
        # No annotation because no file path provided
        assert output["annotations"] == []

    # SCENARIO: custom title override
    # WHY: users may want domain-specific titles like "Drift Analysis" instead
    #      of the default "mltk Test Results"
    # EXPECTED: the custom title appears in the output
    def test_custom_title(self):
        results = [{"name": "test_1", "passed": True, "duration": 0.1}]
        output = format_check_run_output(results, title="Custom Title")

        assert output["title"] == "Custom Title"
        assert "Custom Title" in output["summary"]

    # SCENARIO: summary exceeds GitHub's 65,535 character limit
    # WHY: large test suites (hundreds of tests with long messages) can
    #      produce summaries that exceed the API limit. The function must
    #      truncate gracefully rather than let the API reject it.
    # EXPECTED: summary is truncated to within the limit with a notice
    def test_long_summary_truncated(self):
        # Generate enough results to exceed the limit
        results = [
            {
                "name": f"test_{i}",
                "passed": True,
                "duration": 0.1,
                "message": "A" * 500,
            }
            for i in range(200)
        ]
        output = format_check_run_output(results)

        assert len(output["summary"]) <= _MAX_SUMMARY_LENGTH
        assert "truncated" in output["summary"]

    # SCENARIO: message containing pipe characters
    # WHY: pipe chars break Markdown tables if not escaped. The function
    #      must escape them to keep the summary table well-formed.
    # EXPECTED: pipes in messages are escaped as \|
    def test_pipe_in_message_escaped(self):
        results = [
            {"name": "test_1", "passed": True, "duration": 0.1, "message": "a|b|c"},
        ]
        output = format_check_run_output(results)

        # The raw pipe should be escaped in the table row
        assert "a\\|b\\|c" in output["summary"]


# ---------------------------------------------------------------------------
# create_check_run
# ---------------------------------------------------------------------------


class TestCreateCheckRun:
    # SCENARIO: successful check run creation
    # WHY: verify the request body structure matches what GitHub expects —
    #      wrong field names or missing fields cause silent failures
    # EXPECTED: request body contains all required fields in correct format
    def test_request_body_structure(self):
        response_body = {"id": 1, "name": "mltk", "status": "completed"}
        mock_cm = _make_response(201, response_body)

        captured_requests: list = []

        def capturing_urlopen(req):
            captured_requests.append(req)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            result = create_check_run(
                repo="myorg/ml-service",
                head_sha="abc123def456",
                name="mltk",
                title="2/3 passed",
                summary="## Results\n...",
                conclusion="failure",
                token="ghs_test_token_123",
                details_url="https://example.com/results",
                annotations=[{
                    "path": "models/predict.py",
                    "start_line": 42,
                    "end_line": 42,
                    "annotation_level": "failure",
                    "message": "Drift detected",
                }],
            )

        assert result == response_body
        assert len(captured_requests) == 1

        req = captured_requests[0]
        # Verify URL
        assert "myorg/ml-service" in req.full_url
        assert req.full_url.endswith("/check-runs")
        # Verify auth header
        assert req.get_header("Authorization") == "Bearer ghs_test_token_123"
        # Verify body structure
        body = json.loads(req.data)
        assert body["name"] == "mltk"
        assert body["head_sha"] == "abc123def456"
        assert body["status"] == "completed"
        assert body["conclusion"] == "failure"
        assert body["output"]["title"] == "2/3 passed"
        assert body["details_url"] == "https://example.com/results"
        assert len(body["output"]["annotations"]) == 1
        assert body["output"]["annotations"][0]["path"] == "models/predict.py"

    # SCENARIO: GitHub API returns an error (e.g., 403 Forbidden)
    # WHY: the function must convert HTTP errors into clear RuntimeErrors
    #      so callers can handle them (retry, log, alert, etc.)
    # EXPECTED: RuntimeError raised with status code and message
    def test_api_error_raises_runtime_error(self):
        err = _make_http_error(403, {"message": "Resource not accessible by integration"})

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="403"):
                create_check_run(
                    repo="myorg/ml-service",
                    head_sha="abc123",
                    name="mltk",
                    title="test",
                    summary="test",
                    conclusion="success",
                    token="bad_token",
                )

    # SCENARIO: more than 50 annotations provided
    # WHY: GitHub's API rejects requests with >50 annotations. The function
    #      must silently truncate rather than fail.
    # EXPECTED: only 50 annotations included in the request body
    def test_annotations_capped_at_50(self):
        annotations = [
            {
                "path": f"file_{i}.py",
                "start_line": i,
                "end_line": i,
                "annotation_level": "failure",
                "message": f"Failed test {i}",
            }
            for i in range(75)
        ]

        mock_cm = _make_response(201, {"id": 1})

        captured_requests: list = []

        def capturing_urlopen(req):
            captured_requests.append(req)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            create_check_run(
                repo="myorg/ml-service",
                head_sha="abc123",
                name="mltk",
                title="test",
                summary="test",
                conclusion="failure",
                token="ghs_token",
                annotations=annotations,
            )

        body = json.loads(captured_requests[0].data)
        assert len(body["output"]["annotations"]) == _MAX_ANNOTATIONS_PER_REQUEST

    # SCENARIO: no optional fields provided (no details_url, no annotations)
    # WHY: the minimal check run (just name + sha + conclusion) must work
    #      without optional fields being present in the body
    # EXPECTED: body omits details_url and annotations keys
    def test_minimal_check_run_no_optional_fields(self):
        mock_cm = _make_response(201, {"id": 1})

        captured_requests: list = []

        def capturing_urlopen(req):
            captured_requests.append(req)
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            create_check_run(
                repo="myorg/ml-service",
                head_sha="abc123",
                name="mltk",
                title="All passed",
                summary="OK",
                conclusion="success",
                token="ghs_token",
            )

        body = json.loads(captured_requests[0].data)
        assert "details_url" not in body
        assert "annotations" not in body["output"]


# ---------------------------------------------------------------------------
# GitHubAppAuth
# ---------------------------------------------------------------------------


class TestGitHubAppAuth:
    # SCENARIO: JWT generation produces valid base64url-encoded segments
    # WHY: JWTs have strict formatting rules — each of the three dot-separated
    #      segments must be valid base64url (no padding, URL-safe chars only).
    #      Malformed JWTs are silently rejected by GitHub with a 401.
    # EXPECTED: three dot-separated segments, each valid base64url
    def test_jwt_structure_is_valid(self):
        auth = GitHubAppAuth(
            app_id="12345",
            private_key="test-private-key-content",
            installation_id="67890",
        )
        jwt = auth._generate_jwt()

        parts = jwt.split(".")
        assert len(parts) == 3, "JWT must have exactly 3 segments (header.payload.signature)"

        # Verify header
        import base64

        # Re-add padding for decoding (base64url strips it)
        header_padded = parts[0] + "=" * (4 - len(parts[0]) % 4)
        header = json.loads(base64.urlsafe_b64decode(header_padded))
        assert header["alg"] == "HS256"
        assert header["typ"] == "JWT"

        # Verify payload claims
        payload_padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_padded))
        assert payload["iss"] == "12345"
        assert "iat" in payload
        assert "exp" in payload
        # Expiry should be ~10 minutes after issued-at
        assert payload["exp"] - payload["iat"] == 660  # 60s skew + 600s expiry

        # Verify no padding chars ('=') in any segment (base64url requirement)
        for part in parts:
            assert "=" not in part, "base64url must not contain padding"
            assert "+" not in part, "base64url must not contain '+'"
            assert "/" not in part, "base64url must not contain '/'"

    # SCENARIO: installation token is cached and reused within its validity
    # WHY: requesting a new token on every API call would be slow (extra HTTP
    #      roundtrip) and could hit rate limits. The cache avoids this.
    # EXPECTED: second call returns cached token without making an HTTP request
    def test_token_caching_avoids_extra_requests(self):
        auth = GitHubAppAuth(
            app_id="12345",
            private_key="test-key",
            installation_id="67890",
        )
        token_response = {
            "token": "ghs_cached_token_abc123",
            "expires_at": "2099-12-31T23:59:59Z",
        }
        mock_cm = _make_response(201, token_response)

        call_count = 0

        def counting_urlopen(req):
            nonlocal call_count
            call_count += 1
            return mock_cm

        with patch("urllib.request.urlopen", side_effect=counting_urlopen):
            token1 = auth.get_installation_token()
            token2 = auth.get_installation_token()

        assert token1 == "ghs_cached_token_abc123"
        assert token2 == "ghs_cached_token_abc123"
        assert call_count == 1, "Second call should use cached token, not make HTTP request"

    # SCENARIO: cached token has expired, must fetch a new one
    # WHY: installation tokens expire after 1 hour. The auth class must
    #      detect expiry and transparently fetch a fresh token.
    # EXPECTED: expired token triggers a new HTTP request, returns new token
    def test_expired_token_triggers_refresh(self):
        auth = GitHubAppAuth(
            app_id="12345",
            private_key="test-key",
            installation_id="67890",
        )

        # First token — already expired (expiry in the past)
        first_response = {
            "token": "ghs_old_token",
            "expires_at": "2020-01-01T00:00:00Z",
        }
        # Second token — valid far in the future
        second_response = {
            "token": "ghs_new_token",
            "expires_at": "2099-12-31T23:59:59Z",
        }

        call_count = 0

        def sequenced_urlopen(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(201, first_response)
            return _make_response(201, second_response)

        with patch("urllib.request.urlopen", side_effect=sequenced_urlopen):
            token1 = auth.get_installation_token()
            # First token is expired, so next call should fetch again
            token2 = auth.get_installation_token()

        assert token1 == "ghs_old_token"
        assert token2 == "ghs_new_token"
        assert call_count == 2, "Expired token should trigger a new request"

    # SCENARIO: GitHub API rejects the token request (e.g., bad private key)
    # WHY: clear error messages help users diagnose auth issues — "bad
    #      private key" is much more helpful than a raw 401
    # EXPECTED: RuntimeError raised with status code and GitHub's error message
    def test_token_request_failure_raises_error(self):
        auth = GitHubAppAuth(
            app_id="12345",
            private_key="bad-key",
            installation_id="67890",
        )
        err = _make_http_error(401, {"message": "A JSON web token could not be decoded"})

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="401"):
                auth.get_installation_token()

    # SCENARIO: base64url encoding produces URL-safe output without padding
    # WHY: standard base64 contains '+', '/', and '=' which are not safe in
    #      URLs and JWTs. base64url replaces these. Incorrect encoding causes
    #      silent auth failures that are extremely hard to debug.
    # EXPECTED: output contains only alphanumeric, '-', and '_' characters
    def test_base64url_encoding_no_padding_or_unsafe_chars(self):
        auth = GitHubAppAuth("1", "k", "1")

        # Test with bytes that produce '+' and '/' in standard base64
        # 0xFB, 0xEF, 0xBE = standard base64 "++/+" which must become "--_+"
        test_bytes = bytes([0xFB, 0xEF, 0xBE, 0xFB, 0xEF, 0xBE])
        result = auth._base64url_encode(test_bytes)

        assert "+" not in result
        assert "/" not in result
        assert "=" not in result
        # Only alphanumeric, '-', and '_' allowed
        import re
        assert re.match(r'^[A-Za-z0-9_-]+$', result), f"Invalid base64url chars in: {result}"


# ---------------------------------------------------------------------------
# Hardening: edge-case and stress tests (appended)
# ---------------------------------------------------------------------------


class TestWebhookHMACEmptyPayload:
    """HMAC verification edge cases with empty payload."""

    def test_empty_payload_wrong_secret_fails(self):
        """FAIL: Empty payload with wrong secret.

        Even though the payload is empty, the HMAC depends on
        the secret. A different secret must produce a mismatch.
        """
        payload = b""
        correct_secret = "real-secret"
        wrong_secret = "other-secret"
        signature = _compute_signature(payload, correct_secret)
        assert verify_webhook_signature(
            payload, signature, wrong_secret,
        ) is False

    def test_empty_payload_empty_signature_fails(self):
        """FAIL: Empty payload with empty signature string.

        An empty signature cannot match the sha256= prefix,
        so it must be rejected.
        """
        assert verify_webhook_signature(
            b"", "", "secret",
        ) is False


class TestFormatCheckRunOutputStress:
    """Stress tests for format_check_run_output."""

    def test_sixty_results_all_annotated(self):
        """Verify >50 results produce correct output.

        format_check_run_output returns ALL annotations.
        The capping to 50 is done in create_check_run before
        sending to the API. All 60 annotations should be
        present in the formatted output.
        """
        results = [
            {
                "name": f"test_{i}",
                "passed": False,
                "duration": 0.1,
                "message": f"Failed check {i}",
                "file": f"src/module_{i}.py",
                "line": i + 1,
            }
            for i in range(60)
        ]
        output = format_check_run_output(results)
        assert output["conclusion"] == "failure"
        assert "0/60" in output["summary"]
        assert len(output["annotations"]) == 60


class TestAnnotationsSpecialCharsInPath:
    """Special characters in file paths for annotations."""

    def test_special_chars_in_file_path(self):
        """Annotations with spaces and special chars in path.

        GitHub API accepts any valid path string. The function
        must pass special characters through without mangling.
        """
        results = [
            {
                "name": "special_test",
                "passed": False,
                "duration": 0.2,
                "message": "Threshold exceeded",
                "file": "src/my module/predict (v2).py",
                "line": 10,
            },
        ]
        output = format_check_run_output(results)
        assert output["conclusion"] == "failure"
        assert len(output["annotations"]) == 1
        ann = output["annotations"][0]
        assert ann["path"] == "src/my module/predict (v2).py"
        assert ann["start_line"] == 10


class TestVeryLongSummaryTruncation:
    """Very long summary (>65K chars) truncation."""

    def test_extreme_summary_truncated(self):
        """Summary with extremely long messages is truncated.

        Each result has a 2000-char message. With 50 results,
        the raw summary would be ~100K chars, well above the
        65,535 limit.
        """
        results = [
            {
                "name": f"test_{i}",
                "passed": True,
                "duration": 0.1,
                "message": "B" * 2000,
            }
            for i in range(50)
        ]
        output = format_check_run_output(results)
        assert len(output["summary"]) <= _MAX_SUMMARY_LENGTH
        assert "truncated" in output["summary"]
