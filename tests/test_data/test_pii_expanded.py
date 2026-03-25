"""Tests for expanded PII patterns — Sprint 13 Tier 1 additions."""

from mltk.data.pii import scan_pii


class TestExpandedPiiPatterns:
    """Tests for the 10 new Tier 1 PII patterns."""

    def test_ipv4_detected(self) -> None:
        """Detects IPv4 addresses in text."""
        matches = scan_pii("Server at 192.168.1.100 is down")
        assert any(m.type == "ipv4" for m in matches)

    def test_jwt_detected(self) -> None:
        """Detects JWT tokens (eyJ... format)."""
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        matches = scan_pii(f"Auth: {jwt}")
        assert any(m.type == "jwt" for m in matches)

    def test_pem_key_detected(self) -> None:
        """Detects PEM private key headers."""
        matches = scan_pii("-----BEGIN RSA PRIVATE KEY-----")
        assert any(m.type == "pem_private_key" for m in matches)

    def test_db_connection_string_detected(self) -> None:
        """Detects database connection strings with embedded credentials."""
        matches = scan_pii("postgres://admin:secret123@db.host.com:5432/mydb")
        assert any(m.type == "db_connection_string" for m in matches)

    def test_bearer_token_detected(self) -> None:
        """Detects Bearer authorization tokens."""
        matches = scan_pii("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert any(m.type == "bearer_token" for m in matches)

    def test_google_api_key_detected(self) -> None:
        """Detects Google API keys (AIza prefix)."""
        matches = scan_pii("key=AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY")
        assert any(m.type == "api_key" for m in matches)

    def test_clean_text_still_passes(self) -> None:
        """Normal text without PII returns no matches."""
        matches = scan_pii("The weather is nice today and I like Python programming")
        assert len(matches) == 0
