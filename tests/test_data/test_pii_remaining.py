"""Tests for Sprint 34 Tier 5 PII patterns and allowlist support in mltk.data.pii.

Covers:
  - E.164 international phone numbers (intl_phone)
  - MAC addresses (mac_address)
  - Bitcoin addresses (bitcoin_address)
  - Ethereum addresses (ethereum_address)
  - Allowlist: exact-match suppression in scan_pii()
  - Allowlist: partial suppression (only matching strings are suppressed)

All test values are chosen to be unambiguous matches / non-matches for the
respective patterns.
"""

from __future__ import annotations

from mltk.data.pii import scan_pii

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _types(text: str, **kwargs) -> list[str]:
    """Return the list of PII type strings detected in text."""
    return [m.type for m in scan_pii(text, **kwargs)]


def _matched(text: str, **kwargs) -> list[str]:
    """Return the list of matched strings detected in text."""
    return [m.matched_text for m in scan_pii(text, **kwargs)]


# ---------------------------------------------------------------------------
# intl_phone
# ---------------------------------------------------------------------------

def test_intl_phone_detected() -> None:
    # SCENARIO: Classic US E.164 format with country code +1 and dashes
    # WHY: Most common international phone format; must not be missed
    # EXPECTED: intl_phone type reported for "+1-555-123-4567"
    result = scan_pii("+1-555-123-4567", patterns=["intl_phone"])
    assert len(result) == 1
    assert result[0].type == "intl_phone"
    assert result[0].matched_text == "+1-555-123-4567"


def test_intl_phone_with_country() -> None:
    # SCENARIO: UK number with country code +44 and spaces (London area code)
    # WHY: Multi-digit country codes and space separators must be supported
    # EXPECTED: intl_phone detected in "+44 20 7946 0958"
    result = scan_pii("+44 20 7946 0958", patterns=["intl_phone"])
    assert len(result) >= 1
    assert any(m.type == "intl_phone" for m in result)


# ---------------------------------------------------------------------------
# mac_address
# ---------------------------------------------------------------------------

def test_mac_address_detected() -> None:
    # SCENARIO: Standard MAC address with colon separators (uppercase hex)
    # WHY: Colon-separated MACs are the canonical format in most tools
    # EXPECTED: mac_address type detected for "AA:BB:CC:DD:EE:FF"
    result = scan_pii("Device AA:BB:CC:DD:EE:FF connected.", patterns=["mac_address"])
    assert len(result) == 1
    assert result[0].type == "mac_address"
    assert result[0].matched_text == "AA:BB:CC:DD:EE:FF"


def test_mac_address_dashes() -> None:
    # SCENARIO: MAC address using dash separators (Windows ifconfig style)
    # WHY: Windows frequently presents MACs with dashes; pattern must match both
    # EXPECTED: mac_address detected for "AA-BB-CC-DD-EE-FF"
    result = scan_pii("mac=AA-BB-CC-DD-EE-FF", patterns=["mac_address"])
    assert len(result) == 1
    assert result[0].type == "mac_address"
    assert result[0].matched_text == "AA-BB-CC-DD-EE-FF"


# ---------------------------------------------------------------------------
# bitcoin_address
# ---------------------------------------------------------------------------

def test_bitcoin_address() -> None:
    # SCENARIO: Valid legacy Bitcoin P2PKH address starting with "1"
    # WHY: Bitcoin addresses in logs or data indicate financial PII
    # EXPECTED: bitcoin_address type detected
    # Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf (Satoshi genesis block address)
    btc = "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf"
    result = scan_pii(f"Send to {btc}", patterns=["bitcoin_address"])
    assert len(result) == 1
    assert result[0].type == "bitcoin_address"
    assert result[0].matched_text == btc


# ---------------------------------------------------------------------------
# ethereum_address
# ---------------------------------------------------------------------------

def test_ethereum_address() -> None:
    # SCENARIO: Canonical Ethereum address with "0x" prefix and 40 hex chars
    # WHY: ETH addresses in training data reveal wallet ownership (financial PII)
    # EXPECTED: ethereum_address type detected
    eth = "0x" + "a" * 40
    result = scan_pii(f"Transfer from {eth}", patterns=["ethereum_address"])
    assert len(result) == 1
    assert result[0].type == "ethereum_address"
    assert result[0].matched_text == eth


# ---------------------------------------------------------------------------
# Allowlist — suppression
# ---------------------------------------------------------------------------

def test_allowlist_suppresses() -> None:
    # SCENARIO: An email address that is in the allowlist should not be flagged
    # WHY: Known system/service addresses (e.g., noreply@example.com) should not
    #      cause false positives; allowlist is the mechanism for this
    # EXPECTED: No matches returned when the email is in the allowlist
    text = "Contact noreply@example.com for help."
    # Without allowlist: should detect email
    without = scan_pii(text, patterns=["email"])
    assert len(without) == 1

    # With allowlist: suppressed
    with_allowlist = scan_pii(text, patterns=["email"], allowlist=["noreply@example.com"])
    assert len(with_allowlist) == 0


def test_allowlist_partial() -> None:
    # SCENARIO: Two emails present; only one is in the allowlist
    # WHY: Allowlist must suppress exact matches only — other PII must still be caught
    # EXPECTED: The non-allowlisted email is still reported; the allowlisted one is not
    text = "admin@corp.com and secret@corp.com are in this text."
    result = scan_pii(text, patterns=["email"], allowlist=["admin@corp.com"])
    matched_texts = [m.matched_text for m in result]
    assert "admin@corp.com" not in matched_texts
    assert "secret@corp.com" in matched_texts
    assert len(result) == 1
