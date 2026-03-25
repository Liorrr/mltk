"""Tests for Israel-specific PII patterns in mltk.data.pii -- Sprint 18.

Covers:
  - Teudat Zehut (Israeli national ID) with Luhn checksum validation
  - Israeli mobile and landline phone number patterns
  - IBAN with MOD-97 checksum validation
  - assert_no_pii DataFrame integration with Israeli PII

All TZ test values use the standard Israeli Luhn variant:
  multiply alternating digits by 1/2, cross-sum products >= 10, total % 10 == 0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.pii import _validate_iban, _validate_tz, assert_no_pii, scan_pii


class TestIsraelTz:
    """Teudat Zehut (Israeli national ID) detection and checksum validation."""

    def test_israel_tz_valid(self) -> None:
        # SCENARIO: Israeli ML dataset contains a national ID in a user record.
        # WHY: TZ numbers passing the Luhn variant must be flagged as PII.
        # EXPECTED: scan_pii detects the valid TZ and returns israel_tz match.
        # 025259680: 0+4+5+4+5+9+6+7+0 = 40 (divisible by 10 -> valid)
        matches = scan_pii("User ID: 025259680 registered today")
        types = [m.type for m in matches]
        assert "israel_tz" in types, f"Expected israel_tz match; got types: {types}"
        tz_match = next(m for m in matches if m.type == "israel_tz")
        assert tz_match.matched_text == "025259680"

    def test_israel_tz_invalid_checksum(self) -> None:
        # SCENARIO: A 9-digit number appears in data but is NOT a valid TZ.
        # WHY: Without checksum validation this would be a false positive.
        #      The validator must reject numbers that fail the Luhn variant.
        # EXPECTED: scan_pii does NOT return an israel_tz match for 025259681.
        # 025259681: same as above but last digit changed -> sum=41 -> invalid
        matches = scan_pii("Code: 025259681 processed")
        tz_matches = [m for m in matches if m.type == "israel_tz"]
        assert len(tz_matches) == 0, (
            f"Expected no israel_tz match for invalid checksum; got: {tz_matches}"
        )

    def test_israel_tz_wrong_length(self) -> None:
        # SCENARIO: An 8-digit or 10-digit number appears in text.
        # WHY: Teudat Zehut is strictly 9 digits; wrong lengths must not match.
        # EXPECTED: Neither 8-digit nor 10-digit number detected as israel_tz.
        matches_8 = scan_pii("Reference: 02525968 end")
        matches_10 = scan_pii("Reference: 0252596800 end")
        tz_8 = [m for m in matches_8 if m.type == "israel_tz"]
        tz_10 = [m for m in matches_10 if m.type == "israel_tz"]
        assert len(tz_8) == 0, f"8-digit number should not match; got: {tz_8}"
        assert len(tz_10) == 0, f"10-digit number should not match; got: {tz_10}"

    def test_israel_tz_dashed_format(self) -> None:
        # SCENARIO: TZ stored with dashes: 025-259-680 (common in old systems).
        # WHY: The pattern allows optional dashes; validator must strip them
        #      before running the checksum.
        # EXPECTED: Dashed valid TZ is detected as israel_tz.
        matches = scan_pii("ID: 025-259-680 confirmed")
        types = [m.type for m in matches]
        assert "israel_tz" in types, f"Expected israel_tz for dashed format; got: {types}"

    def test_validate_tz_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_tz helper.
        # WHY: Ensures the algorithm is correct independent of regex matching.
        # EXPECTED: Valid IDs return True; invalid return False; bad input False.
        assert _validate_tz("025259680") is True   # sum=40
        assert _validate_tz("025259681") is False  # sum=41
        assert _validate_tz("000000000") is True   # all zeros -> sum=0
        assert _validate_tz("12345678") is False   # 8 digits
        assert _validate_tz("123456789X") is False  # non-digit


class TestIsraelPhone:
    """Israeli mobile and landline phone number detection."""

    def test_israel_phone_mobile(self) -> None:
        # SCENARIO: User submitted a mobile number in a feedback form.
        # WHY: Israeli mobile numbers start with 05X; must be detected as PII.
        # EXPECTED: 0521234567 detected as israel_phone.
        matches = scan_pii("Call me at 0521234567 tomorrow")
        types = [m.type for m in matches]
        assert "israel_phone" in types, f"Expected israel_phone; got: {types}"

    def test_israel_phone_mobile_dashed(self) -> None:
        # SCENARIO: Mobile number formatted with dashes: 052-123-4567.
        # WHY: Dashes are optional in the pattern; both formats should match.
        # EXPECTED: Dashed mobile detected as israel_phone.
        matches = scan_pii("Reach me at 052-123-4567 please")
        types = [m.type for m in matches]
        assert "israel_phone" in types, f"Expected israel_phone for dashed mobile; got: {types}"

    def test_israel_phone_landline(self) -> None:
        # SCENARIO: Office landline stored in a contact record.
        # WHY: Israeli landlines start with 02-09 (non-05); must be caught.
        # EXPECTED: 0312345678 detected as israel_phone.
        matches = scan_pii("Office line: 0312345678 ext 5")
        types = [m.type for m in matches]
        assert "israel_phone" in types, f"Expected israel_phone for landline; got: {types}"

    def test_israel_phone_landline_dashed(self) -> None:
        # SCENARIO: Landline formatted with dashes: 03-123-4567.
        # WHY: Dashes are optional; both compact and formatted must match.
        # EXPECTED: 03-123-4567 detected as israel_phone.
        matches = scan_pii("Fax: 03-123-4567 available")
        types = [m.type for m in matches]
        assert "israel_phone" in types, f"Expected israel_phone for dashed landline; got: {types}"


class TestIbanValidation:
    """IBAN detection with MOD-97 checksum validation."""

    def test_iban_valid(self) -> None:
        # SCENARIO: Bank account number appears in a financial training dataset.
        # WHY: Valid IBANs passing MOD-97 must be flagged as PII.
        # EXPECTED: GB29NWBK60161331926819 detected as iban (canonical example).
        matches = scan_pii("Account: GB29NWBK60161331926819 ref")
        types = [m.type for m in matches]
        assert "iban" in types, f"Expected iban match; got types: {types}"

    def test_iban_invalid_checksum(self) -> None:
        # SCENARIO: A string has the IBAN shape but wrong check digits.
        # WHY: Without MOD-97 validation this would be a false positive.
        #      GB00NWBK60161331926819 has check digits 00 which fail MOD-97.
        # EXPECTED: scan_pii does NOT return an iban match for the malformed IBAN.
        matches = scan_pii("Bad ref: GB00NWBK60161331926819 invalid")
        iban_matches = [m for m in matches if m.type == "iban"]
        assert len(iban_matches) == 0, (
            f"Expected no iban match for invalid checksum; got: {iban_matches}"
        )

    def test_validate_iban_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_iban helper.
        # WHY: Verifies MOD-97 logic independent of regex and scan_pii plumbing.
        # EXPECTED: Valid IBANs return True; tampered ones return False.
        assert _validate_iban("GB29NWBK60161331926819") is True
        assert _validate_iban("DE89370400440532013000") is True   # German IBAN
        assert _validate_iban("GB00NWBK60161331926819") is False  # bad check digits
        assert _validate_iban("XX") is False                       # too short

    def test_iban_il_valid(self) -> None:
        # SCENARIO: Israeli IBAN (IL prefix) in a banking dataset.
        # WHY: Israeli IBANs are 23 chars; must be caught by the pattern + MOD-97.
        # EXPECTED: IL620108000000099999999 detected as iban.
        # IL62: rearrange -> 010800000009999999IL62
        # Verify: _validate_iban("IL620108000000099999999") should return True
        assert _validate_iban("IL620108000000099999999") is True
        matches = scan_pii("Bank: IL620108000000099999999 transfer")
        types = [m.type for m in matches]
        assert "iban" in types, f"Expected iban for Israeli IBAN; got: {types}"


class TestAssertNoPiiIsrael:
    """Integration tests for assert_no_pii with Israeli PII patterns."""

    def test_assert_no_pii_israel_tz_in_column(self) -> None:
        # SCENARIO: Training dataset has a 'user_id' column containing TZ numbers.
        # WHY: assert_no_pii must catch this before the data enters training.
        # EXPECTED: MltkAssertionError raised with israel_tz in matches_by_type.
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "national_id": ["025259680", "clean text here"],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["israel_tz"])
        result = exc.value.result
        assert result.details["total_matches"] >= 1
        assert "israel_tz" in result.details["matches_by_type"]

    def test_assert_no_pii_clean_dataframe_passes(self) -> None:
        # SCENARIO: Scrubbed dataset with no Israeli PII.
        # WHY: Clean data must pass the assertion without raising.
        # EXPECTED: assert_no_pii returns passed=True.
        df = pd.DataFrame(
            {
                "text": ["No ID here", "Just product feedback", "Great service"],
            }
        )
        result = assert_no_pii(df, patterns=["israel_tz", "israel_phone", "iban"])
        assert result.passed is True

    def test_assert_no_pii_mixed_israel_pii(self) -> None:
        # SCENARIO: Dataset has both a TZ and a mobile phone in different columns.
        # WHY: Multiple Israeli PII types must all be caught in one assertion pass.
        # EXPECTED: MltkAssertionError, total_matches >= 2.
        df = pd.DataFrame(
            {
                "col_tz": ["User: 025259680 registered"],
                "col_phone": ["Contact: 052-123-4567 available"],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["israel_tz", "israel_phone"])
        result = exc.value.result
        assert result.details["total_matches"] >= 2
