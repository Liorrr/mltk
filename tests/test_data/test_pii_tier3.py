"""Tests for Sprint 23 Tier 3 PII patterns in mltk.data.pii.

Covers international PII patterns added in Sprint 23:
  - UK NHS Number (10-digit, MOD-11 checksum)
  - UK National Insurance Number / NINO (format-only: 2L+6D+1L)
  - Germany Steuer-Identifikationsnummer / Steuer-ID (structural validation)
  - India Aadhaar (12-digit, Verhoeff checksum)
  - India PAN (format-only: 5L+4D+1L)
  - assert_no_pii DataFrame integration with Tier 3 patterns

All checksum test values are mathematically verified:
  - NHS 9434765919: sum=299, 299%11=2, check=11-2=9, digits[9]=9 (valid)
  - Aadhaar 234123412346: Verhoeff check passes with D5/sigma tables
  - Steuer-ID 12345678913: body "1234567891" has digit-1 twice, digit-0 absent
"""

from __future__ import annotations

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.pii import (
    _validate_aadhaar,
    _validate_nhs,
    _validate_steuerid,
    assert_no_pii,
    scan_pii,
)


class TestUkNhs:
    """UK NHS Number detection and MOD-11 checksum validation."""

    def test_uk_nhs_valid(self) -> None:
        # SCENARIO: UK medical dataset contains a patient NHS number in a notes field.
        # WHY: Valid NHS numbers passing MOD-11 must be flagged as PII before training.
        # EXPECTED: scan_pii detects 9434765919 as uk_nhs.
        #   Verification: weights 10..2 × digits[0..8] = 299; 299%11=2; check=11-2=9=digits[9].
        matches = scan_pii("Patient NHS: 943 476 5919 admitted")
        types = [m.type for m in matches]
        assert "uk_nhs" in types, f"Expected uk_nhs match; got types: {types}"

    def test_uk_nhs_compact_format(self) -> None:
        # SCENARIO: NHS number stored without spaces (compact 10-digit form).
        # WHY: Both spaced and compact formats appear in real EHR exports.
        # EXPECTED: 9434765919 (no spaces) detected as uk_nhs.
        matches = scan_pii("NHS ID 9434765919 verified")
        types = [m.type for m in matches]
        assert "uk_nhs" in types, f"Expected uk_nhs for compact format; got types: {types}"

    def test_uk_nhs_invalid_checksum(self) -> None:
        # SCENARIO: A 10-digit number has the right shape but a wrong check digit.
        # WHY: Without MOD-11 validation this would be a false positive.
        # EXPECTED: 9434765910 (check digit 0, should be 9) not detected as uk_nhs.
        matches = scan_pii("Reference 9434765910 invalid")
        nhs_matches = [m for m in matches if m.type == "uk_nhs"]
        assert len(nhs_matches) == 0, (
            f"Expected no uk_nhs for wrong check digit; got: {nhs_matches}"
        )

    def test_validate_nhs_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_nhs helper.
        # WHY: Verifies MOD-11 logic independent of regex and scan_pii plumbing.
        # EXPECTED: Valid NHS returns True; tampered last digit returns False.
        assert _validate_nhs("9434765919") is True
        assert _validate_nhs("943 476 5919") is True   # spaces stripped
        assert _validate_nhs("9434765910") is False     # wrong check digit
        assert _validate_nhs("943476591") is False      # only 9 digits
        assert _validate_nhs("94347659190") is False    # 11 digits


class TestUkNino:
    """UK National Insurance Number (NINO) detection (format validation)."""

    def test_uk_nino_valid(self) -> None:
        # SCENARIO: UK payroll dataset contains employee NINOs in a free-text column.
        # WHY: NINO format (2L+6D+1L) is specific enough to be detected as PII.
        # EXPECTED: AB123456A detected as uk_nino.
        matches = scan_pii("Employee NINO: AB 12 34 56 A confirmed")
        types = [m.type for m in matches]
        assert "uk_nino" in types, f"Expected uk_nino match; got types: {types}"

    def test_uk_nino_compact(self) -> None:
        # SCENARIO: NINO stored without spaces in a compact format: AB123456A.
        # WHY: Both spaced and compact NINO formats appear in HMRC integrations.
        # EXPECTED: AB123456A (no spaces) detected as uk_nino.
        matches = scan_pii("National Insurance AB123456A on file")
        types = [m.type for m in matches]
        assert "uk_nino" in types, f"Expected uk_nino for compact format; got types: {types}"

    def test_uk_nino_invalid_prefix(self) -> None:
        # SCENARIO: A string matches the digit pattern but uses excluded prefix letters.
        # WHY: HMRC excludes D, F, I, Q, U, V as first letters -- these must not match.
        # EXPECTED: DA123456A (D is excluded from first position) not detected as uk_nino.
        matches = scan_pii("Code DA123456A processed")
        nino_matches = [m for m in matches if m.type == "uk_nino"]
        assert len(nino_matches) == 0, (
            f"Expected no uk_nino for excluded prefix D; got: {nino_matches}"
        )


class TestDeSteuerid:
    """Germany Steuer-ID detection and structural validation."""

    def test_de_steuerid_valid(self) -> None:
        # SCENARIO: German HR dataset contains employee tax IDs in a text column.
        # WHY: Valid Steuer-IDs must be flagged as PII before training.
        # EXPECTED: 12345678913 detected as de_steuerid.
        #   Verification: body "1234567891" -- digit-1 appears twice, digit-0 absent.
        matches = scan_pii("Steuer-ID: 12345678913 erfasst")
        types = [m.type for m in matches]
        assert "de_steuerid" in types, f"Expected de_steuerid match; got types: {types}"

    def test_de_steuerid_invalid_first_zero(self) -> None:
        # SCENARIO: An 11-digit number starts with 0 (structurally invalid Steuer-ID).
        # WHY: First digit 0 is forbidden; the validator must reject it.
        # EXPECTED: 02345678913 not detected as de_steuerid.
        matches = scan_pii("Code 02345678913 skipped")
        steuer_matches = [m for m in matches if m.type == "de_steuerid"]
        assert len(steuer_matches) == 0, (
            f"Expected no de_steuerid for leading-zero number; got: {steuer_matches}"
        )

    def test_de_steuerid_invalid_all_unique(self) -> None:
        # SCENARIO: 11-digit number where all first-10 digits are unique (no duplicate).
        # WHY: Structural rule requires exactly one duplicated digit in positions 1-10.
        # EXPECTED: 12345678904 not detected as de_steuerid (body "1234567890"
        #   has all 10 distinct digits -- zero duplicated, zero absent -> fails rules 2+3).
        matches = scan_pii("Reference 12345678904 logged")
        steuer_matches = [m for m in matches if m.type == "de_steuerid"]
        assert len(steuer_matches) == 0, (
            f"Expected no de_steuerid for all-unique body; got: {steuer_matches}"
        )

    def test_validate_steuerid_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_steuerid helper.
        # WHY: Verifies structural logic independent of regex and scan_pii plumbing.
        # EXPECTED: Valid structural ID returns True; edge cases return False.
        assert _validate_steuerid("12345678913") is True   # body: 1 twice, 0 absent
        assert _validate_steuerid("02345678913") is False  # first digit 0
        assert _validate_steuerid("12345678904") is False  # body all-unique, no duplicate
        assert _validate_steuerid("1234567891") is False   # only 10 digits
        assert _validate_steuerid("123456789130") is False  # 12 digits


class TestInAadhaar:
    """India Aadhaar number detection and Verhoeff checksum validation."""

    def test_in_aadhaar_valid(self) -> None:
        # SCENARIO: Indian ML dataset contains Aadhaar numbers in a user-info column.
        # WHY: Aadhaar is biometrically linked PII; leaking it violates India DPDP Act.
        # EXPECTED: 2341 2341 2346 detected as in_aadhaar.
        #   Verification: Verhoeff check on 234123412346 passes (c=0 after all 12 digits).
        matches = scan_pii("Aadhaar: 2341 2341 2346 verified")
        types = [m.type for m in matches]
        assert "in_aadhaar" in types, f"Expected in_aadhaar match; got types: {types}"

    def test_in_aadhaar_compact_format(self) -> None:
        # SCENARIO: Aadhaar stored without spaces in a compact 12-digit form.
        # WHY: Both spaced and compact forms appear in UIDAI API responses.
        # EXPECTED: 234123412346 (no spaces) detected as in_aadhaar.
        matches = scan_pii("UID 234123412346 registered")
        types = [m.type for m in matches]
        assert "in_aadhaar" in types, f"Expected in_aadhaar for compact format; got types: {types}"

    def test_in_aadhaar_invalid_verhoeff(self) -> None:
        # SCENARIO: A 12-digit number has the right shape but a wrong check digit.
        # WHY: Without Verhoeff validation this would be a false positive.
        # EXPECTED: 234123412345 (last digit 5 instead of correct 6) not detected.
        matches = scan_pii("Code 234123412345 invalid")
        aadhaar_matches = [m for m in matches if m.type == "in_aadhaar"]
        assert len(aadhaar_matches) == 0, (
            f"Expected no in_aadhaar for wrong Verhoeff check; got: {aadhaar_matches}"
        )

    def test_in_aadhaar_invalid_first_digit(self) -> None:
        # SCENARIO: A 12-digit number starts with 1 (UIDAI policy forbids 0 and 1).
        # WHY: UIDAI specification: first digit must be 2-9.
        # EXPECTED: Any number starting with 1 rejected even if Verhoeff passes.
        assert _validate_aadhaar("100000000000") is False
        assert _validate_aadhaar("000000000000") is False

    def test_validate_aadhaar_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_aadhaar helper.
        # WHY: Verifies Verhoeff logic independent of regex and scan_pii plumbing.
        # EXPECTED: 234123412346 returns True; tampered versions return False.
        assert _validate_aadhaar("234123412346") is True
        assert _validate_aadhaar("2341 2341 2346") is True   # spaces stripped
        assert _validate_aadhaar("234123412345") is False    # wrong check digit
        assert _validate_aadhaar("23412341234") is False     # only 11 digits
        assert _validate_aadhaar("1341234123461") is False   # 13 digits


class TestInPan:
    """India PAN (Permanent Account Number) detection (format validation)."""

    def test_in_pan_valid(self) -> None:
        # SCENARIO: Indian financial dataset has PAN numbers in a text column.
        # WHY: PAN is a government-issued tax ID; detecting it prevents PII leakage.
        # EXPECTED: ABCDE1234F detected as in_pan (5 uppercase letters, 4 digits, 1 letter).
        matches = scan_pii("PAN: ABCDE1234F on record")
        types = [m.type for m in matches]
        assert "in_pan" in types, f"Expected in_pan match; got types: {types}"

    def test_in_pan_invalid_format(self) -> None:
        # SCENARIO: A string resembles a PAN but has the wrong structure.
        # WHY: Wrong digit count or lowercase letters must not match.
        # EXPECTED: ABCDE123F (only 3 digits, missing one) not detected as in_pan.
        matches = scan_pii("Code ABCDE123F processed")
        pan_matches = [m for m in matches if m.type == "in_pan"]
        assert len(pan_matches) == 0, (
            f"Expected no in_pan for wrong digit count; got: {pan_matches}"
        )

    def test_in_pan_lowercase_rejected(self) -> None:
        # SCENARIO: PAN-like string contains lowercase letters.
        # WHY: PAN is always uppercase; lowercase must not trigger detection.
        # EXPECTED: abcde1234f not detected as in_pan (regex requires [A-Z]).
        matches = scan_pii("ref abcde1234f logged")
        pan_matches = [m for m in matches if m.type == "in_pan"]
        assert len(pan_matches) == 0, (
            f"Expected no in_pan for lowercase input; got: {pan_matches}"
        )


class TestAssertNoPiiTier3:
    """Integration tests for assert_no_pii with Tier 3 international PII patterns."""

    def test_assert_no_pii_tier3_nhs_in_column(self) -> None:
        # SCENARIO: UK medical training dataset has NHS numbers in a 'patient_notes' column.
        # WHY: assert_no_pii must catch NHS PII before data enters model training.
        # EXPECTED: MltkAssertionError raised with uk_nhs in matches_by_type.
        df = pd.DataFrame(
            {
                "patient_notes": [
                    "Patient NHS 9434765919 admitted Monday",
                    "No identifying information",
                ],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["uk_nhs"])
        result = exc.value.result
        assert result.details["total_matches"] >= 1
        assert "uk_nhs" in result.details["matches_by_type"]

    def test_assert_no_pii_tier3_aadhaar_in_column(self) -> None:
        # SCENARIO: Indian user dataset contains Aadhaar numbers in a free-text field.
        # WHY: Aadhaar is sensitive biometric PII; its presence must fail the assertion.
        # EXPECTED: MltkAssertionError raised with in_aadhaar in matches_by_type.
        df = pd.DataFrame(
            {
                "user_info": [
                    "UID 234123412346 verified",
                    "No PII here",
                ],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["in_aadhaar"])
        result = exc.value.result
        assert result.details["total_matches"] >= 1
        assert "in_aadhaar" in result.details["matches_by_type"]

    def test_assert_no_pii_tier3_mixed_international(self) -> None:
        # SCENARIO: Multi-country dataset has PAN in one column, NINO in another.
        # WHY: Multiple Tier 3 PII types must all be caught in a single assertion pass.
        # EXPECTED: MltkAssertionError, total_matches >= 2, both types present.
        df = pd.DataFrame(
            {
                "col_india": ["Tax ID ABCDE1234F registered"],
                "col_uk": ["NI number AB123456A on record"],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["in_pan", "uk_nino"])
        result = exc.value.result
        assert result.details["total_matches"] >= 2

    def test_assert_no_pii_tier3_clean_passes(self) -> None:
        # SCENARIO: Scrubbed dataset with no Tier 3 PII.
        # WHY: Clean data must pass the assertion without raising.
        # EXPECTED: assert_no_pii returns passed=True for all Tier 3 patterns.
        df = pd.DataFrame(
            {
                "text": [
                    "Product review: excellent quality",
                    "Delivery was fast and reliable",
                    "No personal data here",
                ],
            }
        )
        result = assert_no_pii(
            df,
            patterns=["uk_nhs", "uk_nino", "de_steuerid", "in_aadhaar", "in_pan"],
        )
        assert result.passed is True
