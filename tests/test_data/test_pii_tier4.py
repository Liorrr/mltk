"""Tests for Sprint 28 Tier 4 EU PII patterns in mltk.data.pii.

Covers EU-specific PII patterns added in Sprint 28:
  - France NIR / numéro de sécurité sociale (15-digit, MOD-97 checksum)
  - Italy Codice Fiscale (16-char, format-only: 6L+2D+1L+2D+1L+3D+1L)
  - Spain DNI (8 digits + 1 letter, modulo-23 letter validation)
  - assert_no_pii DataFrame integration with Tier 4 EU patterns
  - Multi-pattern detection in a single scan

All checksum test values are mathematically verified:
  - France NIR 185012509123815:
      N = 1850125091238, N mod 97 = 82, key = 97 - 82 = 15 (valid)
  - Spain DNI 12345678Z:
      12345678 mod 23 = 14, "TRWAGMYFPDXBNJZSQVHLCKE"[14] = 'Z' (valid)
"""

from __future__ import annotations

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.pii import (
    _validate_dni,
    _validate_nir,
    assert_no_pii,
    scan_pii,
)

# ---------------------------------------------------------------------------
# Mathematically verified test constants
# ---------------------------------------------------------------------------

# France NIR: sex=1, year=85, month=01, dept+commune=25091, order=238, key=15
# N = 1850125091238; N % 97 = 82; key = 97 - 82 = 15
_VALID_FR_NIR = "185012509123815"

# France NIR with wrong check key (91 instead of correct 94)
_INVALID_FR_NIR_CHECKSUM = "185012509123891"

# Italy Codice Fiscale: RSSMRA85M01H501Z (canonical example, Rossi Mario)
_VALID_IT_CF = "RSSMRA85M01H501Z"

# Spain DNI: 12345678 % 23 = 14 → letter index 14 = 'Z'
_VALID_ES_DNI = "12345678Z"

# Spain DNI with wrong letter (A instead of Z)
_INVALID_ES_DNI_LETTER = "12345678A"


# ---------------------------------------------------------------------------
# France NIR tests
# ---------------------------------------------------------------------------


class TestFrNir:
    """France NIR (social security number) detection and MOD-97 checksum validation."""

    def test_fr_nir_valid(self) -> None:
        # SCENARIO: French HR dataset contains employee social security numbers in a text column.
        # WHY: Valid NIR numbers must be flagged as PII before data enters model training.
        # EXPECTED: 185012509123815 detected as fr_nir.
        #   Verification: N=1850125091238; 1850125091238 % 97 = 3; key = 97-3 = 94 = digits[13:].
        matches = scan_pii(f"Numéro SS: {_VALID_FR_NIR} enregistré")
        types = [m.type for m in matches]
        assert "fr_nir" in types, f"Expected fr_nir match; got types: {types}"

    def test_fr_nir_invalid_checksum(self) -> None:
        # SCENARIO: A 15-digit number has the correct NIR shape but a wrong check key.
        # WHY: Without MOD-97 validation this would produce false positives.
        # EXPECTED: 185012509123891 (check key 91, should be 94) not detected as fr_nir.
        matches = scan_pii(f"Référence {_INVALID_FR_NIR_CHECKSUM} invalide")
        nir_matches = [m for m in matches if m.type == "fr_nir"]
        assert len(nir_matches) == 0, (
            f"Expected no fr_nir for wrong check key; got: {nir_matches}"
        )

    def test_validate_nir_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_nir helper.
        # WHY: Verifies MOD-97 logic independent of regex and scan_pii plumbing.
        # EXPECTED: Valid NIR returns True; tampered check digits return False.
        assert _validate_nir(_VALID_FR_NIR) is True
        assert _validate_nir(_INVALID_FR_NIR_CHECKSUM) is False   # key 91 != 94
        assert _validate_nir("18501250912389") is False             # only 14 digits
        assert _validate_nir("1850125091238150") is False           # 16 digits
        # sex digit 3 (fails regex, but also bad key)
        assert _validate_nir("385012509123891") is False

    def test_fr_nir_invalid_month(self) -> None:
        # SCENARIO: A 15-digit number has month 13, which is outside 01-12.
        # WHY: The NIR regex constrains month to 01-12; invalid months must not match.
        # EXPECTED: 1851325091238XX (month=13) not detected even if check key were correct.
        matches = scan_pii("ID 185132509123894 processed")
        nir_matches = [m for m in matches if m.type == "fr_nir"]
        assert len(nir_matches) == 0, (
            f"Expected no fr_nir for invalid month 13; got: {nir_matches}"
        )


# ---------------------------------------------------------------------------
# Italy Codice Fiscale tests
# ---------------------------------------------------------------------------


class TestItCodiceFiscale:
    """Italy Codice Fiscale detection (format validation)."""

    def test_it_codice_fiscale_valid(self) -> None:
        # SCENARIO: Italian payroll dataset contains Codice Fiscale identifiers in a text column.
        # WHY: Codice Fiscale is a personal tax code and government-issued ID under GDPR scope.
        # EXPECTED: RSSMRA85M01H501Z detected as it_codice_fiscale.
        #   Format: 6L(RSSMRA) + 2D(85) + 1L(M) + 2D(01) + 1L(H) + 3D(501) + 1L(Z) = 16 chars.
        matches = scan_pii(f"Codice Fiscale: {_VALID_IT_CF} verificato")
        types = [m.type for m in matches]
        assert "it_codice_fiscale" in types, (
            f"Expected it_codice_fiscale match; got types: {types}"
        )

    def test_it_codice_fiscale_wrong_format(self) -> None:
        # SCENARIO: A string resembles a Codice Fiscale but has incorrect structure.
        # WHY: Fewer than 6 leading letters must not trigger detection.
        # EXPECTED: RSSMR85M01H501Z (only 5 leading letters instead of 6) not detected.
        matches = scan_pii("Code RSSMR85M01H501Z processed")
        cf_matches = [m for m in matches if m.type == "it_codice_fiscale"]
        assert len(cf_matches) == 0, (
            f"Expected no it_codice_fiscale for 5-letter prefix; got: {cf_matches}"
        )

    def test_it_codice_fiscale_lowercase_rejected(self) -> None:
        # SCENARIO: Codice Fiscale in lowercase (as sometimes appears in web form submissions).
        # WHY: The pattern requires uppercase letters; lowercase must not match.
        # EXPECTED: rssmra85m01h501z not detected as it_codice_fiscale.
        matches = scan_pii("ref rssmra85m01h501z logged")
        cf_matches = [m for m in matches if m.type == "it_codice_fiscale"]
        assert len(cf_matches) == 0, (
            f"Expected no it_codice_fiscale for lowercase; got: {cf_matches}"
        )


# ---------------------------------------------------------------------------
# Spain DNI tests
# ---------------------------------------------------------------------------


class TestEsDni:
    """Spain DNI detection and modulo-23 letter validation."""

    def test_es_dni_valid(self) -> None:
        # SCENARIO: Spanish HR dataset contains DNI numbers in a free-text employee column.
        # WHY: DNI is the primary Spanish national identity document; it is PII under GDPR.
        # EXPECTED: 12345678Z detected as es_dni.
        #   Verification: 12345678 % 23 = 14; "TRWAGMYFPDXBNJZSQVHLCKE"[14] = 'Z'.
        matches = scan_pii(f"DNI: {_VALID_ES_DNI} registrado")
        types = [m.type for m in matches]
        assert "es_dni" in types, f"Expected es_dni match; got types: {types}"

    def test_es_dni_invalid_letter(self) -> None:
        # SCENARIO: A DNI-shaped string has the wrong control letter.
        # WHY: Without modulo-23 validation this would be a false positive.
        # EXPECTED: 12345678A (letter A, should be Z) not detected as es_dni.
        matches = scan_pii(f"Reference {_INVALID_ES_DNI_LETTER} invalid")
        dni_matches = [m for m in matches if m.type == "es_dni"]
        assert len(dni_matches) == 0, (
            f"Expected no es_dni for wrong letter A (should be Z); got: {dni_matches}"
        )

    def test_validate_dni_unit(self) -> None:
        # SCENARIO: Direct unit test of the _validate_dni helper.
        # WHY: Verifies modulo-23 logic independent of regex and scan_pii plumbing.
        # EXPECTED: Valid DNI returns True; wrong letter or bad format returns False.
        assert _validate_dni(_VALID_ES_DNI) is True
        assert _validate_dni(_INVALID_ES_DNI_LETTER) is False  # A != Z
        assert _validate_dni("1234567Z") is False               # only 7 digits
        assert _validate_dni("123456789Z") is False             # 9 digits before letter
        assert _validate_dni("12345678z") is False              # lowercase letter


# ---------------------------------------------------------------------------
# DataFrame integration tests
# ---------------------------------------------------------------------------


class TestAssertNoPiiEu:
    """Integration tests for assert_no_pii with Tier 4 EU PII patterns."""

    def test_assert_no_pii_eu(self) -> None:
        # SCENARIO: French compliance dataset with NIR numbers in a 'employee_info' column.
        # WHY: assert_no_pii must catch EU-specific PII before data enters model training.
        # EXPECTED: MltkAssertionError raised with fr_nir in matches_by_type.
        df = pd.DataFrame(
            {
                "employee_info": [
                    f"SSN: {_VALID_FR_NIR} actif",
                    "Aucune information personnelle",
                ],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["fr_nir"])
        result = exc.value.result
        assert result.details["total_matches"] >= 1
        assert "fr_nir" in result.details["matches_by_type"]

    def test_multiple_eu_patterns(self) -> None:
        # SCENARIO: Multi-country EU dataset has French NIR, Italian CF, and Spanish DNI
        #           in separate columns of the same DataFrame.
        # WHY: All three EU patterns must be detected in a single assert_no_pii call.
        # EXPECTED: MltkAssertionError raised, total_matches >= 3, all three types present.
        df = pd.DataFrame(
            {
                "col_fr": [f"NIR {_VALID_FR_NIR} enregistré"],
                "col_it": [f"CF {_VALID_IT_CF} verificato"],
                "col_es": [f"DNI {_VALID_ES_DNI} registrado"],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df, patterns=["fr_nir", "it_codice_fiscale", "es_dni"])
        result = exc.value.result
        assert result.details["total_matches"] >= 3, (
            f"Expected >= 3 matches; got {result.details['total_matches']}"
        )
        assert "fr_nir" in result.details["matches_by_type"]
        assert "it_codice_fiscale" in result.details["matches_by_type"]
        assert "es_dni" in result.details["matches_by_type"]

    def test_assert_no_pii_eu_clean_passes(self) -> None:
        # SCENARIO: Scrubbed EU dataset with no Tier 4 PII patterns.
        # WHY: Clean data must pass all EU pattern assertions without raising.
        # EXPECTED: assert_no_pii returns passed=True for all three Tier 4 EU patterns.
        df = pd.DataFrame(
            {
                "text": [
                    "Produit revu: excellente qualité",
                    "Nessuna informazione personale",
                    "Sin datos personales aquí",
                ],
            }
        )
        result = assert_no_pii(
            df,
            patterns=["fr_nir", "it_codice_fiscale", "es_dni"],
        )
        assert result.passed is True
