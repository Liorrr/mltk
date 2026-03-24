"""Tests for mltk.data.pii -- PII detection in ML training data.

PII tests verify that personally identifiable information is caught
before it enters model training. Each test validates a specific pattern
category against realistic examples.
"""

import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.pii import assert_no_pii, scan_pii


class TestScanPii:
    """Tests for scan_pii -- low-level text scanning."""

    def test_email_detected(self) -> None:
        """Finds email addresses in text.

        Scenario: Customer feedback contains "contact me at user@company.com".
        This PII must not enter training data.
        """
        matches = scan_pii("Please email john.doe@example.com for details")
        assert len(matches) >= 1
        assert any(m.type == "email" for m in matches)

    def test_phone_detected(self) -> None:
        """Finds US phone numbers in text.

        Scenario: Support tickets contain customer phone numbers.
        """
        matches = scan_pii("Call me at 555-123-4567 anytime")
        assert len(matches) >= 1
        assert any(m.type == "phone" for m in matches)

    def test_ssn_detected(self) -> None:
        """Finds Social Security Numbers.

        Scenario: Healthcare data accidentally contains SSNs in notes field.
        """
        matches = scan_pii("SSN: 123-45-6789")
        assert len(matches) >= 1
        assert any(m.type == "ssn" for m in matches)

    def test_credit_card_detected(self) -> None:
        """Finds 16-digit credit card numbers.

        Scenario: Payment processing logs contain card numbers.
        """
        matches = scan_pii("Card: 4111-1111-1111-1111")
        assert len(matches) >= 1
        assert any(m.type == "credit_card" for m in matches)

    def test_api_key_openai_detected(self) -> None:
        """Finds OpenAI API keys (sk-proj- prefix).

        Scenario: Developer accidentally pasted API key in a config
        that became part of training data.
        """
        matches = scan_pii("key: sk-proj-abcdefghijklmnopqrstuvwx")
        assert len(matches) >= 1
        assert any(m.type == "api_key" for m in matches)

    def test_api_key_aws_detected(self) -> None:
        """Finds AWS access key IDs (AKIA prefix).

        Scenario: Cloud credentials leaked into log files used for training.
        """
        matches = scan_pii("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
        assert len(matches) >= 1
        assert any(m.type == "api_key" for m in matches)

    def test_clean_text_no_matches(self) -> None:
        """No PII in clean text -- zero matches.

        Scenario: Normal product description with no personal data.
        """
        matches = scan_pii("This is a great product with excellent features")
        assert len(matches) == 0

    def test_custom_patterns(self) -> None:
        """Only scan for specified pattern categories.

        Scenario: You only care about API keys, not emails or phones.
        """
        text = "Email: test@test.com, Key: sk-proj-aaaabbbbccccddddeeeefffff"
        # Only check API keys
        matches = scan_pii(text, patterns=["api_key"])
        types = {m.type for m in matches}
        assert "api_key" in types
        assert "email" not in types


class TestAssertNoPii:
    """Tests for assert_no_pii -- DataFrame-level PII assertion."""

    def test_clean_dataframe_passes(self) -> None:
        """PASS: DataFrame with no PII in text columns.

        Scenario: Clean training data after PII scrubbing pipeline.
        """
        df = pd.DataFrame(
            {
                "text": ["Good product", "Fast delivery", "Great quality"],
                "rating": [5, 4, 5],
            }
        )
        result = assert_no_pii(df)
        assert result.passed is True

    def test_pii_in_column_detected(self) -> None:
        """FAIL: Email found in feedback column.

        Scenario: Customer wrote their email in the feedback text.
        This must be caught before training.
        """
        df = pd.DataFrame(
            {
                "feedback": [
                    "Great product!",
                    "Contact me at alice@example.com",
                    "Love it",
                ],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df)
        assert "PII" in str(exc.value)

    def test_subset_columns(self) -> None:
        """PASS: PII in unchecked columns is ignored.

        Scenario: The 'email' column legitimately contains emails (it's the
        user identifier). Only scan the 'notes' column for leaked PII.
        """
        df = pd.DataFrame(
            {
                "email": ["user@company.com"],
                "notes": ["No PII here"],
            }
        )
        result = assert_no_pii(df, columns=["notes"])
        assert result.passed is True

    def test_multiple_pii_types(self) -> None:
        """FAIL: Multiple PII types detected across columns.

        Scenario: Messy data dump contains both emails and phone numbers.
        """
        df = pd.DataFrame(
            {
                "col_a": ["Call 555-123-4567"],
                "col_b": ["Email test@test.com"],
            }
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_pii(df)
        result = exc.value.result
        assert result.details["total_matches"] >= 2
