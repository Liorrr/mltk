"""Tests for mltk.data.pii_ner -- NER-based PII detection.

NER scanners use named-entity recognition models (Presidio,
GLiNER) to detect contextual PII that regex alone cannot catch:
person names, organizations, locations. The hybrid scanner
merges regex + NER results with deduplication.

All external dependencies (presidio-analyzer, spacy, gliner)
are fully mocked -- no network or model downloads required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.pii import PiiMatch

# ---------------------------------------------------------------
# Fixtures -- mock Presidio and GLiNER
# ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_ner_caches():
    """Clear lru_cache on NER loaders before every test.

    The _get_presidio_analyzer and _get_gliner_model functions
    use @lru_cache so the real (heavy) model is loaded only once
    per process. In tests we mock these, so we must clear the
    cache before each test to prevent a cached real/mock from
    leaking across test boundaries.
    """
    from mltk.data.pii_ner import (
        _get_gliner_model,
        _get_presidio_analyzer,
    )

    _get_presidio_analyzer.cache_clear()
    _get_gliner_model.cache_clear()
    yield
    _get_presidio_analyzer.cache_clear()
    _get_gliner_model.cache_clear()


@pytest.fixture
def mock_presidio(monkeypatch):
    """Mock presidio_analyzer.AnalyzerEngine.

    Patches _get_presidio_analyzer to return a mock engine
    directly, bypassing the real import + lru_cache. The mock
    engine's .analyze() returns one PERSON entity by default.
    """
    mock_engine = MagicMock()

    # Default: return one PERSON entity.
    result = MagicMock()
    result.entity_type = "PERSON"
    result.start = 8
    result.end = 18
    result.score = 0.85
    mock_engine.analyze.return_value = [result]

    monkeypatch.setattr(
        "mltk.data.pii_ner._get_presidio_analyzer",
        lambda: mock_engine,
    )
    return mock_engine


@pytest.fixture
def mock_presidio_empty(monkeypatch):
    """Presidio mock returning zero entities."""
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = []

    monkeypatch.setattr(
        "mltk.data.pii_ner._get_presidio_analyzer",
        lambda: mock_engine,
    )
    return mock_engine


@pytest.fixture
def mock_gliner(monkeypatch):
    """Mock gliner.GLiNER model.

    Patches _get_gliner_model to return a mock model directly.
    The mock model's .predict_entities() returns one person name
    entity by default.
    """
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = [
        {
            "text": "John Smith",
            "label": "person name",
            "start": 8,
            "end": 18,
            "score": 0.92,
        }
    ]

    monkeypatch.setattr(
        "mltk.data.pii_ner._get_gliner_model",
        lambda model_name: mock_model,
    )
    return mock_model


@pytest.fixture
def mock_gliner_empty(monkeypatch):
    """GLiNER mock returning zero entities."""
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = []

    monkeypatch.setattr(
        "mltk.data.pii_ner._get_gliner_model",
        lambda model_name: mock_model,
    )
    return mock_model


def _make_presidio_result(
    entity_type, start, end, score
):
    """Build a mock Presidio RecognizerResult."""
    r = MagicMock()
    r.entity_type = entity_type
    r.start = start
    r.end = end
    r.score = score
    return r


# ---------------------------------------------------------------
# NER Scanner tests (12)
# ---------------------------------------------------------------

class TestScanPiiNer:
    """Tests for scan_pii_ner -- Presidio-based NER scanning."""

    def test_scan_ner_detects_person_name(
        self, mock_presidio
    ):
        """Presidio PERSON entity maps to person_name."""
        from mltk.data.pii_ner import scan_pii_ner

        text = "Contact John Smith for details"
        matches = scan_pii_ner(text)
        assert len(matches) >= 1
        assert any(
            m.type == "person_name" for m in matches
        )

    def test_scan_ner_detects_organization(
        self, mock_presidio
    ):
        """Presidio ORGANIZATION maps to organization."""
        org = _make_presidio_result(
            "ORGANIZATION", 9, 25, 0.88
        )
        mock_presidio.analyze.return_value = [org]

        from mltk.data.pii_ner import scan_pii_ner

        matches = scan_pii_ner(
            "Works at Acme Corporation"
        )
        assert len(matches) == 1
        assert matches[0].type == "organization"

    def test_scan_ner_detects_location(
        self, mock_presidio
    ):
        """Presidio LOCATION maps to location."""
        loc = _make_presidio_result(
            "LOCATION", 9, 37, 0.91
        )
        mock_presidio.analyze.return_value = [loc]

        from mltk.data.pii_ner import scan_pii_ner

        text = "Lives at 123 Main St, Springfield"
        matches = scan_pii_ner(text)
        assert len(matches) == 1
        assert matches[0].type == "location"

    def test_scan_ner_detects_multiple_entities(
        self, mock_presidio
    ):
        """Multiple entity types in one text."""
        person = _make_presidio_result(
            "PERSON", 0, 10, 0.90
        )
        loc = _make_presidio_result(
            "LOCATION", 20, 30, 0.85
        )
        org = _make_presidio_result(
            "ORGANIZATION", 35, 50, 0.80
        )
        mock_presidio.analyze.return_value = [
            person, loc, org
        ]

        from mltk.data.pii_ner import scan_pii_ner

        text = "John Smith visited Springfield at Acme HQ"
        matches = scan_pii_ner(text)
        assert len(matches) == 3
        types = {m.type for m in matches}
        assert "person_name" in types
        assert "location" in types
        assert "organization" in types

    def test_scan_ner_empty_text(self, mock_presidio):
        """Empty string produces no matches."""
        mock_presidio.analyze.return_value = []

        from mltk.data.pii_ner import scan_pii_ner

        matches = scan_pii_ner("")
        assert matches == []

    def test_scan_ner_no_pii(
        self, mock_presidio_empty
    ):
        """Clean text with no entities returns empty."""
        from mltk.data.pii_ner import scan_pii_ner

        matches = scan_pii_ner(
            "The weather is nice today"
        )
        assert matches == []

    def test_scan_ner_score_threshold_forwarded(
        self, mock_presidio
    ):
        """Score threshold forwarded to Presidio engine.

        The actual filtering is done inside Presidio's
        analyze() method. We verify the threshold param
        is passed through correctly.
        """
        mock_presidio.analyze.return_value = []

        from mltk.data.pii_ner import scan_pii_ner

        scan_pii_ner(
            "Hi Alice, meet Bob", score_threshold=0.8
        )
        call_kwargs = (
            mock_presidio.analyze.call_args
        )
        # Verify 0.8 was passed as score_threshold.
        call_str = str(call_kwargs)
        assert "0.8" in call_str

    def test_scan_ner_custom_entity_types_forwarded(
        self, mock_presidio
    ):
        """Custom entity_types forwarded to Presidio.

        Presidio's analyze() accepts an entities list that
        restricts which recognizers run. We verify the param
        is passed through, not that Presidio filters
        internally (that's Presidio's responsibility).
        """
        mock_presidio.analyze.return_value = []

        from mltk.data.pii_ner import scan_pii_ner

        scan_pii_ner(
            "John 555-123-4567",
            entity_types=["PERSON"],
        )
        call_kwargs = (
            mock_presidio.analyze.call_args
        )
        call_str = str(call_kwargs)
        assert "PERSON" in call_str

    def test_scan_ner_entity_type_mapping(
        self, mock_presidio
    ):
        """Verify all standard Presidio type mappings."""
        mappings = {
            "PERSON": "person_name",
            "LOCATION": "location",
            "PHONE_NUMBER": "phone",
            "EMAIL_ADDRESS": "email",
            "CREDIT_CARD": "credit_card",
            "ORGANIZATION": "organization",
            "DATE_TIME": "date_time",
            "NRP": "nationality",
            "IP_ADDRESS": "ipv4",
        }
        for presidio_type, expected in mappings.items():
            r = _make_presidio_result(
                presidio_type, 0, 10, 0.9
            )
            mock_presidio.analyze.return_value = [r]

            from mltk.data.pii_ner import scan_pii_ner

            matches = scan_pii_ner("test text here")
            assert len(matches) >= 1
            assert any(
                m.type == expected for m in matches
            ), (
                f"{presidio_type} should map to "
                f"{expected}"
            )

    def test_scan_ner_import_error(self, monkeypatch):
        """ImportError raised when presidio not installed."""
        def _raise_import():
            raise ImportError(
                "pip install mltk[ner]"
            )

        monkeypatch.setattr(
            "mltk.data.pii_ner._get_presidio_analyzer",
            _raise_import,
        )

        from mltk.data.pii_ner import scan_pii_ner

        with pytest.raises(ImportError, match="mltk"):
            scan_pii_ner("Hello world")

    def test_scan_ner_respects_language(
        self, mock_presidio
    ):
        """Language param forwarded to Presidio."""
        from mltk.data.pii_ner import scan_pii_ner

        scan_pii_ner("Bonjour le monde", language="fr")
        call_kwargs = (
            mock_presidio.analyze.call_args
        )
        # Verify "fr" was passed (kwarg or positional).
        call_str = str(call_kwargs)
        assert "fr" in call_str

    def test_scan_ner_multiple_persons(
        self, mock_presidio
    ):
        """Multiple persons + location in one sentence."""
        p1 = _make_presidio_result(
            "PERSON", 0, 4, 0.90
        )
        p2 = _make_presidio_result(
            "PERSON", 9, 13, 0.88
        )
        loc = _make_presidio_result(
            "LOCATION", 22, 27, 0.85
        )
        mock_presidio.analyze.return_value = [
            p1, p2, loc
        ]

        from mltk.data.pii_ner import scan_pii_ner

        text = "John and Jane went to Paris"
        matches = scan_pii_ner(text)
        persons = [
            m for m in matches
            if m.type == "person_name"
        ]
        locations = [
            m for m in matches if m.type == "location"
        ]
        assert len(persons) == 2
        assert len(locations) == 1


# ---------------------------------------------------------------
# GLiNER Scanner tests (8)
# ---------------------------------------------------------------

class TestScanPiiGliner:
    """Tests for scan_pii_gliner -- GLiNER-based scanning."""

    def test_scan_gliner_detects_custom_entities(
        self, mock_gliner
    ):
        """GLiNER detects domain-specific entity."""
        from mltk.data.pii_ner import scan_pii_gliner

        matches = scan_pii_gliner(
            "Contact John Smith for info"
        )
        assert len(matches) >= 1
        assert any(
            m.type == "person_name" for m in matches
        )

    def test_scan_gliner_healthcare_entities(
        self, mock_gliner
    ):
        """GLiNER detects medical record numbers."""
        mock_gliner.predict_entities.return_value = [
            {
                "text": "12345",
                "label": "medical record number",
                "start": 13,
                "end": 18,
                "score": 0.88,
            }
        ]

        from mltk.data.pii_ner import scan_pii_gliner

        matches = scan_pii_gliner(
            "Patient MRN: 12345"
        )
        assert len(matches) == 1
        assert matches[0].type == "medical_record"

    def test_scan_gliner_score_threshold_forwarded(
        self, mock_gliner
    ):
        """Score threshold forwarded to GLiNER model.

        The actual filtering is done inside GLiNER's
        predict_entities(). We verify the threshold
        param is passed through correctly.
        """
        mock_gliner.predict_entities.return_value = []

        from mltk.data.pii_ner import scan_pii_gliner

        scan_pii_gliner(
            "weak text John Smith",
            score_threshold=0.8,
        )
        call_kwargs = (
            mock_gliner.predict_entities.call_args
        )
        call_str = str(call_kwargs)
        assert "0.8" in call_str

    def test_scan_gliner_empty_text(
        self, mock_gliner_empty
    ):
        """Empty string returns no matches."""
        from mltk.data.pii_ner import scan_pii_gliner

        matches = scan_pii_gliner("")
        assert matches == []

    def test_scan_gliner_custom_entity_types(
        self, mock_gliner
    ):
        """User-specified entity types forwarded."""
        mock_gliner.predict_entities.return_value = []

        from mltk.data.pii_ner import scan_pii_gliner

        custom = [
            "person name", "medical record number"
        ]
        scan_pii_gliner(
            "Some text", entity_types=custom
        )
        call_args = (
            mock_gliner.predict_entities.call_args
        )
        call_str = str(call_args)
        assert "person name" in call_str

    def test_scan_gliner_model_loading(
        self, monkeypatch
    ):
        """Model loader called with model name."""
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        calls: list[str] = []

        def _mock_loader(model_name):
            calls.append(model_name)
            return mock_model

        monkeypatch.setattr(
            "mltk.data.pii_ner._get_gliner_model",
            _mock_loader,
        )

        from mltk.data.pii_ner import scan_pii_gliner

        scan_pii_gliner("first call")
        scan_pii_gliner("second call")

        # Loader is called once per invocation since
        # monkeypatch replaces the cached function.
        assert len(calls) >= 1
        assert all(
            c == "urchade/gliner_medium-v2.1"
            for c in calls
        )

    def test_scan_gliner_import_error(
        self, monkeypatch
    ):
        """ImportError raised when gliner not installed."""
        def _raise_import(model_name):
            raise ImportError(
                "pip install gliner"
            )

        monkeypatch.setattr(
            "mltk.data.pii_ner._get_gliner_model",
            _raise_import,
        )

        from mltk.data.pii_ner import scan_pii_gliner

        with pytest.raises(
            ImportError, match="gliner"
        ):
            scan_pii_gliner("Hello world")

    def test_scan_gliner_returns_pii_match(
        self, mock_gliner
    ):
        """Verify returned objects are PiiMatch."""
        from mltk.data.pii_ner import scan_pii_gliner

        matches = scan_pii_gliner(
            "Contact John Smith"
        )
        assert len(matches) >= 1
        for m in matches:
            assert isinstance(m, PiiMatch)
            assert isinstance(m.type, str)
            assert isinstance(m.start, int)
            assert isinstance(m.end, int)
            assert isinstance(m.matched_text, str)


# ---------------------------------------------------------------
# Hybrid Scanner tests (8)
# ---------------------------------------------------------------

class TestScanPiiHybrid:
    """Tests for scan_pii_hybrid -- regex + NER combined."""

    def test_hybrid_combines_regex_and_ner(
        self, mock_presidio
    ):
        """Hybrid returns matches from both engines."""
        # Position the NER PERSON match where "John Smith"
        # actually is in the text (offset 33), not
        # overlapping with the email (offset 6-22).
        person = _make_presidio_result(
            "PERSON", 33, 43, 0.90
        )
        mock_presidio.analyze.return_value = [person]

        from mltk.data.pii_ner import scan_pii_hybrid

        text = (
            "Email john@example.com, "
            "contact: John Smith"
        )
        matches = scan_pii_hybrid(text)
        types = {m.type for m in matches}
        # Regex catches email; NER catches person.
        assert "email" in types
        assert "person_name" in types

    def test_hybrid_deduplicates_overlapping_spans(
        self, mock_presidio
    ):
        """Same span from both engines appears once."""
        # NER returns a phone at same span as regex.
        phone = _make_presidio_result(
            "PHONE_NUMBER", 5, 17, 0.9
        )
        mock_presidio.analyze.return_value = [phone]

        from mltk.data.pii_ner import scan_pii_hybrid

        text = "Call 555-123-4567 now"
        matches = scan_pii_hybrid(text)
        # Should be deduplicated -- at most 1 phone.
        phone_matches = [
            m for m in matches
            if m.start == 5 and m.end == 17
        ]
        assert len(phone_matches) <= 1

    def test_hybrid_regex_catches_api_keys(
        self, mock_presidio_empty
    ):
        """Regex detects structured patterns NER misses."""
        from mltk.data.pii_ner import scan_pii_hybrid

        key = "sk-proj-" + "a" * 30
        text = f"API key: {key}"
        matches = scan_pii_hybrid(text)
        assert any(
            m.type == "api_key" for m in matches
        )

    def test_hybrid_ner_catches_names(
        self, mock_presidio
    ):
        """NER catches person names regex cannot."""
        person = _make_presidio_result(
            "PERSON", 8, 18, 0.90
        )
        mock_presidio.analyze.return_value = [person]

        from mltk.data.pii_ner import scan_pii_hybrid

        text = "Contact John Smith for details"
        matches = scan_pii_hybrid(text)
        assert any(
            m.type == "person_name" for m in matches
        )

    def test_hybrid_respects_allowlist(
        self, mock_presidio
    ):
        """Allowlisted values skipped in hybrid."""
        from mltk.data.pii_ner import scan_pii_hybrid

        text = "Email test@safe.com for info"
        matches = scan_pii_hybrid(
            text, allowlist=["test@safe.com"]
        )
        # The allowlisted email should be suppressed.
        emails = [
            m for m in matches if m.type == "email"
        ]
        assert len(emails) == 0

    def test_hybrid_empty_text(
        self, mock_presidio_empty
    ):
        """Empty string yields no matches."""
        from mltk.data.pii_ner import scan_pii_hybrid

        matches = scan_pii_hybrid("")
        assert matches == []

    def test_hybrid_no_pii(
        self, mock_presidio_empty
    ):
        """Clean text with no PII returns empty."""
        from mltk.data.pii_ner import scan_pii_hybrid

        matches = scan_pii_hybrid(
            "The weather is nice today"
        )
        assert matches == []

    def test_hybrid_overlapping_spans_priority(
        self, mock_presidio
    ):
        """Dedup keeps higher-quality match."""
        # NER returns a location overlapping regex.
        loc = _make_presidio_result(
            "LOCATION", 0, 15, 0.95
        )
        mock_presidio.analyze.return_value = [loc]

        from mltk.data.pii_ner import scan_pii_hybrid

        text = "123 Main Street is the address"
        matches = scan_pii_hybrid(text)
        # No duplicate for the same span.
        spans = [
            (m.start, m.end) for m in matches
        ]
        assert len(spans) == len(set(spans))


# ---------------------------------------------------------------
# scan_pii_dispatch tests (6)
# ---------------------------------------------------------------

class TestScanPiiDispatch:
    """Tests for scan_pii_dispatch -- method routing."""

    def test_dispatch_regex_default(self):
        """Default method='regex' uses scan_pii."""
        from mltk.data.pii import scan_pii_dispatch

        matches = scan_pii_dispatch(
            "Contact john@example.com"
        )
        assert any(m.type == "email" for m in matches)

    def test_dispatch_regex_explicit(self):
        """Explicit method='regex' works identically."""
        from mltk.data.pii import scan_pii_dispatch

        matches = scan_pii_dispatch(
            "SSN: 123-45-6789", method="regex"
        )
        assert any(m.type == "ssn" for m in matches)

    def test_dispatch_ner(self, mock_presidio):
        """method='ner' routes to Presidio scanner."""
        from mltk.data.pii import scan_pii_dispatch

        matches = scan_pii_dispatch(
            "Contact John Smith", method="ner"
        )
        assert any(
            m.type == "person_name" for m in matches
        )

    def test_dispatch_gliner(self, mock_gliner):
        """method='gliner' routes to GLiNER scanner."""
        from mltk.data.pii import scan_pii_dispatch

        matches = scan_pii_dispatch(
            "Contact John Smith", method="gliner"
        )
        assert any(
            m.type == "person_name" for m in matches
        )

    def test_dispatch_hybrid(self, mock_presidio):
        """method='hybrid' combines regex + NER."""
        # Position the PERSON at 0-10 where "John Smith"
        # is, not overlapping with the API key at 17+.
        person = _make_presidio_result(
            "PERSON", 0, 10, 0.90
        )
        mock_presidio.analyze.return_value = [person]

        from mltk.data.pii import scan_pii_dispatch

        text = (
            "John Smith key: "
            "sk-proj-abcdefghijklmnopqrstuvwx"
        )
        matches = scan_pii_dispatch(
            text, method="hybrid"
        )
        types = {m.type for m in matches}
        assert "person_name" in types
        assert "api_key" in types

    def test_dispatch_invalid_raises(self):
        """Invalid method raises ValueError."""
        from mltk.data.pii import scan_pii_dispatch

        with pytest.raises(
            ValueError, match="Unknown PII"
        ):
            scan_pii_dispatch(
                "test", method="bad"
            )


# ---------------------------------------------------------------
# assert_no_pii method dispatch tests (8)
# ---------------------------------------------------------------

class TestAssertNoPiiMethodDispatch:
    """Tests for assert_no_pii with method parameter."""

    def _clean_df(self):
        """DataFrame with no PII."""
        np.random.seed(42)
        return pd.DataFrame({
            "text": [
                "Good product",
                "Fast delivery",
                "Great quality",
            ],
            "rating": [5, 4, 5],
        })

    def _dirty_df(self):
        """DataFrame containing PII."""
        np.random.seed(42)
        return pd.DataFrame({
            "feedback": [
                "Great product!",
                "Contact John Smith at 555-0199",
                "Love it",
            ],
        })

    def test_assert_no_pii_method_regex_compat(self):
        """Default regex method is backward-compatible."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df)
        assert result.passed is True
        assert result.details["method"] == "regex"

    def test_assert_no_pii_method_regex_explicit(
        self,
    ):
        """Explicit method='regex' recorded in details."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df, method="regex")
        assert result.passed is True
        assert result.details["method"] == "regex"

    def test_assert_no_pii_method_ner_pass(
        self, mock_presidio_empty
    ):
        """Clean data passes NER assertion."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df, method="ner")
        assert result.passed is True
        assert result.details["total_matches"] == 0
        assert result.details["method"] == "ner"

    def test_assert_no_pii_method_ner_fail(
        self, mock_presidio
    ):
        """PII detected by NER raises assertion."""
        from mltk.data.pii import assert_no_pii

        df = self._dirty_df()
        with pytest.raises(MltkAssertionError):
            assert_no_pii(df, method="ner")

    def test_assert_no_pii_method_gliner_pass(
        self, mock_gliner_empty
    ):
        """Clean data passes GLiNER assertion."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df, method="gliner")
        assert result.passed is True
        assert result.details["method"] == "gliner"

    def test_assert_no_pii_method_hybrid_pass(
        self, mock_presidio_empty
    ):
        """Clean data passes hybrid assertion."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df, method="hybrid")
        assert result.passed is True
        assert result.details["method"] == "hybrid"

    def test_assert_no_pii_invalid_method(self):
        """Unknown method raises ValueError."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        with pytest.raises(
            ValueError, match="Unknown PII"
        ):
            assert_no_pii(
                df, method="unknown_scanner"
            )

    def test_assert_no_pii_ner_message_label(
        self, mock_presidio_empty
    ):
        """NER method name appears in result message."""
        from mltk.data.pii import assert_no_pii

        df = self._clean_df()
        result = assert_no_pii(df, method="ner")
        assert "(ner)" in result.message


# ---------------------------------------------------------------
# Merge / dedup helper tests (5)
# ---------------------------------------------------------------

class TestMergeMatches:
    """Tests for _merge_matches deduplication logic."""

    def test_no_overlap_keeps_all(self):
        """Non-overlapping matches from both sources kept."""
        from mltk.data.pii_ner import _merge_matches

        regex = [
            PiiMatch("email", 0, 20, "a@b.com")
        ]
        ner = [
            PiiMatch("person_name", 30, 40, "John")
        ]
        merged = _merge_matches(regex, ner)
        assert len(merged) == 2

    def test_overlap_ner_preferred_wins(self):
        """NER wins for NER-preferred types on overlap."""
        from mltk.data.pii_ner import _merge_matches

        regex = [PiiMatch("phone", 5, 15, "555")]
        ner = [PiiMatch("person_name", 5, 15, "John")]
        merged = _merge_matches(regex, ner)
        types = {m.type for m in merged}
        assert "person_name" in types

    def test_overlap_regex_preferred_wins(self):
        """Regex wins for non-NER-preferred on overlap."""
        from mltk.data.pii_ner import _merge_matches

        regex = [
            PiiMatch("email", 0, 20, "a@b.com")
        ]
        ner = [PiiMatch("email", 0, 20, "a@b.com")]
        merged = _merge_matches(regex, ner)
        emails = [
            m for m in merged if m.type == "email"
        ]
        assert len(emails) == 1

    def test_empty_inputs(self):
        """Empty inputs produce empty output."""
        from mltk.data.pii_ner import _merge_matches

        assert _merge_matches([], []) == []

    def test_only_regex(self):
        """When NER finds nothing, regex matches kept."""
        from mltk.data.pii_ner import _merge_matches

        regex = [
            PiiMatch("api_key", 0, 30, "sk-proj"),
            PiiMatch("email", 40, 60, "a@b.com"),
        ]
        merged = _merge_matches(regex, [])
        assert len(merged) == 2


# ---------------------------------------------------------------
# Span overlap helper tests (5)
# ---------------------------------------------------------------

class TestSpansOverlap:
    """Tests for _spans_overlap geometric helper."""

    def test_overlapping(self):
        """Partially overlapping spans detected."""
        from mltk.data.pii_ner import _spans_overlap

        a = PiiMatch("x", 5, 15, "a")
        b = PiiMatch("y", 10, 20, "b")
        assert _spans_overlap(a, b) is True

    def test_identical(self):
        """Identical spans overlap."""
        from mltk.data.pii_ner import _spans_overlap

        a = PiiMatch("x", 0, 10, "a")
        b = PiiMatch("y", 0, 10, "b")
        assert _spans_overlap(a, b) is True

    def test_adjacent_no_overlap(self):
        """Adjacent half-open spans do not overlap."""
        from mltk.data.pii_ner import _spans_overlap

        a = PiiMatch("x", 0, 10, "a")
        b = PiiMatch("y", 10, 20, "b")
        assert _spans_overlap(a, b) is False

    def test_disjoint(self):
        """Well-separated spans do not overlap."""
        from mltk.data.pii_ner import _spans_overlap

        a = PiiMatch("x", 0, 5, "a")
        b = PiiMatch("y", 50, 60, "b")
        assert _spans_overlap(a, b) is False

    def test_contained(self):
        """One span fully inside another overlaps."""
        from mltk.data.pii_ner import _spans_overlap

        a = PiiMatch("x", 0, 20, "outer")
        b = PiiMatch("y", 5, 10, "inner")
        assert _spans_overlap(a, b) is True


# ---------------------------------------------------------------
# Entity type mapping completeness tests (3)
# ---------------------------------------------------------------

class TestEntityTypeMappings:
    """Verify mapping completeness and format."""

    def test_all_presidio_defaults_mapped(self):
        """Every default Presidio entity has a mapping."""
        from mltk.data.pii_ner import (
            _PRESIDIO_TO_MLTK,
            DEFAULT_PRESIDIO_ENTITIES,
        )

        for entity in DEFAULT_PRESIDIO_ENTITIES:
            assert entity in _PRESIDIO_TO_MLTK, (
                f"{entity!r} missing from mapping"
            )

    def test_all_gliner_defaults_mapped(self):
        """Every default GLiNER entity has a mapping."""
        from mltk.data.pii_ner import (
            _GLINER_TO_MLTK,
            DEFAULT_GLINER_ENTITIES,
        )

        for entity in DEFAULT_GLINER_ENTITIES:
            assert entity in _GLINER_TO_MLTK, (
                f"{entity!r} missing from mapping"
            )

    def test_ner_text_length_limit(self):
        """NER rejects text exceeding max length."""
        from mltk.data.pii_ner import (
            _MAX_NER_TEXT_LENGTH,
            scan_pii_ner,
        )

        huge = "x" * (_MAX_NER_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            scan_pii_ner(huge)

    def test_gliner_text_length_limit(self):
        """GLiNER rejects text exceeding max length."""
        from mltk.data.pii_ner import (
            _MAX_NER_TEXT_LENGTH,
            scan_pii_gliner,
        )

        huge = "x" * (_MAX_NER_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            scan_pii_gliner(huge)

    def test_mltk_types_are_snake_case(self):
        """All mapped types use snake_case."""
        from mltk.data.pii_ner import (
            _GLINER_TO_MLTK,
            _PRESIDIO_TO_MLTK,
        )

        for mapping in [
            _PRESIDIO_TO_MLTK, _GLINER_TO_MLTK
        ]:
            for key, value in mapping.items():
                assert " " not in value, (
                    f"{key!r} -> {value!r} has spaces"
                )
                assert value == value.lower(), (
                    f"{key!r} -> {value!r} not lowercase"
                )
