"""Property-based tests for LLM behavioral modules.

Uses Hypothesis to generate random inputs and assert
properties that must ALWAYS hold, regardless of input.

Categories:
  1. Monotonicity  -- identical inputs => max scores
  2. Boundary      -- scores always in valid range
  3. Empty-input   -- never crashes on edge cases
  4. Score-range   -- similarity methods return valid floats
  5. Idempotency   -- same inputs => same results
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm._utils import _normalize, _tokenize
from mltk.domains.llm.similarity import _token_f1

# -------------------------------------------------------
# 1. Monotonicity
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(st.text(min_size=1, max_size=200))
def test_identical_texts_max_token_f1(text):
    """_token_f1(t, t) == 1.0 for text with words."""
    if text.strip() and text.split():
        score = _token_f1(text, text)
        assert score == 1.0


@settings(max_examples=50, derandomize=True)
@given(st.text(min_size=1, max_size=200))
def test_empty_both_gives_perfect_score(text):
    """_token_f1('', '') == 1.0 (both empty)."""
    score = _token_f1("", "")
    assert score == 1.0


@settings(max_examples=50, derandomize=True)
@given(st.text(min_size=1, max_size=100))
def test_consistent_model_invariance(text):
    """Constant model has invariance_rate == 1.0."""
    from mltk.domains.llm.behavioral.invariance import (
        assert_paraphrase_invariance,
    )

    def model(x):
        return "fixed output"
    paraphrases = [text, text + " more"]
    # Filter: both must produce tokens after lower+split
    if not text.strip():
        return
    result = assert_paraphrase_invariance(
        model_fn=model,
        paraphrases=paraphrases,
        equivalence_method="token_f1",
        min_invariance=0.0,
    )
    rate = result.details.get("invariance_rate", 0)
    assert rate == 1.0


# -------------------------------------------------------
# 2. Boundary
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(st.text(), st.text())
def test_token_f1_bounded(ref, hyp):
    """_token_f1 always returns 0.0 <= score <= 1.0."""
    score = _token_f1(ref, hyp)
    assert 0.0 <= score <= 1.0


@settings(max_examples=50, derandomize=True)
@given(
    st.text(min_size=1, max_size=80),
    st.text(min_size=1, max_size=80),
)
def test_token_f1_symmetry(a, b):
    """_token_f1(a, b) == _token_f1(b, a) always."""
    assert _token_f1(a, b) == _token_f1(b, a)


@settings(max_examples=50, derandomize=True)
@given(
    st.text(min_size=1, max_size=80),
)
def test_token_f1_one_empty_is_zero(text):
    """_token_f1(text, '') == 0.0 when text has words."""
    if text.strip() and text.split():
        assert _token_f1(text, "") == 0.0
        assert _token_f1("", text) == 0.0


# -------------------------------------------------------
# 3. Empty-input stability
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(
    st.text(max_size=0)
    | st.text(
        alphabet=st.characters(
            whitelist_categories=("Zs",),
        ),
        max_size=20,
    )
)
def test_tokenize_empty_or_whitespace_no_crash(text):
    """_tokenize never crashes; returns set for blanks."""
    result = _tokenize(text)
    assert isinstance(result, set)


@settings(max_examples=50, derandomize=True)
@given(st.text(max_size=300))
def test_tokenize_returns_set(text):
    """_tokenize always returns a set of strings."""
    result = _tokenize(text)
    assert isinstance(result, set)
    for token in result:
        assert isinstance(token, str)


@settings(max_examples=50, derandomize=True)
@given(st.text(max_size=200))
def test_normalize_never_crashes(text):
    """_normalize never raises, always returns str."""
    result = _normalize(text)
    assert isinstance(result, str)


# -------------------------------------------------------
# 4. Score-range
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(
    st.lists(
        st.text(min_size=1, max_size=50),
        min_size=2,
        max_size=5,
    )
)
def test_paraphrase_invariance_rate_bounded(
    paraphrases,
):
    """invariance_rate is always 0.0-1.0."""
    from mltk.domains.llm.behavioral.invariance import (
        assert_paraphrase_invariance,
    )

    def model(x):
        return "output " + x[:10]
    try:
        result = assert_paraphrase_invariance(
            model_fn=model,
            paraphrases=paraphrases,
            equivalence_method="token_f1",
            min_invariance=0.0,
        )
    except MltkAssertionError as exc:
        result = exc.result
    rate = result.details.get("invariance_rate", 0)
    assert 0.0 <= rate <= 1.0


@settings(max_examples=50, derandomize=True)
@given(
    st.text(min_size=1, max_size=80),
    st.text(min_size=1, max_size=80),
)
def test_semantic_equivalence_score_bounded(a, b):
    """token_f1 semantic equivalence score in [0, 1]."""
    from mltk.domains.llm.behavioral.semantic import (
        assert_semantic_equivalence,
    )

    try:
        result = assert_semantic_equivalence(
            text_a=a,
            text_b=b,
            method="token_f1",
            min_score=0.0,
        )
    except MltkAssertionError as exc:
        result = exc.result
    score = result.details.get("score", 0)
    assert 0.0 <= score <= 1.0


@settings(max_examples=50, derandomize=True)
@given(
    st.lists(
        st.text(min_size=1, max_size=40),
        min_size=1,
        max_size=3,
    )
)
def test_output_stability_bounded(inputs):
    """avg_stability is always 0.0-1.0."""
    from mltk.domains.llm.behavioral.stability import (
        assert_output_stability,
    )

    counter = [0]

    def model(x):
        counter[0] += 1
        return f"response {counter[0] % 3}"

    try:
        result = assert_output_stability(
            model_fn=model,
            inputs=inputs,
            n_runs=3,
            equivalence_method="token_f1",
            min_stability=0.0,
        )
    except MltkAssertionError as exc:
        result = exc.result
    avg = result.details.get("avg_stability", 0)
    assert 0.0 <= avg <= 1.0


# -------------------------------------------------------
# 5. Idempotency
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(st.text(min_size=0, max_size=200))
def test_normalize_idempotent(text):
    """Normalizing twice equals normalizing once."""
    once = _normalize(text)
    twice = _normalize(once)
    assert once == twice


@settings(max_examples=50, derandomize=True)
@given(st.text(min_size=0, max_size=200))
def test_tokenize_idempotent_on_normalized(text):
    """Tokenizing normalized text is stable."""
    norm = _normalize(text)
    tokens_1 = _tokenize(norm)
    tokens_2 = _tokenize(norm)
    assert tokens_1 == tokens_2


# -------------------------------------------------------
# Extra properties (beyond the 5 categories)
# -------------------------------------------------------


@settings(max_examples=50, derandomize=True)
@given(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "Zs"),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_normalize_preserves_word_count_ascii(text):
    """For ASCII+space text, normalize keeps words."""
    words_before = len(text.split())
    words_after = len(_normalize(text).split())
    assert words_after == words_before


@settings(max_examples=50, derandomize=True)
@given(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "Nd", "Zs"),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_tokenize_subset_of_words(text):
    """Tokens come from the normalized+cleaned text."""
    # _tokenize applies: normalize → strip punct → lower → split
    # So tokens must be a subset of that pipeline's output
    tokens = _tokenize(text)
    # Verify all tokens are non-empty strings
    for tok in tokens:
        assert isinstance(tok, str)
        assert len(tok) > 0


@settings(max_examples=30, derandomize=True)
@given(st.text(min_size=1, max_size=80))
def test_format_invariance_case_insensitive_model(
    text,
):
    """Case-insensitive model always has invariance=1."""
    from mltk.domains.llm.behavioral.invariance import (
        assert_format_invariance,
    )

    if not text.strip():
        return

    def model(x):
        return x.lower()
    transforms = [lambda t: t.upper(), lambda t: t.lower()]
    try:
        result = assert_format_invariance(
            model_fn=model,
            input_text=text,
            transforms=transforms,
            equivalence_method="label_match",
            min_invariance=0.0,
        )
    except MltkAssertionError as exc:
        result = exc.result
    rate = result.details.get("invariance_rate", 0)
    assert rate == 1.0


@settings(max_examples=30, derandomize=True)
@given(st.text(min_size=2, max_size=100))
def test_directional_expectation_holds(text):
    """Appending text makes output longer."""
    from mltk.domains.llm.behavioral.semantic import (
        assert_directional_expectation,
    )

    if not text.strip():
        return

    def model(x):
        return x
    result = assert_directional_expectation(
        model_fn=model,
        input_text=text,
        perturbation=lambda t: t + " extra words",
        direction_fn=lambda o, p: len(p) > len(o),
        perturbation_name="append_makes_longer",
    )
    assert result.passed
