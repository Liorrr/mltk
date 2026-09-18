"""Unicode attack detection for LLM safety testing.

Detects zero-width invisible characters, bidirectional override controls
(Trojan Source / CVE-2021-42574), and homoglyph tokens that can be used to
bypass filters or deceive readers.

Homoglyphs come in two shapes, and they need different evidence.

A *mixed-script* spoof carries its own: ASCII Latin next to Cyrillic **inside
one word** is never accidental.  The qualifier is load-bearing, because a
token is ``\\S+`` rather than a word — a URL, a parenthesised formula and (CJK
writes no spaces) an entire Japanese sentence all arrive as one token.  So the
rule looks for the mixture within a single word, and requires that word to be
Latin script or a confusable partner throughout; otherwise ordinary typography
like ``設定はＯＮにしてWiFiを有効化する`` would convict.

A word ends at punctuation **and** at a change of script family.  Both halves
matter, and in opposite directions: without the purity rule the CJK sentence
above is a false positive, and without the script boundary ``ログインはpаypalへ``
is a false *negative* — the same absence of spaces that makes CJK space-free
makes it punctuation-free between words, so the spoof would hide inside its
neighbours.

A *single-script* spoof carries no evidence at all: a token written entirely
in Cyrillic or Greek letters that are Latin twins is indistinguishable, in
isolation, from an ordinary word in that language.  Following UTS #39, which
resolves whole-script confusables against a script context rather than per
token, that rule is gated on the script of the letters surrounding the token.
Mathematical alphanumerics are the exception the gate cannot serve — they have
no native prose, so Latin context is guaranteed for them by construction — and
are therefore opt-in.  See ``single_script_spoofs`` in
:func:`detect_unicode_attacks`.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# Explicit invisible characters to flag as zero-width attacks regardless of
# Unicode category. Defensive list covering all six common invisible chars.
_ZW_EXPLICIT: frozenset[str] = frozenset({
    chr(0x200B),  # ZERO WIDTH SPACE
    chr(0x200C),  # ZERO WIDTH NON-JOINER
    chr(0x200D),  # ZERO WIDTH JOINER
    chr(0xFEFF),  # ZERO WIDTH NO-BREAK SPACE / BOM
    chr(0x2060),  # WORD JOINER
    chr(0x00AD),  # SOFT HYPHEN
})

# Whitespace that must NOT be flagged even when it falls in the Cf category
_SAFE_WHITESPACE: frozenset[str] = frozenset({"\n", "\r", "\t"})

# Bidi direction controls implicated in Trojan Source (CVE-2021-42574).
# Exactly 9 codepoints — ordinary RTL *letters* (Hebrew, Arabic) are NOT here.
_BIDI_CONTROLS: frozenset[str] = frozenset({
    chr(0x202A),  # LEFT-TO-RIGHT EMBEDDING
    chr(0x202B),  # RIGHT-TO-LEFT EMBEDDING
    chr(0x202C),  # POP DIRECTIONAL FORMATTING
    chr(0x202D),  # LEFT-TO-RIGHT OVERRIDE
    chr(0x202E),  # RIGHT-TO-LEFT OVERRIDE
    chr(0x2066),  # LEFT-TO-RIGHT ISOLATE
    chr(0x2067),  # RIGHT-TO-LEFT ISOLATE
    chr(0x2068),  # FIRST STRONG ISOLATE
    chr(0x2069),  # POP DIRECTIONAL ISOLATE
})

# Cf-category chars that are LEGITIMATE in real text (Arabic/Syriac/Kaithi number
# and ayah marks). Excluded from the zero-width net so normal Arabic/Syriac content
# is not flagged as an attack (review finding: avoids wrong CRITICAL fail).
_LEGIT_CF: frozenset[str] = frozenset(
    chr(cp)
    for cp in (
        0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605,  # Arabic number signs
        0x06DD,  # Arabic end of ayah
        0x070F,  # Syriac abbreviation mark
        0x08E2,  # Arabic disputed end of ayah
        0x110BD, 0x110CD,  # Kaithi number signs
    )
)

# Human-readable labels for failure message construction
_CAT_LABELS: dict[str, str] = {
    "zero_width": "zero-width",
    "bidi": "bidi override",
    "homoglyph": "homoglyph",
}

_TOKEN_RE = re.compile(r"\S+")


def _is_zero_width(ch: str) -> bool:
    """Return True if *ch* is an invisible zero-width attack character.

    Bidi controls are explicitly excluded so they are reported only by the
    ``bidi`` category and never double-counted under ``zero_width``.
    """
    if ch in _SAFE_WHITESPACE or ch in _BIDI_CONTROLS or ch in _LEGIT_CF:
        return False
    return ch in _ZW_EXPLICIT or unicodedata.category(ch) == "Cf"


def _is_ascii_latin(ch: str) -> bool:
    """Return True if *ch* is an ASCII Latin letter [A-Za-z]."""
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def _is_latin_script(ch: str) -> bool:
    """Return True if *ch* is an ordinary Latin-script letter.

    Wider than :func:`_is_ascii_latin`: accented and extended Latin count too,
    so Vietnamese, Polish or Turkish prose reads as Latin context rather than
    as a competing script.  Deliberately excludes the fullwidth and
    mathematical-alphanumeric forms — those are Latin script on paper but are
    themselves spoof vectors, so they must never vouch for a neighbour.
    """
    cp = ord(ch)
    return (
        ("A" <= ch <= "Z")
        or ("a" <= ch <= "z")
        or 0x00C0 <= cp <= 0x024F  # Latin-1 Supplement + Extended-A/B
        or 0x1E00 <= cp <= 0x1EFF  # Latin Extended Additional (Vietnamese)
    )


# Cyrillic letters that are visual twins of ASCII Latin (phishing set).
# Palochka U+04CF stands in for Latin l. И/Й/etc. are *not* here so real
# words such as КИЕВ stay unflagged.
_CYRILLIC_LOOKALIKES: frozenset[str] = frozenset(
    "аеорсухіјАВЕКМНОРСТХІ" + chr(0x04CF)
)

# Greek letters that are visual twins of ASCII Latin. Used only for
# whole-token spoofs — mixed Latin+Greek scientific text is not flagged.
_GREEK_LOOKALIKES: frozenset[str] = frozenset("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧαοικνυηρχτ")


def _is_fullwidth_latin(ch: str) -> bool:
    """Fullwidth Latin letters (U+FF21–FF3A, U+FF41–FF5A)."""
    cp = ord(ch)
    return 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A


def _is_math_alphanumeric(ch: str) -> bool:
    """Mathematical Alphanumeric Symbols block (U+1D400–U+1D7FF)."""
    return 0x1D400 <= ord(ch) <= 0x1D7FF


def _is_cyrillic(ch: str) -> bool:
    """Cyrillic block (U+0400–U+04FF)."""
    return 0x0400 <= ord(ch) <= 0x04FF


def _is_confusable_script(ch: str) -> bool:
    """Return True if *ch* is a mixed-script partner of ASCII Latin.

    Cyrillic, fullwidth Latin, and mathematical alphanumeric symbols.
    Greek is excluded here: scientific text mixes Latin with Greek
    (e.g. ``α-helix``). Whole-word Greek spoofs are handled separately.
    """
    return (
        _is_cyrillic(ch)
        or _is_fullwidth_latin(ch)
        or _is_math_alphanumeric(ch)
    )


def _is_latin_family(ch: str) -> bool:
    """Latin script, or a script that can pass for it inside one word.

    The union of real Latin and every confusable partner, and deliberately not
    mode-dependent: this decides where a *word* ends, not whether a word is a
    spoof.  If mathematical alphanumerics left the family under ``"auto"``,
    ``pay𝐩𝐚𝐥`` would split into two words and the mixture would vanish
    before any rule could judge it.
    """
    return _is_latin_script(ch) or _is_cyrillic(ch) or _is_fullwidth_latin(ch) or (
        _is_math_alphanumeric(ch)
    )


def _alnum_runs(token: str) -> list[str]:
    """Split *token* into words.

    Tokens are ``\\S+``, so a token is not a word: a URL, a parenthesised
    formula and — because CJK writes no spaces — an entire Japanese sentence
    all arrive as one token.  Two things end a word here.

    Punctuation is the obvious one.  The other is a **change of script
    family**: ``これはpаypalです`` carries no separator at all, yet it is three
    words, and treating it as one would hide the spoof in the middle of it
    behind the purity test in :func:`_has_mixed_script_spoof`.  Splitting on
    punctuation alone is what made that a false negative — the same property
    that makes CJK space-free makes it punctuation-free between words.

    Digits are script-neutral and stay inside the current run, so ``раур1`` is
    not cut in half.
    """
    runs: list[str] = []
    current: list[str] = []
    family: bool | None = None
    for ch in token:
        if not ch.isalnum():
            if current:
                runs.append("".join(current))
                current = []
            family = None
            continue
        if ch.isalpha():
            ch_family = _is_latin_family(ch)
            if family is not None and ch_family != family:
                runs.append("".join(current))
                current = []
            family = ch_family
        current.append(ch)
    if current:
        runs.append("".join(current))
    return runs


# How many mathematical-alphanumeric letters must sit inside one Latin word
# before the mixture reads as a spoof rather than as notation.  ``𝐀x`` is a
# bold matrix applied to a plain variable and ``x𝐢`` is a basis component —
# one bold letter beside Latin is how mathematics is written.  Three or more
# replacing a chunk of an ASCII word (``pay𝐩𝐚𝐥``) has no notational reading.
# Same floor, and the same reasoning, as the three-letter minimum in
# :func:`_is_single_script_spoof`.
_MIN_MATH_RUN = 3


def _has_mixed_script_spoof(token: str, *, allow_math: bool = False) -> bool:
    """True if some word inside *token* mixes ASCII Latin with a partner script.

    Three conditions, and each one is load-bearing:

    * The Latin and the partner must sit in the **same word** — see
      :func:`_alnum_runs`, which ends a word at punctuation *and* at a change
      of script family.  A spoof is one word wearing another word's face, so
      ``pаypal`` qualifies, standalone and embedded in space-free CJK, while
      ``f(𝐱)`` — a Latin function name applied to a mathematical variable —
      does not, and neither does a URL whose path carries a Cyrillic segment.
    * Every letter of that word must be Latin script or a partner.  Fullwidth
      Latin is ordinary typography beside CJK (``ＰＤＦ``, ``ＯＮ``, ``ＣＰＵ``), and CJK
      has no spaces, so ``请把ＣＰＵ的temp参数设置好`` reaches this rule as one token.
      Requiring purity clears it while keeping ``payｐａｌ``, whose every letter
      is Latin or fullwidth.
    * Mathematical alphanumerics need :data:`_MIN_MATH_RUN` of them in the
      word, unless *allow_math*.  Unlike Cyrillic and fullwidth they carry no
      suspicion on their own: ``𝐀x`` is notation, and this rule has no context
      test that could tell notation from a spoof.  Length is the signal that
      is actually available — see :data:`_MIN_MATH_RUN`.

    Cyrillic gains nothing from the purity test on its own — no language
    writes Latin and Cyrillic inside one word — but it costs nothing either,
    so all partners run the same rule.

    Args:
        token: The whitespace-delimited token to classify.
        allow_math: Treat a single mathematical-alphanumeric letter as
            sufficient evidence.  Only ``single_script_spoofs="always"`` sets
            this, matching :func:`_is_single_script_spoof`.
    """
    for run in _alnum_runs(token):
        letters = [c for c in run if c.isalpha()]
        if not any(_is_ascii_latin(c) for c in letters):
            continue
        if not any(
            _is_cyrillic(c) or _is_fullwidth_latin(c) for c in letters
        ):
            n_math = sum(1 for c in letters if _is_math_alphanumeric(c))
            if n_math == 0 or not (allow_math or n_math >= _MIN_MATH_RUN):
                continue
        # Purity decides NOT to convict, so it stays generous about what
        # counts as Latin-family: math always qualifies here, whatever the
        # mode, or a formula would start convicting its neighbours.
        if all(_is_latin_script(c) or _is_confusable_script(c) for c in letters):
            return True
    return False


def _is_lookalike_letter(ch: str) -> bool:
    """Letter that can stand in for ASCII Latin in a whole-word spoof.

    Mathematical alphanumerics are deliberately absent.  Cyrillic, Greek and
    fullwidth letters each have native prose to sit in, which is what lets the
    script context clear an innocent word; U+1D400–U+1D7FF has none — it is
    the Unicode spelling of LaTeX-style notation and appears only inside Latin
    text.  Latin context is therefore guaranteed for it by construction, so
    gating on context would convict every legitimate formula rather than
    protect it.  They are reinstated by ``_is_single_script_spoof``'s
    ``allow_math``, which only ``single_script_spoofs="always"`` sets.

    The mixed-script rule reaches the same conclusion by a different route:
    it has no context test at all, so it leans on length instead — see
    :data:`_MIN_MATH_RUN`.
    """
    return (
        ch in _CYRILLIC_LOOKALIKES
        or ch in _GREEK_LOOKALIKES
        or _is_fullwidth_latin(ch)
    )


def _is_single_script_spoof(token: str, *, allow_math: bool = False) -> bool:
    """True if every letter of *token* is a Latin twin and none is ASCII Latin.

    This is a *candidate* test, not a verdict.  "Drawn entirely from the
    lookalike subset" is a property of any word written in that subset, and
    ordinary words are: МОСКВА, СССР, хор, και and τον all satisfy it.
    Whether a candidate is actually a spoof depends on the script it sits in
    — see :class:`_ScriptContext` and the ``single_script_spoofs`` argument of
    :func:`detect_unicode_attacks`.

    Requires at least three letters so short scientific tokens do not fire.
    Real Cyrillic/Greek that includes a non-twin letter (КИЕВ, привет)
    returns False outright.

    Args:
        token: The whitespace-delimited token to classify.
        allow_math: Count mathematical alphanumerics as Latin twins.  Only
            ``single_script_spoofs="always"`` sets this — see
            :func:`_is_lookalike_letter` for why context cannot decide them.
    """
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 3:
        return False
    if any(_is_ascii_latin(c) for c in letters):
        return False
    return all(
        _is_lookalike_letter(c) or (allow_math and _is_math_alphanumeric(c))
        for c in letters
    )


# Every separator ``str.splitlines()`` treats as a line break.  The token
# regex (``\\S+``) already ends a token at each of these, so the context index
# must end a line at each of them too — otherwise a prompt separated from a
# response by U+2028 (routine out of PDF, DOCX and JSON-origin extractors)
# silently lends its Latin context across what the corpus considers a line
# boundary, which is exactly what the boundary exists to prevent.
_LINE_BREAKS: frozenset[str] = frozenset(
    chr(cp)
    for cp in (
        0x000A,  # LINE FEED
        0x000B,  # LINE TABULATION
        0x000C,  # FORM FEED
        0x000D,  # CARRIAGE RETURN
        0x001C,  # FILE SEPARATOR
        0x001D,  # GROUP SEPARATOR
        0x001E,  # RECORD SEPARATOR
        0x0085,  # NEXT LINE
        0x2028,  # LINE SEPARATOR
        0x2029,  # PARAGRAPH SEPARATOR
    )
)

# How many letters on each side of a candidate token make up its script
# context.  ~24 letters per side is roughly four words either way: wide enough
# that a spoof planted in English prose sees plenty of Latin, narrow enough
# that a short foreign fragment inside a mostly-English document is judged by
# its own script rather than by the document average.
_CONTEXT_WINDOW_LETTERS = 24

# How lopsided the window must be before it counts as "predominantly Latin".
# A bare majority is not enough: text that genuinely interleaves two scripts
# on one line is ambiguous by construction, and a 2:1 margin keeps such text
# out of the rule while leaving ordinary Latin prose (where the margin is
# effectively infinite) comfortably inside it.
_LATIN_CONTEXT_MAJORITY = 2


class _ScriptContext:
    """Letter-script index over a text, used to judge single-script spoofs.

    A single-script confusable is only an attack *relative to its
    surroundings* — UTS #39 resolves whole-script confusables against a script
    context rather than per token.  ``МОСКВА`` inside Russian prose is a city;
    the same token dropped into an English paragraph is a spoof.  This class
    answers "are the letters around ``text[start:end]`` predominantly Latin
    script?" in O(log n) per token after an O(n) build.

    The context of a token is the letters on its own line, capped to
    :data:`_CONTEXT_WINDOW_LETTERS` on each side.  The line boundary matters:
    eval records routinely put an English prompt and a Russian or Japanese
    response in one string, and the prompt must not lend Latin context to the
    response.  Only letters are counted, so punctuation, digits and markup
    never dilute the window, and Latin must hold a
    :data:`_LATIN_CONTEXT_MAJORITY`-to-one margin over every other script —
    which is what keeps Cyrillic, Greek *and* CJK passages (the natural home
    of fullwidth Latin) from reading as Latin context.
    """

    __slots__ = ("_positions", "_lines", "_latin_prefix")

    def __init__(self, text: str) -> None:
        positions: list[int] = []
        lines: list[int] = []
        latin_prefix: list[int] = [0]
        line = 0
        for i, ch in enumerate(text):
            # CRLF bumps the counter twice; harmless, since the lookups only
            # need _lines monotonic and the skipped number holds no letters.
            if ch in _LINE_BREAKS:  # context never crosses a line break
                line += 1
                continue
            if not ch.isalpha():
                continue
            positions.append(i)
            lines.append(line)
            latin_prefix.append(latin_prefix[-1] + (1 if _is_latin_script(ch) else 0))
        self._positions = positions
        self._lines = lines
        self._latin_prefix = latin_prefix

    def _counts(self, lo: int, hi: int) -> tuple[int, int]:
        """Return ``(latin, non_latin)`` counts for the letter slice lo:hi."""
        if hi <= lo:
            return (0, 0)
        latin = self._latin_prefix[hi] - self._latin_prefix[lo]
        return (latin, (hi - lo) - latin)

    def is_latin_context(self, start: int, end: int) -> bool:
        """True if Latin script dominates the letters surrounding [start, end).

        The candidate token's own letters are excluded — a spoof must be
        justified by its neighbours, never by itself.  A token whose line
        holds no other letters (a bare token, or one alone on its line) has no
        context at all and returns False.
        """
        lo = bisect_left(self._positions, start)
        hi = bisect_left(self._positions, end)
        if hi <= lo:  # token has no letters; nothing to judge
            return False
        line = self._lines[lo]
        line_lo = bisect_left(self._lines, line)
        line_hi = bisect_right(self._lines, line)
        before = self._counts(max(line_lo, lo - _CONTEXT_WINDOW_LETTERS), lo)
        after = self._counts(hi, min(line_hi, hi + _CONTEXT_WINDOW_LETTERS))
        latin = before[0] + after[0]
        non_latin = before[1] + after[1]
        return latin > 0 and latin >= _LATIN_CONTEXT_MAJORITY * non_latin


# Accepted values for the ``single_script_spoofs`` argument.
_SINGLE_SCRIPT_MODES: frozenset[str] = frozenset({"auto", "always", "never"})


def _is_pictographic(ch: str) -> bool:
    """Return True if *ch* is an emoji / pictographic character.

    Used only to recognise legitimate emoji ZWJ sequences (e.g. 👨‍💻) so a
    zero-width joiner between two emoji is not mistaken for a smuggling attack.

    The variation-selector range (U+FE00-FE0F) is load-bearing: emoji VS16
    (U+FE0F) is what makes sequences like 👨‍❤️‍👨 register as emoji context.
    Do not drop it.
    """
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF  # SMP emoji: pictographs, emoticons, transport,
                                   # flags (1F1E6-1F1FF), skin-tone (1F3FB-1F3FF), cards
        or 0x2600 <= cp <= 0x27BF  # Misc symbols + Dingbats (❤ ✌ ✍ ...)
        or 0x2B00 <= cp <= 0x2BFF  # Misc symbols & arrows (⭐ ⬛ ...)
        or 0xFE00 <= cp <= 0xFE0F  # Variation selectors (incl. emoji VS16 U+FE0F)
    )


def _in_emoji_zwj_context(text: str, i: int) -> bool:
    """Return True if the ZWJ at ``text[i]`` joins two pictographs.

    A ZWJ (U+200D) is legitimate inside an emoji sequence but an attack when
    smuggled into ordinary text. It counts as emoji context only with a
    pictographic character on BOTH sides; a ZWJ at either end of the string,
    or with a non-emoji neighbour, is still treated as an attack.
    """
    return (
        0 < i < len(text) - 1
        and _is_pictographic(text[i - 1])
        and _is_pictographic(text[i + 1])
    )


def detect_unicode_attacks(
    text: str,
    checks: tuple[str, ...] = ("zero_width", "bidi", "homoglyph"),
    single_script_spoofs: str = "auto",
) -> dict:
    """Detect unicode-based attack patterns in text.

    Scans for zero-width invisible characters, bidi direction overrides, and
    homoglyph tokens.  Only the categories named in ``checks`` are scanned;
    others produce no key in the result.

    Two homoglyph rules run under the ``homoglyph`` category:

    * **mixed-script** — one word inside the token mixes ASCII Latin with
      Cyrillic, fullwidth Latin or mathematical alphanumerics (``pаypal``,
      ``payｐａｌ``, ``pay𝐩𝐚𝐥``).  Always on, in every mode: the mixture
      inside a word is itself the evidence, so no context is needed.  "Word"
      means one run of alphanumerics, ending at punctuation or at a change of
      script family, whose letters are all Latin or a partner — which is what
      separates a spoof from a URL with a Cyrillic path segment, from
      ``f(𝐱)``, and from CJK prose carrying a fullwidth acronym beside an
      ASCII one, while still catching ``pаypal`` embedded in space-free
      Japanese, Chinese, Korean or Thai.  Mathematical alphanumerics need
      three of them in the word before they count, since one bold letter
      beside a plain one (``𝐀x``) is notation rather than a spoof.
    * **single-script** — every letter of the token is a Latin twin and none
      is ASCII Latin (``раура``).  Gated by ``single_script_spoofs``, because
      ordinary Russian and Greek words (``МОСКВА``, ``και``) have exactly the
      same shape and are told apart only by the script around them.

    Args:
        text: Input text to analyse.
        checks: Tuple of category names to check.  Any subset of
            ``("zero_width", "bidi", "homoglyph")``.
        single_script_spoofs: How to apply the single-script rule.

            * ``"auto"`` (default) — flag a candidate only when the letters
              surrounding it, on the same line, are predominantly Latin
              script (accents included).  A spoof planted
              in English prose is flagged; the same token inside Russian,
              Greek or CJK text, or passed on its own with no context, is not.
              Mathematical alphanumerics are excluded from this mode
              entirely: U+1D400–U+1D7FF appears only inside Latin prose, so
              the context test can never clear a formula such as
              ``𝐚𝐛𝐜``.
            * ``"always"`` — flag every candidate regardless of context, and
              count mathematical alphanumerics as Latin twins.  For corpora
              known to be Latin-only and free of Unicode mathematics;
              produces false positives on any text genuinely written in
              Cyrillic or Greek, and on ordinary mathematical notation.
            * ``"never"`` — mixed-script rule only.

    Returns:
        Dict with one key per requested category plus ``"total"``.
        ``zero_width`` and ``bidi`` values are lists of
        ``{"codepoint": "U+XXXX", "index": N}`` dicts.
        ``homoglyph`` values are lists of
        ``{"token": "...", "index": N, "reason": "mixed_script"|"single_script"}``
        where *index* is the character offset in the original text.

    Raises:
        ValueError: If ``single_script_spoofs`` is not one of ``"auto"``,
            ``"always"`` or ``"never"``.

    Example:
        >>> text = "hello" + chr(0x200B) + "world"
        >>> detect_unicode_attacks(text, checks=("zero_width",))
        {'zero_width': [{'codepoint': 'U+200B', 'index': 5}], 'total': 1}
    """
    if single_script_spoofs not in _SINGLE_SCRIPT_MODES:
        allowed = ", ".join(sorted(_SINGLE_SCRIPT_MODES))
        raise ValueError(
            f"single_script_spoofs must be one of {allowed}; "
            f"got {single_script_spoofs!r}"
        )
    result: dict = {}
    total = 0

    if "zero_width" in checks:
        zw_findings = []
        for i, ch in enumerate(text):
            if not _is_zero_width(ch):
                continue
            # U+200D is legitimate between two emoji (e.g. 👨‍💻); flag it only
            # when NOT joining pictographs, i.e. smuggled into ordinary text.
            if ch == "‍" and _in_emoji_zwj_context(text, i):
                continue
            zw_findings.append({"codepoint": f"U+{ord(ch):04X}", "index": i})
        result["zero_width"] = zw_findings
        total += len(zw_findings)

    if "bidi" in checks:
        bidi_findings = [
            {"codepoint": f"U+{ord(ch):04X}", "index": i}
            for i, ch in enumerate(text)
            if ch in _BIDI_CONTROLS
        ]
        result["bidi"] = bidi_findings
        total += len(bidi_findings)

    if "homoglyph" in checks:
        hg_findings = []
        # Built lazily: only texts that actually contain a single-script
        # candidate pay the O(n) indexing cost.
        context: _ScriptContext | None = None
        allow_math = single_script_spoofs == "always"
        for m in _TOKEN_RE.finditer(text):
            token = m.group()
            if _has_mixed_script_spoof(token, allow_math=allow_math):
                reason = "mixed_script"
            elif (
                single_script_spoofs != "never"
                and _is_single_script_spoof(token, allow_math=allow_math)
            ):
                if single_script_spoofs == "auto":
                    if context is None:
                        context = _ScriptContext(text)
                    if not context.is_latin_context(m.start(), m.end()):
                        continue
                reason = "single_script"
            else:
                continue
            hg_findings.append(
                {"token": token, "index": m.start(), "reason": reason}
            )
        result["homoglyph"] = hg_findings
        total += len(hg_findings)

    result["total"] = total
    return result


@timed_assertion
def assert_no_unicode_attacks(
    text: str,
    *,
    checks: tuple[str, ...] = ("zero_width", "bidi", "homoglyph"),
    single_script_spoofs: str = "auto",
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that *text* contains no unicode-based attack characters.

    Checks for zero-width invisible characters, bidi direction overrides
    (Trojan Source / CVE-2021-42574), and homoglyph tokens that could bypass
    keyword filters or deceive readers.

    Args:
        text: Text to check for unicode attacks.
        checks: Categories to scan.  Any subset of
            ``("zero_width", "bidi", "homoglyph")``.
        single_script_spoofs: ``"auto"`` (default), ``"always"`` or
            ``"never"`` — see :func:`detect_unicode_attacks`.
        severity: ``CRITICAL`` (default) raises ``AssertionError`` on failure;
            ``WARNING``/``INFO`` records the finding without raising.

    Returns:
        TestResult with ``passed=True`` when no attacks are detected.

    Raises:
        ValueError: If ``single_script_spoofs`` is not a recognised mode.

    Note:
        Homoglyph detection flags (1) words that MIX ASCII Latin with
        Cyrillic, fullwidth Latin or mathematical alphanumeric symbols
        (``pаypal``, ``payｐａｌ``), and (2) whole-word single-script spoofs
        whose every letter is a Latin lookalike (``раура``).

        Rule 1 runs in every mode and needs no context, but the mixture must
        occur inside one word — one run of alphanumerics, ending at
        punctuation or at a change of script family, whose letters are all
        Latin or a partner.  Ordinary CJK typography puts a fullwidth acronym
        next to an ASCII word in a space-free sentence (設定はＯＮにしてWiFiを
        有効化する), and that is not a spoof; a Latin/Cyrillic spoof dropped into
        the same space-free prose (ログインはpаypalへ) is, and the script boundary
        is what keeps it visible.  Mathematical alphanumerics count here only
        three-or-more to a word: 𝐀x is a bold matrix applied to a plain
        variable, while pay𝐩𝐚𝐥 has no reading but a spoof.

        Rule 2 is context-gated: under the default
        ``single_script_spoofs="auto"`` it fires only where the surrounding
        letters are predominantly Latin script, so ordinary Russian and Greek
        words that happen to be built from Latin twins (МОСКВА, СССР, хор,
        και) are not flagged inside their own script, and neither is a bare
        token passed with no context.  Mathematical alphanumerics are left out
        of ``"auto"`` altogether — Unicode mathematics only ever appears in
        Latin prose, so a context test cannot distinguish 𝐚𝐛𝐜 from 𝐩𝐚𝐲𝐩𝐚𝐥;
        pass ``single_script_spoofs="always"`` (Latin-only corpus, no
        mathematics) to include them, which also drops rule 1's
        three-letter floor.  ``"never"`` runs the mixed-script rule
        alone.  Mixed Latin+Greek scientific text (α-helix, 5μm, kΩ) is never
        flagged.
        Variation-selector smuggling (U+FE0x) remains out of scope.
        ``zero_width`` excludes legitimate Arabic/Syriac/Kaithi format marks to
        avoid false positives on real RTL text, and excludes zero-width joiners
        (U+200D) that sit between two emoji/pictographic characters, since
        that is the legitimate emoji-ZWJ-sequence pattern (e.g. 👨‍💻).

    Example:
        >>> assert_no_unicode_attacks("Hello, world!")
        <TestResult name='llm.no_unicode_attacks' passed=True ...>
    """
    findings = detect_unicode_attacks(
        text, checks=checks, single_script_spoofs=single_script_spoofs
    )
    total = findings["total"]
    passed = total == 0

    checks_str = ", ".join(checks)
    if passed:
        message = f"No unicode attacks detected (checks: {checks_str})"
    else:
        parts = [
            f"{len(findings[cat])} {_CAT_LABELS.get(cat, cat)}"
            for cat in checks
            if cat in findings
        ]
        message = f"Found {', '.join(parts)}"

    # Build detail kwargs: aggregate count + up to 3-item sample per category
    detail_kwargs: dict = {
        "total_attacks": total,
        "checks": checks_str,
    }
    if "homoglyph" in checks:
        detail_kwargs["single_script_spoofs"] = single_script_spoofs
    for cat in ("zero_width", "bidi", "homoglyph"):
        if cat in findings and findings[cat]:
            detail_kwargs[f"{cat}_count"] = len(findings[cat])
            detail_kwargs[f"{cat}_sample"] = findings[cat][:3]

    return assert_true(
        passed,
        name="llm.no_unicode_attacks",
        message=message,
        severity=severity,
        **detail_kwargs,
    )
