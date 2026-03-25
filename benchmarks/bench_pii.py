"""Benchmark PII scanning (scan_pii_fast) at varying text sizes and pattern counts."""

from __future__ import annotations

import time

from mltk._rust import RUST_AVAILABLE, scan_pii_fast

MODE = "RUST" if RUST_AVAILABLE else "PYTHON"

# Common PII patterns used in ML output audits
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("email", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("phone_us", r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit_card", r"\b(?:\d[ -]?){13,16}\b"),
    ("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("date_iso", r"\b\d{4}-\d{2}-\d{2}\b"),
]

# Base paragraph that naturally contains examples of each pattern
_BASE_PARAGRAPH = (
    "User alice@example.com called from 555-867-5309 on 2024-01-15. "
    "SSN on file: 123-45-6789. IP address: 192.168.1.100. "
    "Card ending in 4111 1111 1111 1111 was charged. "
    "Contact bob.jones@corp.io for details. "
    "Secondary phone: 800.555.0199 (toll-free).\n"
)

# Text sizes: repeat base paragraph to reach approximate character counts
TEXT_SIZES = [
    ("10 KB", _BASE_PARAGRAPH * 70),
    ("100 KB", _BASE_PARAGRAPH * 700),
    ("1 MB", _BASE_PARAGRAPH * 7_000),
]

# Pattern-count scenarios
PATTERN_SCENARIOS = [
    ("2 patterns", DEFAULT_PATTERNS[:2]),
    ("6 patterns", DEFAULT_PATTERNS),
    ("12 patterns", DEFAULT_PATTERNS * 2),  # doubled for stress test
]

HEADER = f"{'Scenario':<14} | {'Text size':<8} | {'chars':>10} | {'matches':>8} | {'ms':>10} | Mode"
SEP = "-" * len(HEADER)

print(f"\n{'PII Scan Benchmark':^{len(HEADER)}}")
print(SEP)
print(HEADER)
print(SEP)

for pat_label, patterns in PATTERN_SCENARIOS:
    for size_label, text in TEXT_SIZES:
        # Warm-up
        scan_pii_fast(text[:500], patterns)

        start = time.perf_counter()
        hits = scan_pii_fast(text, patterns)
        elapsed = (time.perf_counter() - start) * 1000

        print(
            f"{pat_label:<14} | {size_label:<8} | {len(text):>10,} | "
            f"{len(hits):>8,} | {elapsed:>10.2f} | {MODE}"
        )

    print(SEP)

print()
