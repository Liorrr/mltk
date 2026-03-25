"""Benchmark drift detection functions (KS, PSI, KL, JS, Wasserstein)."""

from __future__ import annotations

import time

import numpy as np

from mltk._rust import (
    RUST_AVAILABLE,
    js_divergence,
    kl_divergence,
    ks_test,
    psi,
    wasserstein,
)

SIZES = [10_000, 100_000, 1_000_000]
MODE = "RUST" if RUST_AVAILABLE else "PYTHON"

FUNCTIONS = [
    ("ks_test", lambda ref, cur: ks_test(ref, cur)),
    ("psi(bins=10)", lambda ref, cur: psi(ref, cur, 10)),
    ("kl_divergence", lambda ref, cur: kl_divergence(ref, cur, 10)),
    ("js_divergence", lambda ref, cur: js_divergence(ref, cur, 10)),
    ("wasserstein", lambda ref, cur: wasserstein(ref, cur)),
]

HEADER = f"{'Function':<22} | {'n':>12} | {'ms':>10} | Mode"
SEP = "-" * len(HEADER)

print(f"\n{'Drift Benchmark':^{len(HEADER)}}")
print(SEP)
print(HEADER)
print(SEP)

for n in SIZES:
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(n).tolist()
    cur = (rng.standard_normal(n) + 0.5).tolist()

    for func_name, func in FUNCTIONS:
        # Warm-up pass (exclude from timing)
        if n == SIZES[0]:
            func(ref[:1000], cur[:1000])

        start = time.perf_counter()
        func(ref, cur)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"{func_name:<22} | {n:>12,} | {elapsed:>10.2f} | {MODE}")

    print(SEP)

print()
