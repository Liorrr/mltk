"""Benchmark drift detection functions (KS, PSI, KL, JS, Wasserstein, cosine SIMD)."""

from __future__ import annotations

import time

import numpy as np

from mltk._rust import (
    RUST_AVAILABLE,
    bertscore_precision_recall,
    centroid_cosine_distance,
    cosine_similarity,
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

# ─── Cosine / BERTScore Benchmarks ───────────────────────────────────────────

EMBEDDING_DIMS = [128, 768, 1536]
EMBEDDING_SIZES = [100, 1_000, 5_000]

COSINE_HEADER = f"{'Function':<30} | {'n_tokens':>10} | {'dim':>6} | {'ms':>10} | Mode"
COSINE_SEP = "-" * len(COSINE_HEADER)

print(f"\n{'Cosine / BERTScore Benchmark':^{len(COSINE_HEADER)}}")
print(COSINE_SEP)
print(COSINE_HEADER)
print(COSINE_SEP)

rng = np.random.default_rng(99)

for dim in EMBEDDING_DIMS:
    # Single-pair cosine similarity
    a = rng.standard_normal(dim).tolist()
    b = rng.standard_normal(dim).tolist()

    # Warm-up
    cosine_similarity(a, b)

    start = time.perf_counter()
    for _ in range(10_000):
        cosine_similarity(a, b)
    elapsed = (time.perf_counter() - start) * 1000 / 10_000

    print(f"{'cosine_similarity':<30} | {'1':>10} | {dim:>6} | {elapsed:>10.4f} | {MODE}")

print(COSINE_SEP)

for n_tokens in EMBEDDING_SIZES:
    for dim in EMBEDDING_DIMS:
        ref_embs = rng.standard_normal((n_tokens, dim)).tolist()
        cur_embs = (rng.standard_normal((n_tokens, dim)) + 0.05).tolist()

        # Warm-up with small slice
        if n_tokens == EMBEDDING_SIZES[0]:
            centroid_cosine_distance(ref_embs[:10], cur_embs[:10])

        start = time.perf_counter()
        centroid_cosine_distance(ref_embs, cur_embs)
        elapsed = (time.perf_counter() - start) * 1000

        print(
            f"{'centroid_cosine_distance':<30} | {n_tokens:>10,} | {dim:>6} | "
            f"{elapsed:>10.2f} | {MODE}"
        )

print(COSINE_SEP)

# BERTScore benchmark — smaller sizes (O(N*M) per token pair)
BERTSCORE_SIZES = [16, 64, 256]

for n_tokens in BERTSCORE_SIZES:
    for dim in [128, 768]:
        ref_embs = rng.standard_normal((n_tokens, dim)).tolist()
        hyp_embs = rng.standard_normal((n_tokens, dim)).tolist()

        # Warm-up
        if n_tokens == BERTSCORE_SIZES[0]:
            bertscore_precision_recall(ref_embs[:4], hyp_embs[:4])

        start = time.perf_counter()
        bertscore_precision_recall(ref_embs, hyp_embs)
        elapsed = (time.perf_counter() - start) * 1000

        print(
            f"{'bertscore_precision_recall':<30} | {n_tokens:>10} | {dim:>6} | "
            f"{elapsed:>10.2f} | {MODE}"
        )

print(COSINE_SEP)
print()
