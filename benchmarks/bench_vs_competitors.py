"""Benchmark mltk assertion speed on realistic datasets."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def bench(name: str, func, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    for _ in range(100):
        try:
            func(*args, **kwargs)
        except Exception:
            pass
    elapsed = (time.perf_counter() - start) / 100 * 1000
    print(f"{name:40s} | {elapsed:8.2f} ms")


# Generate test data
n = 10_000
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "id": range(n),
    "value": rng.standard_normal(n),
    "label": rng.choice([0, 1], n),
    "text": [f"sample text {i}" for i in range(n)],
})

print(f"mltk Benchmark — {n:,} rows")
print("=" * 55)

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
from mltk.data import assert_schema, assert_no_nulls, assert_row_count, assert_range

bench("assert_schema", assert_schema, df, {"id": "int64", "value": "float64"})
bench("assert_no_nulls", assert_no_nulls, df)
bench("assert_row_count", assert_row_count, df, min_rows=100)
bench("assert_range", assert_range, df, "value", min_val=-10, max_val=10)

# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
from mltk.data import assert_no_drift

ref = pd.Series(rng.standard_normal(n))
cur = pd.Series(rng.standard_normal(n) + 0.1)
bench("assert_no_drift (KS)", assert_no_drift, ref, cur, method="ks")
bench("assert_no_drift (PSI)", assert_no_drift, ref, cur, method="psi")

# ---------------------------------------------------------------------------
# Feature-label correlation stability (new S33 assertion)
# ---------------------------------------------------------------------------
from mltk.data import assert_feature_label_correlation_stable

train_df = pd.DataFrame({
    "feat_a": rng.standard_normal(n),
    "feat_b": rng.standard_normal(n),
    "label": rng.choice([0, 1], n),
})
test_df = pd.DataFrame({
    "feat_a": rng.standard_normal(n),
    "feat_b": rng.standard_normal(n),
    "label": rng.choice([0, 1], n),
})
bench(
    "assert_feature_label_corr_stable",
    assert_feature_label_correlation_stable,
    train_df, test_df,
    feature_cols=["feat_a", "feat_b"],
    label_col="label",
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
from mltk.model import assert_metric

y_true = rng.choice([0, 1], 1000)
y_pred = rng.choice([0, 1], 1000)
bench("assert_metric (accuracy)", assert_metric, y_true, y_pred, metric="accuracy", threshold=0.0)

# ---------------------------------------------------------------------------
# Output drift (new S33 assertion)
# ---------------------------------------------------------------------------
from mltk.monitor import assert_no_output_drift

ref_scores = rng.random(n).tolist()
cur_scores = (rng.random(n) + 0.02).tolist()
bench("assert_no_output_drift (KS)", assert_no_output_drift, ref_scores, cur_scores, method="ks")
bench("assert_no_output_drift (PSI)", assert_no_output_drift, ref_scores, cur_scores, method="psi")

print("\nDone.")
