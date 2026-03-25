# Tabular Testing

Tabular-specific assertions for feature drift across DataFrames, feature importance stability, and class balance checks.

**Module:** `mltk.domains.tabular`

---

## assert_feature_drift

Per-column drift detection on entire DataFrames. Runs `assert_no_drift` on each numeric column.

```python
from mltk.domains.tabular import assert_feature_drift

assert_feature_drift(ref_df, cur_df, method="psi", threshold=0.1)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ref_df` | `pd.DataFrame` | *(required)* | Reference DataFrame (baseline) |
| `cur_df` | `pd.DataFrame` | *(required)* | Current DataFrame to compare |
| `method` | `str` | `"psi"` | Drift method per column (`"psi"` or `"ks"`) |
| `threshold` | `float` | `0.1` | Drift threshold per column |
| `columns` | `list[str] \| None` | `None` | Columns to check. `None` = all shared numeric columns |

### Returns

`TestResult` with details:
- `drifted_features` -- dict of drifted feature names to drift scores
- `stable_features` -- dict of stable feature names to drift scores
- `total_features` -- number of features checked
- `method` -- drift method used
- `threshold` -- configured threshold

---

## assert_feature_importance_stable

Validate SHAP feature ranking stability between model versions.

```python
from mltk.domains.tabular import assert_feature_importance_stable

assert_feature_importance_stable(shap_ref, shap_cur, max_rank_change=3)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `shap_ref` | `dict[str, float]` | *(required)* | Reference feature importances `{feature: importance}` |
| `shap_cur` | `dict[str, float]` | *(required)* | Current feature importances |
| `max_rank_change` | `int` | `3` | Maximum allowed rank change for any feature |

### Returns

`TestResult` with severity `WARNING` (not `CRITICAL`):
- `max_change` -- maximum rank change observed
- `max_rank_change` -- configured threshold
- `rank_changes` -- dict of feature name to rank change

!!! note "WARNING severity"
    Unlike most assertions, `assert_feature_importance_stable` uses `Severity.WARNING`. It does not raise an exception on failure -- it returns a `TestResult` with `passed=False`.

---

## assert_class_balance

Convenience wrapper for class imbalance detection on tabular datasets. Delegates to `mltk.data.labels.assert_label_balance`.

```python
from mltk.domains.tabular import assert_class_balance

assert_class_balance(df, label_col="target", max_ratio=10.0)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame containing the label column |
| `label_col` | `str` | *(required)* | Name of the label column |
| `max_ratio` | `float` | `10.0` | Maximum majority/minority ratio |

### Returns

`TestResult` with the same details as `assert_label_balance` (see [data-labels](data-labels.md)).

---
