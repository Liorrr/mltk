# Training Bug Detection

Catch the most common ML training bugs before they reach production. Data leakage (train/test contamination) causes artificially inflated metrics that collapse in deployment.

**Module:** `mltk.training`

---

## assert_no_train_test_overlap
Verify zero row overlap between train and test DataFrames on key columns.

## assert_temporal_split
Verify train data is strictly before test data (no temporal leakage).

## assert_no_target_leakage
Detect features too correlated with the target variable (proxy leakage).

---
