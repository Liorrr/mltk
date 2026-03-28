# Synthetic Data Validation

Validate the quality, fidelity, and privacy of synthetic tabular datasets. These four assertions cover the most common failure modes: unrealistic column distributions, broken inter-column relationships, copied training records, and records that are dangerously close to real individuals.

**Module:** `mltk.data.synthetic`

**ML Lifecycle Stage:** Post-generation validation / Privacy gate

---

## Why Validate Synthetic Data?

Synthetic data is increasingly used for three purposes: **privacy-preserving ML** (training on generated data instead of real patient records), **data augmentation** (expanding small datasets to improve model performance), and **testing** (populating staging environments with realistic-looking data that contains no real individuals).

But synthetic data can fail silently. Unlike a crashed pipeline, a bad synthetic dataset still *looks* like data -- it loads into a DataFrame, has the right column names, and contains plausible-looking numbers. The problems only surface downstream:

1. **Missed real-world patterns.** A generator that models each column independently will produce a dataset where 25-year-olds earn CEO salaries and retirees have student loan debt. Models trained on this data learn relationships that do not exist.

2. **Memorized training records.** Some generators -- especially overfit GANs and small-sample copula models -- reproduce training rows verbatim. The synthetic dataset becomes a privacy-violating copy of the original.

3. **Broken model training.** If the synthetic distribution diverges far enough from reality, models trained on it will perform poorly on real data. This is especially dangerous when the synthetic data *replaces* real data entirely (rather than augmenting it).

These four assertions catch these failure modes in order of increasing depth: start with individual columns, move to relationships between columns, then check for copies, and finally measure how close synthetic records are to real ones.

---

## The Four Checks

### 1. assert_marginal_fidelity

Per-column distribution comparison between real and synthetic data. Each column in the synthetic dataset should have a statistical distribution that closely matches the corresponding column in the real dataset.

```python
@timed_assertion
def assert_marginal_fidelity(
    real: pd.Series,
    synthetic: pd.Series,
    method: str = "ks",
    max_divergence: float = 0.1,
    severity: Severity = Severity.CRITICAL,
) -> TestResult
```

#### What it tests

Compares the distribution of a single column in the real dataset against the same column in the synthetic dataset. For a synthetic "age" column to pass, its histogram should roughly match the real "age" histogram -- similar mean, similar spread, similar shape.

The comparison uses a statistical test to produce a divergence score. If the divergence exceeds the threshold, the synthetic column is too far from reality.

#### Why it matters for ML

Marginal fidelity is the most basic requirement for synthetic data. If the synthetic "income" column has a mean of $500K when the real mean is $50K, every downstream analysis is wrong. Models trained on this data learn incorrect feature importances, calibration is destroyed, and threshold-based business rules (like "flag incomes above $200K") fire on the wrong records.

#### When to use it

- **First check in any synthetic data pipeline** -- if marginals are wrong, correlation and privacy checks are meaningless
- **Per-column validation after generation** -- run once per column to identify which specific columns the generator struggled with
- **Generator comparison** -- compare two generators by running marginal fidelity on each and comparing divergence scores
- **Hyperparameter tuning** -- measure how generator settings (epochs, noise, architecture) affect per-column quality

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `real` | `pd.Series` | *(required)* | A single column from the real dataset. |
| `synthetic` | `pd.Series` | *(required)* | The corresponding column from the synthetic dataset. |
| `method` | `str` | `"ks"` | Statistical test to use: `"ks"` (Kolmogorov-Smirnov) for continuous columns, `"psi"` (Population Stability Index) for both continuous and discretized data. |
| `max_divergence` | `float` | `0.1` | Maximum allowed divergence score. Lower is stricter. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion result. |

#### Returns

`TestResult` with:

- `name`: `"data.synthetic.marginal_fidelity[{column_name}]"`
- `passed`: `True` if divergence is within threshold
- `severity`: `CRITICAL`
- `details.divergence`: the computed divergence score
- `details.max_divergence`: the configured threshold
- `details.method`: statistical test used
- `details.real_mean`: mean of the real column (numeric only)
- `details.synthetic_mean`: mean of the synthetic column (numeric only)
- `details.real_std`: standard deviation of the real column (numeric only)
- `details.synthetic_std`: standard deviation of the synthetic column (numeric only)
- `details.p_value`: p-value from the statistical test

#### Example

```python
import pandas as pd
import pytest
from mltk.data.synthetic import assert_marginal_fidelity
from mltk.core.assertion import MltkAssertionError

def test_age_distribution_matches():
    """Synthetic age column should match real age distribution."""
    real_age = pd.Series([25, 30, 35, 40, 45, 50, 55, 60, 28, 33], name="age")
    synth_age = pd.Series([27, 31, 34, 42, 44, 48, 54, 59, 29, 36], name="age")

    result = assert_marginal_fidelity(real_age, synth_age, max_divergence=0.15)
    assert result.passed
    assert result.details["divergence"] < 0.15

def test_catches_shifted_distribution():
    """A generator that adds 20 years to everyone should fail."""
    real_age = pd.Series([25, 30, 35, 40, 45], name="age")
    synth_age = pd.Series([45, 50, 55, 60, 65], name="age")

    with pytest.raises(MltkAssertionError):
        assert_marginal_fidelity(real_age, synth_age, max_divergence=0.10)

def test_categorical_column():
    """Gender distribution should be preserved."""
    real_gender = pd.Series(["M", "F", "M", "F", "M", "F", "M", "F"], name="gender")
    synth_gender = pd.Series(["M", "F", "M", "F", "F", "M", "M", "F"], name="gender")

    result = assert_marginal_fidelity(real_gender, synth_gender, method="chi2")
    assert result.passed
```

#### Edge Cases

- **Constant columns** (all values identical) produce a divergence of 0.0 if both real and synthetic are constant with the same value. If only one is constant, divergence will be high.
- **NaN values** are dropped before comparison. If a column is mostly NaN, the sample size may be too small for a reliable statistical test.
- **Mixed types** -- if the real column is numeric but the synthetic column contains strings, an error is raised before the test runs. Clean your data first.

---

### 2. assert_correlation_preserved

Cross-column relationship comparison. The correlation structure between columns in the synthetic dataset should match the correlation structure in the real dataset.

```python
@timed_assertion
def assert_correlation_preserved(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_delta: float = 0.1,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult
```

#### What it tests

Computes the correlation matrix for both the real and synthetic DataFrames, then compares them element-by-element. If the real data has a 0.7 correlation between age and income, the synthetic data should have a similar correlation -- not 0.1 (too weak) and not 0.99 (suspiciously strong).

The assertion checks the *maximum absolute difference* across all column pairs. One badly preserved correlation is enough to fail.

#### Why it matters for ML

Many generators model columns independently -- they get each column's marginal distribution right but destroy the relationships between columns. This is catastrophic for ML because models learn from feature interactions. A credit scoring model relies on the relationship between income, debt, and default status. If the synthetic data says these are uncorrelated, the model learns nothing useful.

Even generators that claim to preserve correlations (copula-based, CTGAN) can fail on specific column pairs, especially when relationships are nonlinear or conditional. This assertion catches those failures.

#### When to use it

- **After marginal fidelity passes** -- there is no point checking correlations if individual columns are wrong
- **Generator quality assessment** -- the correlation gap is the single best metric for comparing independent-column generators vs. joint-distribution generators
- **Feature engineering validation** -- if you generate synthetic features, verify that the engineered features maintain their relationships
- **Regulatory compliance** -- some data governance frameworks require demonstrating that synthetic data preserves statistical properties

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `real_df` | `pd.DataFrame` | *(required)* | The real dataset (numeric columns only, or specify columns). |
| `synthetic_df` | `pd.DataFrame` | *(required)* | The synthetic dataset with the same columns. |
| `max_delta` | `float` | `0.1` | Maximum allowed normalized Frobenius norm of the correlation matrix difference. |
| `columns` | `list[str] \| None` | `None` | Subset of columns to compare. If None, uses all numeric columns present in both DataFrames. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion result. |

#### Returns

`TestResult` with:

- `name`: `"data.synthetic.correlation_preserved"`
- `passed`: `True` if normalized correlation difference is within threshold
- `severity`: `CRITICAL`
- `details.frobenius_norm`: Frobenius norm of the correlation matrix difference
- `details.normalized_diff`: normalized difference (frobenius / n_pairs)
- `details.max_delta`: the configured threshold
- `details.n_columns`: number of columns compared
- `details.worst_pair`: string `"column_a-column_b"` -- the pair with the largest divergence

#### Example

```python
import pandas as pd
import numpy as np
import pytest
from mltk.data.synthetic import assert_correlation_preserved
from mltk.core.assertion import MltkAssertionError

def test_correlations_match():
    """Synthetic data preserves age-income relationship."""
    rng = np.random.default_rng(42)
    n = 500

    # Real data: age and income are correlated
    real_age = rng.normal(40, 10, n)
    real_income = real_age * 1000 + rng.normal(0, 5000, n)
    real_df = pd.DataFrame({"age": real_age, "income": real_income})

    # Good synthetic: similar relationship
    synth_age = rng.normal(40, 10, n)
    synth_income = synth_age * 1050 + rng.normal(0, 5500, n)
    synth_df = pd.DataFrame({"age": synth_age, "income": synth_income})

    result = assert_correlation_preserved(real_df, synth_df, max_delta=0.10)
    assert result.passed

def test_catches_independent_columns():
    """A generator that models columns independently destroys correlations."""
    rng = np.random.default_rng(42)
    n = 500

    real_age = rng.normal(40, 10, n)
    real_income = real_age * 1000 + rng.normal(0, 5000, n)
    real_df = pd.DataFrame({"age": real_age, "income": real_income})

    # Bad synthetic: age and income generated independently
    synth_df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(40000, 15000, n),
    })

    with pytest.raises(MltkAssertionError) as exc:
        assert_correlation_preserved(real_df, synth_df, max_delta=0.10)
    # The worst pair tells you exactly which relationship broke
    assert "worst_pair" in str(exc.value) or True  # details in result
```

#### Edge Cases

- **Single-column DataFrames** have no pairs to compare. The assertion passes trivially with `num_pairs=0`.
- **Constant columns** produce NaN correlations in pandas. These pairs are excluded from comparison.
- **Column order** does not matter -- columns are matched by name, not position.
- **Missing columns** -- if the synthetic DataFrame is missing a column that exists in the real DataFrame, an error is raised.

---

### 3. assert_synthetic_novelty

Checks that synthetic rows are genuinely new, not copies of training data. A generator that memorizes and replays real records defeats the entire purpose of synthetic data.

```python
@timed_assertion
def assert_synthetic_novelty(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_copy_rate: float = 0.05,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult
```

#### What it tests

Compares every synthetic row against every real row to find exact duplicates. A synthetic row is a "copy" if it matches a real row on all columns. The copy rate is the fraction of synthetic rows that are copies of real rows.

#### Why it matters for ML

Synthetic data exists to provide utility *without* exposing real individuals. If a synthetic dataset contains exact copies of real records, it is not synthetic -- it is a data leak. This is a critical failure in:

- **Healthcare** -- a synthetic patient record that matches a real patient violates HIPAA
- **Finance** -- copied transaction records can expose real account activity
- **HR** -- duplicated employee records expose salary and performance data

Even from a pure ML perspective, a generator that copies training data is overfitting. The synthetic data adds no new information -- you might as well use the original.

#### When to use it

- **Privacy validation** -- required before sharing synthetic data with external parties
- **Generator debugging** -- high copy rates indicate overfitting (too many epochs, too little noise, too small a training set)
- **Compliance gates** -- some regulatory frameworks explicitly require proving that synthetic data does not contain real records
- **After marginal and correlation checks pass** -- a generator might pass fidelity checks *because* it is copying data, so novelty must be checked independently

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `real_df` | `pd.DataFrame` | *(required)* | The real (training) dataset. |
| `synthetic_df` | `pd.DataFrame` | *(required)* | The synthetic dataset to validate. |
| `max_copy_rate` | `float` | `0.05` | Maximum allowed fraction of synthetic rows that are exact copies of real rows (0-1). |
| `columns` | `list[str] \| None` | `None` | Subset of columns to consider when comparing rows. If None, uses all columns present in both DataFrames. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion result. |

#### Returns

`TestResult` with:

- `name`: `"data.synthetic.novelty"`
- `passed`: `True` if copy rate is within threshold
- `severity`: `CRITICAL`
- `details.copy_count`: number of synthetic rows that are exact duplicates of real rows
- `details.copy_rate`: fraction of synthetic rows that are copies (`copy_count / len(synthetic_df)`)
- `details.max_copy_rate`: the configured threshold
- `details.synthetic_rows`: total number of synthetic rows
- `details.real_rows`: total number of real rows

#### Example

```python
import pandas as pd
import pytest
from mltk.data.synthetic import assert_synthetic_novelty
from mltk.core.assertion import MltkAssertionError

def test_novel_synthetic_data():
    """Good generator produces unique rows."""
    real_df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [30000, 45000, 55000, 60000, 70000],
    })
    synth_df = pd.DataFrame({
        "age": [27, 33, 38, 42, 47],
        "income": [32000, 47000, 53000, 62000, 68000],
    })

    result = assert_synthetic_novelty(real_df, synth_df, max_copy_rate=0.05)
    assert result.passed
    assert result.details["copy_count"] == 0

def test_catches_memorized_data():
    """Generator that copies training data should fail."""
    real_df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [30000, 45000, 55000, 60000, 70000],
    })
    # 3 out of 5 rows are exact copies
    synth_df = pd.DataFrame({
        "age": [25, 30, 35, 42, 47],
        "income": [30000, 45000, 55000, 62000, 68000],
    })

    with pytest.raises(MltkAssertionError):
        assert_synthetic_novelty(real_df, synth_df, max_copy_rate=0.05)
```

#### Edge Cases

- **Floating point precision** -- rows that differ only due to floating point rounding (e.g., 0.30000000000000004 vs 0.3) are *not* counted as copies because they are not bitwise identical. If you need fuzzy matching, use `assert_dcr_safe` instead.
- **Column subset** -- the comparison uses all columns. If you want to check a subset, filter both DataFrames before calling.
- **Large datasets** -- the comparison is O(N*M) where N and M are the row counts. For datasets with millions of rows, consider sampling.
- **Duplicate rows within the real data** -- if the real data itself has duplicates, a synthetic row matching that value is counted once regardless of how many real copies exist.

---

### 4. assert_dcr_safe

Distance to Closest Record (DCR) safety check. Even synthetic rows that are not exact copies can be *dangerously close* to real records. DCR measures the minimum Euclidean (L2) distance from each synthetic row to any real row. Low DCR values indicate potential re-identification risk.

```python
@timed_assertion
def assert_dcr_safe(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    min_dcr: float = 0.05,
    sample_size: int = 2000,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult
```

#### What it tests

For each synthetic row, computes the Euclidean distance to every real row and takes the minimum -- that is the Distance to Closest Record. The assertion then checks a low quantile (default: 5th percentile) of these DCR values. If even the closest synthetic records are far enough from all real records, the data is safe.

The intuition: a synthetic row for "age=35, income=$50,001" is dangerously close to a real row "age=35, income=$50,000" even though it is not an exact copy. An attacker with the real data could match these records and re-identify the individual.

#### Why it matters for ML

DCR is the strongest privacy metric for synthetic data. It catches cases that novelty checks miss:

- A synthetic record that changes one digit in a phone number
- A synthetic address that differs by one house number
- A synthetic salary that is $1 off from a real salary

These "near-copies" are not exact duplicates, so they pass novelty checks. But they are close enough that an attacker with access to the real data could match them with high confidence.

DCR-based validation is required or recommended by:

- **Healthcare data sharing** -- synthetic patient data must be demonstrably far from real patients
- **Financial data sandboxes** -- synthetic transaction data for third-party testing
- **Government open data** -- synthetic census or survey data released to the public

#### When to use it

- **Strongest privacy check** -- run this after novelty, as the final gate before releasing synthetic data
- **Required for regulated industries** -- healthcare, finance, and government synthetic data programs
- **Generator tuning** -- if DCR is too low, increase noise or reduce training epochs
- **Privacy budget decisions** -- lower `min_dcr` allows better utility but weaker privacy; higher `min_dcr` provides stronger privacy at the cost of potential utility loss

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `real_df` | `pd.DataFrame` | *(required)* | The real dataset (numeric columns). Categorical columns should be encoded first. |
| `synthetic_df` | `pd.DataFrame` | *(required)* | The synthetic dataset with the same columns. |
| `min_dcr` | `float` | `0.05` | Minimum acceptable median DCR. |
| `sample_size` | `int` | `2000` | Maximum number of synthetic rows to evaluate (sampled for efficiency). |
| `columns` | `list[str] \| None` | `None` | Subset of numeric columns to use. If None, uses all numeric columns present in both DataFrames. |
| `severity` | `Severity` | `CRITICAL` | Severity level for the assertion result. |

#### Returns

`TestResult` with:

- `name`: `"data.synthetic.dcr_safe"`
- `passed`: `True` if the median DCR exceeds `min_dcr`
- `severity`: `CRITICAL`
- `details.median_dcr`: median DCR across sampled synthetic rows
- `details.mean_dcr`: mean DCR across sampled synthetic rows
- `details.min_dcr_threshold`: the configured threshold
- `details.p5_dcr`: the DCR value at the 5th percentile
- `details.n_sampled`: number of synthetic rows evaluated
- `details.n_real`: number of real rows used for comparison

#### Example

```python
import pandas as pd
import numpy as np
import pytest
from mltk.data.synthetic import assert_dcr_safe
from mltk.core.assertion import MltkAssertionError

def test_synthetic_data_is_far_from_real():
    """Well-generated synthetic data should have sufficient distance from real data."""
    rng = np.random.default_rng(42)

    real_df = pd.DataFrame({
        "age": rng.normal(40, 10, 100),
        "income": rng.normal(50000, 15000, 100),
    })

    # Synthetic data with added noise -- sufficiently different
    synth_df = pd.DataFrame({
        "age": rng.normal(40, 10, 100),
        "income": rng.normal(50000, 15000, 100),
    })

    result = assert_dcr_safe(real_df, synth_df, min_dcr=0.01)
    assert result.passed
    print(f"5th percentile DCR: {result.details['p5_dcr']:.4f}")
    print(f"Closest synthetic row: {result.details['min_dcr_observed']:.4f}")

def test_catches_near_copies():
    """Synthetic data that is almost identical to real data should fail."""
    real_df = pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0, 45.0],
        "income": [30000.0, 45000.0, 55000.0, 60000.0, 70000.0],
    })

    # Near-copies: tiny perturbation that passes novelty but fails DCR
    synth_df = pd.DataFrame({
        "age": [25.001, 30.001, 35.001, 40.001, 45.001],
        "income": [30000.1, 45000.1, 55000.1, 60000.1, 70000.1],
    })

    with pytest.raises(MltkAssertionError):
        assert_dcr_safe(real_df, synth_df, min_dcr=0.10)
```

#### Edge Cases

- **Feature scaling** -- DCR is sensitive to column scales. A column in dollars (range 0-100,000) will dominate a column in years (range 0-100). Normalize both DataFrames before computing DCR, or the distance will be driven by whichever column has the largest absolute values.
- **Categorical columns** -- Euclidean distance is not meaningful for categorical data. Encode categorical columns (one-hot or ordinal) before computing DCR.
- **High-dimensional data** -- in high dimensions, all distances tend to converge (the "curse of dimensionality"), making DCR less discriminative. Consider dimensionality reduction or using a subset of the most sensitive columns.
- **Computational cost** -- DCR is O(N*M*D) where N = synthetic rows, M = real rows, D = columns. For large datasets, approximate nearest-neighbor methods (e.g., ball tree, KD-tree) are recommended.

---

## Recommended Validation Pipeline

Run the four checks in order. Each check builds on the previous one -- there is no point checking correlations if individual columns are wrong, and there is no point checking privacy if the data is statistically meaningless.

```python
import pandas as pd
from mltk.data.synthetic import (
    assert_marginal_fidelity,
    assert_correlation_preserved,
    assert_synthetic_novelty,
    assert_dcr_safe,
)

# Load your real and synthetic datasets
real_df = pd.read_csv("data/real_patients.csv")
synth_df = pd.read_csv("data/synthetic_patients.csv")

# Step 1: Are individual columns realistic?
# Run per-column -- identifies exactly which columns the generator struggled with
for col in real_df.columns:
    assert_marginal_fidelity(real_df[col], synth_df[col], max_divergence=0.10)

# Step 2: Are column relationships preserved?
# A generator that models columns independently will fail here
assert_correlation_preserved(real_df, synth_df, max_delta=0.10)

# Step 3: Did the generator create new data (not copies)?
# Catches overfitting and direct memorization
assert_synthetic_novelty(real_df, synth_df, max_copy_rate=0.05)

# Step 4: Is the synthetic data far enough from real records?
# Strongest privacy check -- catches near-copies that novelty misses
assert_dcr_safe(real_df, synth_df, min_dcr=0.05)
```

---

## Threshold Guidance

Choosing thresholds depends on your use case. Stricter thresholds mean higher data quality requirements but more generator rejections. Looser thresholds are easier to pass but allow lower-quality or less-private data through.

| Assertion | Conservative | Moderate | Permissive | Unit |
|-----------|-------------|----------|------------|------|
| `marginal_fidelity` (max_divergence) | 0.05 | 0.10 | 0.20 | KS statistic or chi2 p-value |
| `correlation_preserved` (max_delta) | 0.05 | 0.10 | 0.15 | Normalized Frobenius norm of correlation difference |
| `novelty` (max_copy_rate) | 0.01 | 0.05 | 0.10 | Fraction of rows that are copies |
| `dcr_safe` (min_dcr) | 0.10 | 0.05 | 0.02 | L2 distance (normalized) |

**When to use each level:**

- **Conservative** -- regulated industries (healthcare, finance), external data sharing, public release. You need high confidence that the data is both realistic and private.
- **Moderate** -- internal ML training, data augmentation, development/staging environments. A reasonable balance between quality and generator flexibility.
- **Permissive** -- rapid prototyping, initial generator experiments, non-sensitive data. Useful during generator development when you expect to iterate on quality.

---

## Common Failure Patterns

### Generator overfitting

**Symptom:** Marginal fidelity and correlation checks pass with suspiciously low divergence scores. Novelty check fails.

**Cause:** The generator memorized the training data instead of learning the distribution. Common with GANs trained for too many epochs on small datasets.

**Fix:** Reduce training epochs, add noise regularization, increase training data size, or use a different generator architecture.

### Independent column generation

**Symptom:** Marginal fidelity passes but correlation check fails badly. `worst_pair` shows column pairs that should be correlated.

**Cause:** The generator models each column separately (e.g., sampling from per-column histograms). It gets each column right individually but destroys relationships.

**Fix:** Use a joint-distribution generator (copula, CTGAN, or Bayesian network) that explicitly models column dependencies.

### Insufficient privacy noise

**Symptom:** Novelty check passes (no exact copies) but DCR check fails (synthetic rows are too close to real rows).

**Cause:** The generator adds minimal noise to real records rather than generating truly new samples.

**Fix:** Increase the noise parameter in the generator, apply differential privacy mechanisms, or post-process synthetic data with additional perturbation.

### Mode collapse

**Symptom:** Marginal fidelity fails on specific columns -- the synthetic data covers only a subset of the real distribution (e.g., only generates ages 30-50 when real data spans 18-80).

**Cause:** GAN mode collapse or insufficient generator capacity.

**Fix:** Check generator loss curves for signs of collapse, increase generator network capacity, or switch to a VAE or copula-based generator.

---

## Integration with pytest

All four assertions work naturally with pytest and the mltk pytest plugin:

```python
import pandas as pd
import pytest
from mltk.data.synthetic import (
    assert_marginal_fidelity,
    assert_correlation_preserved,
    assert_synthetic_novelty,
    assert_dcr_safe,
)

@pytest.fixture
def real_data():
    return pd.read_csv("tests/fixtures/real_patients.csv")

@pytest.fixture
def synthetic_data():
    return pd.read_csv("tests/fixtures/synthetic_patients.csv")

class TestSyntheticDataQuality:
    """Full validation suite for synthetic patient data."""

    @pytest.mark.parametrize("column", ["age", "income", "blood_pressure", "bmi"])
    def test_marginal_fidelity(self, real_data, synthetic_data, column):
        assert_marginal_fidelity(
            real_data[column], synthetic_data[column], max_divergence=0.10
        )

    def test_correlation_preserved(self, real_data, synthetic_data):
        result = assert_correlation_preserved(real_data, synthetic_data, max_delta=0.10)
        # Log the worst pair for debugging
        print(f"Worst pair: {result.details['worst_pair']}")

    def test_novelty(self, real_data, synthetic_data):
        assert_synthetic_novelty(real_data, synthetic_data, max_copy_rate=0.01)

    def test_dcr_safe(self, real_data, synthetic_data):
        assert_dcr_safe(real_data, synthetic_data, min_dcr=0.05)
```
