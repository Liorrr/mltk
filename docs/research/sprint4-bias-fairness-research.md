# Sprint 4 Research: ML Bias & Fairness Testing Methods

Research completed: March 25, 2026

---

## 1. Demographic Parity (Statistical Parity / Group Fairness)

### Definition
A classifier satisfies demographic parity if its predictions are statistically independent of the sensitive attribute. The positive prediction rate should be equal across all groups.

### Formula
```
P(Y_hat=1 | A=a) = P(Y_hat=1 | A=b)   for all groups a, b

Demographic Parity Difference = max_a(selection_rate(a)) - min_a(selection_rate(a))
Demographic Parity Ratio     = min_a(selection_rate(a)) / max_a(selection_rate(a))
```

Where `selection_rate(a) = count(Y_hat=1 AND A=a) / count(A=a)`

### Interpretation
- **Difference = 0** or **Ratio = 1**: Perfect parity
- **Difference > 0.1**: Commonly flagged as potentially unfair
- **Ratio < 0.8**: Fails the "80% rule" (see Disparate Impact)

### Typical Thresholds
| Metric | Fair | Warning | Fail |
|--------|------|---------|------|
| DP Difference | < 0.05 | 0.05-0.10 | > 0.10 |
| DP Ratio | > 0.90 | 0.80-0.90 | < 0.80 |

### When to Use
- Hiring, lending, admissions -- where equal opportunity to receive a positive outcome matters
- When you want outcome independence from group membership
- Regulatory compliance screening (first-pass fairness check)

### Limitations
- Ignores ground truth labels entirely -- a model that is correct more often for one group may still violate DP
- Cannot be satisfied simultaneously with calibration when base rates differ between groups

### Code Example (Pure Python -- mltk style)
```python
import numpy as np

def demographic_parity_difference(y_pred, sensitive_feature):
    """Compute demographic parity difference.

    Returns the difference between the highest and lowest group selection rates.
    A value of 0 indicates perfect parity.
    """
    groups = np.unique(sensitive_feature)
    rates = []
    for g in groups:
        mask = sensitive_feature == g
        rate = np.mean(y_pred[mask])
        rates.append(rate)
    return max(rates) - min(rates)

def demographic_parity_ratio(y_pred, sensitive_feature):
    """Compute demographic parity ratio.

    Returns the ratio of the lowest to highest group selection rate.
    A value of 1 indicates perfect parity. Below 0.8 = disparate impact.
    """
    groups = np.unique(sensitive_feature)
    rates = []
    for g in groups:
        mask = sensitive_feature == g
        rate = np.mean(y_pred[mask])
        rates.append(rate)
    max_rate = max(rates)
    if max_rate == 0:
        return 1.0
    return min(rates) / max_rate
```

---

## 2. Equalized Odds (Separation)

### Definition
A classifier satisfies equalized odds if the True Positive Rate (TPR) and False Positive Rate (FPR) are equal across all groups. This is stricter than demographic parity because it conditions on the true label.

### Formula
```
P(Y_hat=1 | Y=1, A=a) = P(Y_hat=1 | Y=1, A=b)   (equal TPR)
P(Y_hat=1 | Y=0, A=a) = P(Y_hat=1 | Y=0, A=b)   (equal FPR)

Equalized Odds Difference = max(TPR_difference, FPR_difference)
Equalized Odds Ratio = min(TPR_ratio, FPR_ratio)
```

Where:
- `TPR(a) = count(Y_hat=1 AND Y=1 AND A=a) / count(Y=1 AND A=a)`
- `FPR(a) = count(Y_hat=1 AND Y=0 AND A=a) / count(Y=0 AND A=a)`

### Interpretation
- **Difference = 0** or **Ratio = 1**: Perfect equalized odds
- Ensures equal error rates across groups, not just equal selection rates
- A relaxed variant, **Equal Opportunity**, only requires equal TPR (ignoring FPR)

### Typical Thresholds
| Metric | Fair | Warning | Fail |
|--------|------|---------|------|
| EO Difference | < 0.05 | 0.05-0.10 | > 0.10 |
| EO Ratio | > 0.90 | 0.80-0.90 | < 0.80 |

### When to Use
- When error consequences are asymmetric (e.g., criminal justice, medical diagnosis)
- When you need the model to perform equally well for all groups, not just predict equally often
- When ground truth labels are available and trusted

### Limitations
- Requires labeled data per group
- Cannot be simultaneously satisfied with calibration when base rates differ (impossibility theorem)
- More restrictive than demographic parity -- harder to achieve

### Code Example
```python
import numpy as np

def equalized_odds_difference(y_true, y_pred, sensitive_feature):
    """Compute equalized odds difference.

    Returns max(TPR_difference, FPR_difference) across groups.
    A value of 0 indicates perfect equalized odds.
    """
    groups = np.unique(sensitive_feature)
    tprs, fprs = [], []
    for g in groups:
        mask = sensitive_feature == g
        yt, yp = y_true[mask], y_pred[mask]
        pos = yt == 1
        neg = yt == 0
        tpr = np.mean(yp[pos]) if pos.sum() > 0 else 0.0
        fpr = np.mean(yp[neg]) if neg.sum() > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    return max(tpr_diff, fpr_diff)

def equalized_odds_ratio(y_true, y_pred, sensitive_feature):
    """Compute equalized odds ratio.

    Returns min(TPR_ratio, FPR_ratio) across groups.
    A value of 1 indicates perfect equalized odds.
    """
    groups = np.unique(sensitive_feature)
    tprs, fprs = [], []
    for g in groups:
        mask = sensitive_feature == g
        yt, yp = y_true[mask], y_pred[mask]
        pos = yt == 1
        neg = yt == 0
        tpr = np.mean(yp[pos]) if pos.sum() > 0 else 0.0
        fpr = np.mean(yp[neg]) if neg.sum() > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    def safe_ratio(values):
        mx = max(values)
        if mx == 0:
            return 1.0
        return min(values) / mx

    return min(safe_ratio(tprs), safe_ratio(fprs))
```

---

## 3. Predictive Parity (Calibration)

### Definition
A classifier satisfies predictive parity if the Positive Predictive Value (PPV / precision) is equal across all groups. When the model predicts positive, the probability that the outcome is truly positive should be the same regardless of group membership.

### Formula
```
P(Y=1 | Y_hat=1, A=a) = P(Y=1 | Y_hat=1, A=b)   for all groups a, b

Predictive Parity Difference = max_a(PPV(a)) - min_a(PPV(a))
```

Where `PPV(a) = count(Y=1 AND Y_hat=1 AND A=a) / count(Y_hat=1 AND A=a)`

### Interpretation
- **Difference = 0**: Perfect predictive parity
- Ensures that a positive prediction is equally trustworthy across groups
- Also known as "outcome test" -- given the same prediction, is the outcome the same?

### Typical Thresholds
| Metric | Fair | Warning | Fail |
|--------|------|---------|------|
| PP Difference | < 0.05 | 0.05-0.10 | > 0.10 |

### Trade-offs (Impossibility Theorem)
**Critical**: When base rates differ between groups, it is mathematically impossible to simultaneously satisfy:
1. Demographic parity
2. Equalized odds
3. Predictive parity

This is the Chouldechova-Kleinberg impossibility theorem. In practice, you must choose which fairness criterion matters most for your application:

| Use Case | Primary Metric | Rationale |
|----------|---------------|-----------|
| Hiring / lending | Demographic parity | Equal opportunity regardless of group |
| Criminal justice | Equalized odds | Equal error rates (mistakes cost freedom) |
| Medical diagnosis | Predictive parity | Positive prediction must be equally reliable |

### When to Use
- When the trustworthiness of positive predictions matters (medical screening, credit scoring)
- When different groups may have genuinely different base rates
- When you want the model's confidence to be calibrated per group

### Limitations
- Does not control for false negative rates
- A model can satisfy predictive parity while having wildly different approval rates

### Code Example
```python
import numpy as np

def predictive_parity_difference(y_true, y_pred, sensitive_feature):
    """Compute predictive parity difference.

    Returns the difference between the highest and lowest group PPV.
    A value of 0 indicates perfect predictive parity.
    """
    groups = np.unique(sensitive_feature)
    ppvs = []
    for g in groups:
        mask = sensitive_feature == g
        yt, yp = y_true[mask], y_pred[mask]
        pred_pos = yp == 1
        if pred_pos.sum() == 0:
            ppvs.append(0.0)
        else:
            ppvs.append(np.mean(yt[pred_pos]))
    return max(ppvs) - min(ppvs)
```

---

## 4. Disparate Impact Ratio (The 80% / Four-Fifths Rule)

### Definition
Disparate impact occurs when a selection process has a disproportionate negative effect on a protected group. The four-fifths rule, codified in the 1978 EEOC Uniform Guidelines, states that a selection rate for any group below 80% of the highest group's rate indicates potential adverse impact.

### Formula
```
Disparate Impact Ratio = selection_rate(unprivileged) / selection_rate(privileged)

Where selection_rate(g) = count(positive outcomes for group g) / count(members of group g)
```

### Interpretation
- **Ratio >= 0.80**: Generally considered acceptable (no adverse impact)
- **Ratio < 0.80**: Indicates potential disparate impact -- triggers further investigation
- **Ratio = 1.0**: Perfect parity

### Historical and Legal Context
- Originated from California FEPC guidelines (1972), codified in 1978 EEOC Uniform Guidelines
- Used in US employment law (Title VII of the Civil Rights Act)
- Applied in lending (Equal Credit Opportunity Act, Fair Housing Act)
- FACCt 2024 paper clarifies: the four-fifths rule is a screening tool, not a legal standard for proving discrimination

### 2025 Policy Change
Executive Order 14281 (April 23, 2025) instructed federal agencies to stop enforcing disparate impact cases. However, statutory and case law precedent remains, and state-level enforcement continues. For ML tooling, the 80% threshold remains the industry standard benchmark.

### When to Use
- Hiring algorithms, lending models, insurance pricing
- Any selection system subject to anti-discrimination law
- As a simple, legally-grounded first-pass fairness screen

### Code Example
```python
import numpy as np

def disparate_impact_ratio(y_pred, sensitive_feature, privileged_group=None):
    """Compute disparate impact ratio (four-fifths rule).

    If privileged_group is specified, computes ratio against that group.
    Otherwise, uses the group with the highest selection rate.
    Returns ratio in [0, 1]. Below 0.8 indicates potential disparate impact.
    """
    groups = np.unique(sensitive_feature)
    rates = {}
    for g in groups:
        mask = sensitive_feature == g
        rates[g] = np.mean(y_pred[mask])

    if privileged_group is not None:
        priv_rate = rates[privileged_group]
    else:
        priv_rate = max(rates.values())

    if priv_rate == 0:
        return 1.0

    # Return the minimum ratio across all unprivileged groups
    ratios = [r / priv_rate for g, r in rates.items()
              if (privileged_group is None or g != privileged_group)]
    return min(ratios) if ratios else 1.0
```

---

## 5. Calibration Across Groups

### Definition
A model is calibrated across groups if, for any given predicted probability score, the actual positive rate is the same across all demographic groups. This means predicted confidence is equally reliable regardless of group membership.

### Formula
```
P(Y=1 | Score=s, A=a) = P(Y=1 | Score=s, A=b)   for all s, a, b

ECE (Expected Calibration Error) per group:
ECE(a) = sum_b  |bin_count(b,a)/N(a)| * |avg_confidence(b,a) - avg_accuracy(b,a)|

Calibration Gap = max_a(ECE(a)) - min_a(ECE(a))
```

### Interpretation
- **Calibration gap = 0**: Model is equally calibrated for all groups
- A model can be well-calibrated overall but poorly calibrated for minority groups
- Essential in medical, financial, and legal contexts where probability estimates drive decisions

### Typical Thresholds
| Metric | Fair | Warning | Fail |
|--------|------|---------|------|
| ECE per group | < 0.05 | 0.05-0.10 | > 0.10 |
| Max ECE gap | < 0.03 | 0.03-0.05 | > 0.05 |

### When to Use
- Probabilistic predictions (not just binary classification)
- Risk scoring (credit risk, disease probability, recidivism)
- When downstream decisions use the probability value directly, not just the binary label

### Code Example
```python
import numpy as np

def calibration_by_group(y_true, y_prob, sensitive_feature, n_bins=10):
    """Compute Expected Calibration Error per demographic group.

    Returns dict mapping group -> ECE value.
    """
    groups = np.unique(sensitive_feature)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    result = {}

    for g in groups:
        mask = sensitive_feature == g
        yt = y_true[mask]
        yp = y_prob[mask]
        ece = 0.0
        n = len(yt)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
            count = in_bin.sum()
            if count == 0:
                continue
            avg_conf = yp[in_bin].mean()
            avg_acc = yt[in_bin].mean()
            ece += (count / n) * abs(avg_conf - avg_acc)

        result[g] = ece

    return result

def calibration_gap(y_true, y_prob, sensitive_feature, n_bins=10):
    """Max ECE difference between any two groups."""
    eces = calibration_by_group(y_true, y_prob, sensitive_feature, n_bins)
    vals = list(eces.values())
    return max(vals) - min(vals)
```

---

## 6. Fairlearn Library (v0.13.0, October 2025)

### Overview
Microsoft's Fairlearn is the most widely-used Python library for ML fairness assessment and mitigation. Apache-2.0 licensed, follows scikit-learn API conventions.

### Key Assessment APIs

```python
from fairlearn.metrics import (
    MetricFrame,               # Core: disaggregated metric evaluation
    selection_rate,            # P(Y_hat=1) per group
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate,
    false_positive_rate,
)
from sklearn.metrics import accuracy_score

# MetricFrame: the central class
mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
)

mf.overall          # Overall metric values
mf.by_group         # Per-group breakdown (pandas DataFrame)
mf.difference()     # Max - min across groups
mf.ratio()          # Min / max across groups
mf.group_min()      # Worst-performing group
mf.group_max()      # Best-performing group
```

### Mitigation Algorithms

| Stage | Algorithm | Description |
|-------|-----------|-------------|
| Pre-processing | CorrelationRemover | Removes correlation between features and sensitive attribute |
| In-processing | ExponentiatedGradient | Trains with fairness constraints (reduction approach) |
| In-processing | GridSearch | Grid search over constraint-satisfying models |
| Post-processing | ThresholdOptimizer | Adjusts decision thresholds per group |

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

# In-processing: train with fairness constraint
mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=DemographicParity(),
)
mitigator.fit(X_train, y_train, sensitive_features=A_train)

# Post-processing: adjust thresholds
optimizer = ThresholdOptimizer(
    estimator=base_model,
    constraints="demographic_parity",
    predict_method="predict_proba",
    prefit=True,
)
optimizer.fit(X_train, y_train, sensitive_features=A_train)
```

### Supported Constraint Classes
- `DemographicParity()` -- equal selection rates
- `EqualizedOdds()` -- equal TPR and FPR
- `TruePositiveRateParity()` -- equal opportunity (relaxed EO)
- `FalsePositiveRateParity()`
- `ErrorRateParity()`
- `BoundedGroupLoss()` -- bounded worst-case loss

### mltk Design Implications
- mltk should NOT depend on fairlearn -- compute metrics natively for zero-dependency assertions
- mltk SHOULD match fairlearn's metric naming conventions for familiarity
- mltk SHOULD support `sensitive_features` parameter pattern (array same length as predictions)
- mltk CAN offer optional fairlearn integration for users who want mitigation

---

## 7. EU AI Act Fairness Requirements

### Mandatory for High-Risk AI Systems (Compliance Deadline: August 2, 2026)

High-risk categories include: employment/recruitment, credit scoring, insurance, law enforcement, education, immigration, judicial decision-making.

### Article 10: Data and Data Governance

| Requirement | What It Means for Testing |
|-------------|--------------------------|
| Art. 10(2)(f) | MUST identify, detect, prevent, and mitigate harmful biases |
| Art. 10(3) | Training/validation/test data must be relevant, representative, free of errors |
| Art. 10(5) | MAY process special category data (race, gender) specifically to detect bias |
| Art. 9 | Risk management system must address bias as part of lifecycle |

### What Testing Tools Must Provide

1. **Disaggregated performance metrics** -- accuracy/F1/etc. broken down by demographic group
2. **Bias detection** -- at minimum: selection rate disparity, error rate disparity
3. **Documentation** -- audit trail showing bias was tested and results recorded
4. **Ongoing monitoring** -- not just one-time testing, but continuous in production
5. **Data representativeness checks** -- validate training data covers relevant demographics

### mltk Opportunity
Giskard (5.2K stars) was the leading open-source bias testing tool but is in transition (v2 deprecated, v3 unreleased). mltk can capture this gap with stable, well-documented fairness assertions. An EU AI Act compliance report template is already on the backlog.

---

## 8. Israel Anti-Discrimination & AI Requirements

### Current State (March 2026)
- **No AI-specific legislation** -- Israel uses a sectoral, principles-based approach
- Equal Opportunities in Employment Law (1988) prohibits discrimination in hiring, terms, promotions, termination
- Burden of proof shifts to employer once claimant shows prima facie discrimination
- Six core AI ethics principles adopted: human-centricity, **equality and non-discrimination**, transparency, reliability, accountability, privacy

### Emerging Regulation
- Bank of Israel directives on AI in lending expected late 2025/early 2026, will serve as blueprint for employment sector
- Ministry of Justice discussing a potential "Framework Law" for AI covering algorithmic discrimination
- Privacy Protection Authority (PPA) conducting proactive compliance audits of high-risk AI systems
- Enforcement trend: shifting from reactive to proactive

### Implications for mltk
- Israel does not mandate specific fairness metrics, but the non-discrimination principle applies
- Any mltk user deploying in Israel should test for demographic parity and disparate impact as baseline
- EU AI Act compliance covers Israeli companies exporting to the EU (which is most Israeli tech companies)

---

## Sprint 4 Recommendation: Which Metrics to Implement

### Must-Have (Sprint 4 Core)

These three metrics are explicitly listed in the backlog for `assert_no_bias`:

| Metric | Function Name | Rationale |
|--------|--------------|-----------|
| **Demographic Parity** | `demographic_parity_diff`, `demographic_parity_ratio` | Most common fairness metric, simple, no labels needed for ratio check |
| **Equalized Odds** | `equalized_odds_diff`, `equalized_odds_ratio` | Industry standard for error-rate fairness, required by Fairlearn/AIF360 |
| **Predictive Parity** | `predictive_parity_diff` | Completes the "big three" -- covers precision fairness |

### Should-Have (Sprint 4 Stretch)

| Metric | Function Name | Rationale |
|--------|--------------|-----------|
| **Disparate Impact Ratio** | `disparate_impact_ratio` | Legal standard (80% rule), trivial to compute, high value |
| **Equal Opportunity** | `equal_opportunity_diff` | Relaxed equalized odds (TPR only), common in practice |

### Defer to Sprint 9 (Monitoring/Tabular)

| Metric | Rationale |
|--------|-----------|
| Calibration across groups | Requires probability outputs, more complex, fits monitoring sprint |
| Conditional use accuracy equality | Niche, rarely tested |
| Individual fairness (Lipschitz) | Requires distance metric, very different paradigm |

### Proposed API Design for `assert_no_bias`

```python
from mltk.model import assert_no_bias

# Minimal API -- checks demographic parity by default
assert_no_bias(
    y_pred=predictions,
    sensitive_feature=gender_column,
)

# Full API with metric selection
assert_no_bias(
    y_true=labels,
    y_pred=predictions,
    sensitive_feature=race_column,
    method="equalized_odds",   # "demographic_parity" | "equalized_odds" | "predictive_parity" | "disparate_impact"
    threshold=0.1,             # max allowed difference (or min ratio for disparate_impact)
)

# Multi-metric check
assert_no_bias(
    y_true=labels,
    y_pred=predictions,
    sensitive_feature=gender_column,
    method=["demographic_parity", "equalized_odds"],
    threshold={"demographic_parity": 0.1, "equalized_odds": 0.1},
)
```

### Rust Acceleration Candidates for Sprint 4

| Function | Rationale |
|----------|-----------|
| `demographic_parity_diff` | Simple group-by aggregation, benefits from Rust at scale |
| `equalized_odds_diff` | Confusion matrix per group, vectorizable |
| `disparate_impact_ratio` | Trivial computation but included for API completeness |

### Default Thresholds

```python
BIAS_THRESHOLDS = {
    "demographic_parity": 0.10,    # max difference
    "equalized_odds": 0.10,        # max difference
    "predictive_parity": 0.10,     # max PPV difference
    "disparate_impact": 0.80,      # min ratio (four-fifths rule)
    "equal_opportunity": 0.10,     # max TPR difference
}
```

---

## Sources

- [Fairlearn Common Fairness Metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)
- [Google ML Crash Course: Demographic Parity](https://developers.google.com/machine-learning/crash-course/fairness/demographic-parity)
- [Google ML Crash Course: Equality of Opportunity](https://developers.google.com/machine-learning/crash-course/fairness/equality-of-opportunity)
- [Fairlearn GitHub (v0.13.0)](https://github.com/fairlearn/fairlearn)
- [GeeksforGeeks: Fairness Metrics](https://www.geeksforgeeks.org/artificial-intelligence/fairness-metrics-demographic-parity-equalized-odds/)
- [MIT OCW: Fairness Criteria](https://ocw.mit.edu/courses/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/pages/module-three-framework/fairness-criteria/)
- [Wikipedia: Fairness (machine learning)](https://en.wikipedia.org/wiki/Fairness_(machine_learning))
- [Superwise: ML Fairness Metrics](https://superwise.ai/blog/gentle-introduction-ml-fairness-metrics/)
- [Number Analytics: Equalized Odds](https://www.numberanalytics.com/blog/equalized-odds-fairness-metric-machine-learning)
- [Insightful Data Lab: Predictive Parity](https://insightful-data-lab.com/2025/08/29/predictive-parity-calibration/)
- [FAccT 2024: The Four-Fifths Rule Is Not Disparate Impact](https://facctconference.org/static/papers24/facct24-53.pdf)
- [Giskard: 80% Rule](https://www.giskard.ai/knowledge/how-to-test-ml-models-5-the-80-rule-to-measure-disparity)
- [EEOC Guidance on AI and Disparate Impact](https://ogletree.com/insights-resources/blog-posts/eeoc-issues-new-guidance-on-employer-use-of-ai-and-disparate-impact-potential/)
- [EU AI Act Article 10](https://artificialintelligenceact.eu/article/10/)
- [EU AI Act 2026 Compliance Guide](https://secureprivacy.ai/blog/eu-ai-act-2026-compliance)
- [Scrut: Fairness and Bias Mitigation in AI](https://www.scrut.io/glossary/fairness-and-bias-mitigation)
- [Israel AI Regulation Overview](https://regulations.ai/regulations/RAI-IL-NA-SUMMARY-2026)
- [White & Case: AI Watch Israel](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-israel)
- [Greenberg Traurig: 2026 AI Outlook Israel](https://www.gtlaw-israelpractice.com/2025/12/24/as-2025-draws-to-a-close-ai-regulation-continues-to-accelerate-across-the-globe-states-in-the-u-s-and-regions-like-the-eu-have-been-particularly-active-creating-a-complex-landscape-for-businesses-and/)
- [Herzog: Israel Labour Law Year in Review 2025](https://herzoglaw.co.il/en/news-and-insights/labour-law-year-in-review-2025-looking-ahead-to-2026/)
- [dida blog: Fairness in Machine Learning](https://dida.do/blog/fairness-in-ml)
