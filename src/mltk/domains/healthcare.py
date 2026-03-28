"""Healthcare ML evaluation -- clinical-grade diagnostic quality metrics.

Healthcare ML models fail in ways that generic accuracy metrics miss entirely.
A cancer screening model with 99% accuracy on a dataset where 99% of patients
are healthy is useless -- it could achieve that score by always predicting
"healthy" and missing every single cancer case.

This module provides five assertions that evaluate diagnostic model quality
from complementary clinical angles:

- **Sensitivity** -- true positive rate: does the model catch sick patients?
- **Specificity** -- true negative rate: does the model avoid false alarms?
- **PPV** -- positive predictive value: when it says "positive," is it right?
- **NPV** -- negative predictive value: when it says "negative," is it right?
- **Clinical Agreement** -- Cohen's Kappa: agreement beyond random chance.

All computations are pure numpy (no external dependencies beyond numpy).
Each assertion follows the ``@timed_assertion`` pattern, returns a
``TestResult``, and integrates directly into pytest pipelines.

Clinical context matters:
    - A screening test (e.g., mammography) needs HIGH sensitivity -- missing
      a cancer is worse than a false alarm.
    - A confirmatory test (e.g., biopsy) needs HIGH specificity -- you do
      not want unnecessary surgery.
    - In rare diseases (prevalence < 1%), even 99% specificity gives a PPV
      of only ~50%.  Always check PPV alongside sensitivity/specificity.
    - Cohen's Kappa corrects for the base-rate illusion that inflates
      raw accuracy on imbalanced datasets.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compute TP, TN, FP, FN from binary arrays.

    Positive class is 1, negative class is 0.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Tuple of (TP, TN, FP, FN) as integers.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def _validate_binary_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assertion_name: str,
    severity: Severity,
) -> TestResult | None:
    """Validate that inputs are non-empty binary arrays.

    Returns a failing TestResult if validation fails, or None if
    inputs are valid.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        assertion_name: Name for the assertion result.
        severity: Severity level.

    Returns:
        TestResult on validation failure, None on success.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return assert_true(
            False,
            name=assertion_name,
            message="Cannot compute metric on empty arrays",
            severity=severity,
        )

    if len(y_true) != len(y_pred):
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"Array length mismatch: "
                f"y_true={len(y_true)}, y_pred={len(y_pred)}"
            ),
            severity=severity,
        )

    unique_true = set(np.unique(y_true).tolist())
    unique_pred = set(np.unique(y_pred).tolist())
    all_values = unique_true | unique_pred

    if not all_values <= {0, 1}:
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"Non-binary values detected. "
                f"Expected {{0, 1}}, got {sorted(all_values)}"
            ),
            severity=severity,
        )

    return None


# ------------------------------------------------------------------
# Public assertions
# ------------------------------------------------------------------


@timed_assertion
def assert_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_sensitivity: float = 0.9,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that model sensitivity meets a minimum threshold.

    Sensitivity (true positive rate / recall) measures the fraction
    of actual positive cases that the model correctly identifies.
    In clinical terms: of all patients who ARE sick, how many does
    the model catch?

    A cancer screening model with 60% sensitivity misses 40% of
    cancers -- those patients walk away with a false sense of
    security and delayed treatment.  For screening applications,
    sensitivity thresholds of 90-95% are typical regulatory
    requirements.

    Formula:
        sensitivity = TP / (TP + FN)

    When TP + FN = 0 (no actual positives in the dataset), the
    metric is undefined and the assertion fails with a descriptive
    message.

    Args:
        y_true: Ground truth binary labels (0=negative, 1=positive).
        y_pred: Predicted binary labels (0=negative, 1=positive).
        min_sensitivity: Minimum acceptable sensitivity (default 0.9).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with ``sensitivity``, ``min_sensitivity``,
        ``tp``, ``fn``, and ``n_positive`` in details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1])
        >>> y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 1])
        >>> result = assert_sensitivity(y_true, y_pred, min_sensitivity=0.8)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    validation_err = _validate_binary_inputs(
        y_t, y_p, "domains.healthcare.sensitivity", severity,
    )
    if validation_err is not None:
        return validation_err

    tp, _tn, _fp, fn = _confusion_counts(y_t, y_p)
    n_positive = tp + fn

    if n_positive == 0:
        return assert_true(
            False,
            name="healthcare.sensitivity",
            message=(
                "Sensitivity undefined: no positive cases "
                "in y_true (TP + FN = 0)"
            ),
            severity=severity,
            sensitivity=float("nan"),
            min_sensitivity=min_sensitivity,
            tp=tp,
            fn=fn,
            n_positive=0,
        )

    sensitivity = tp / n_positive
    passed = sensitivity >= min_sensitivity

    message = (
        f"Sensitivity: {sensitivity:.4f} >= {min_sensitivity} "
        f"({tp}/{n_positive} positives detected)"
        if passed
        else (
            f"Low sensitivity: {sensitivity:.4f} "
            f"< {min_sensitivity} "
            f"({fn} of {n_positive} positives missed)"
        )
    )

    return assert_true(
        passed,
        name="healthcare.sensitivity",
        message=message,
        severity=severity,
        sensitivity=sensitivity,
        min_sensitivity=min_sensitivity,
        tp=tp,
        fn=fn,
        n_positive=n_positive,
    )


@timed_assertion
def assert_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_specificity: float = 0.9,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that model specificity meets a minimum threshold.

    Specificity (true negative rate) measures the fraction of
    actual negative cases that the model correctly identifies.
    In clinical terms: of all patients who are HEALTHY, how many
    does the model correctly clear?

    A model with 80% specificity sends 20% of healthy patients
    for unnecessary follow-up -- biopsies, imaging, anxiety, and
    wasted healthcare resources.  For confirmatory tests,
    specificity thresholds of 95%+ are common.

    Formula:
        specificity = TN / (TN + FP)

    When TN + FP = 0 (no actual negatives in the dataset), the
    metric is undefined and the assertion fails.

    Args:
        y_true: Ground truth binary labels (0=negative, 1=positive).
        y_pred: Predicted binary labels (0=negative, 1=positive).
        min_specificity: Minimum acceptable specificity (default 0.9).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with ``specificity``, ``min_specificity``,
        ``tn``, ``fp``, and ``n_negative`` in details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 0, 1, 1, 0, 0, 0])
        >>> y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        >>> result = assert_specificity(y_true, y_pred, min_specificity=0.8)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    validation_err = _validate_binary_inputs(
        y_t, y_p, "domains.healthcare.specificity", severity,
    )
    if validation_err is not None:
        return validation_err

    _tp, tn, fp, _fn = _confusion_counts(y_t, y_p)
    n_negative = tn + fp

    if n_negative == 0:
        return assert_true(
            False,
            name="healthcare.specificity",
            message=(
                "Specificity undefined: no negative cases "
                "in y_true (TN + FP = 0)"
            ),
            severity=severity,
            specificity=float("nan"),
            min_specificity=min_specificity,
            tn=tn,
            fp=fp,
            n_negative=0,
        )

    specificity = tn / n_negative
    passed = specificity >= min_specificity

    message = (
        f"Specificity: {specificity:.4f} >= {min_specificity} "
        f"({tn}/{n_negative} negatives correctly cleared)"
        if passed
        else (
            f"Low specificity: {specificity:.4f} "
            f"< {min_specificity} "
            f"({fp} of {n_negative} negatives falsely flagged)"
        )
    )

    return assert_true(
        passed,
        name="healthcare.specificity",
        message=message,
        severity=severity,
        specificity=specificity,
        min_specificity=min_specificity,
        tn=tn,
        fp=fp,
        n_negative=n_negative,
    )


@timed_assertion
def assert_ppv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_ppv: float = 0.8,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that positive predictive value meets a minimum threshold.

    PPV (positive predictive value / precision) answers: when the
    model says "positive," how often is it actually correct?

    This is the metric patients care about most.  If a test comes
    back positive, PPV tells them the probability they actually
    have the disease.  The base-rate trap makes this critical:
    for a rare disease with 1% prevalence, even a test with 99%
    sensitivity and 99% specificity has a PPV of only ~50%.  Half
    of "positive" results are false alarms.

    Formula:
        ppv = TP / (TP + FP)

    When TP + FP = 0 (model never predicts positive), the metric
    is undefined and the assertion fails.

    Args:
        y_true: Ground truth binary labels (0=negative, 1=positive).
        y_pred: Predicted binary labels (0=negative, 1=positive).
        min_ppv: Minimum acceptable PPV (default 0.8).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with ``ppv``, ``min_ppv``, ``tp``, ``fp``,
        and ``n_predicted_positive`` in details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1])
        >>> y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 1])
        >>> result = assert_ppv(y_true, y_pred, min_ppv=0.7)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    validation_err = _validate_binary_inputs(
        y_t, y_p, "domains.healthcare.ppv", severity,
    )
    if validation_err is not None:
        return validation_err

    tp, _tn, fp, _fn = _confusion_counts(y_t, y_p)
    n_predicted_positive = tp + fp

    if n_predicted_positive == 0:
        return assert_true(
            False,
            name="healthcare.ppv",
            message=(
                "PPV undefined: model never predicted positive "
                "(TP + FP = 0)"
            ),
            severity=severity,
            ppv=float("nan"),
            min_ppv=min_ppv,
            tp=tp,
            fp=fp,
            n_predicted_positive=0,
        )

    ppv = tp / n_predicted_positive
    passed = ppv >= min_ppv

    message = (
        f"PPV: {ppv:.4f} >= {min_ppv} "
        f"({tp}/{n_predicted_positive} positive predictions "
        f"correct)"
        if passed
        else (
            f"Low PPV: {ppv:.4f} < {min_ppv} "
            f"({fp} of {n_predicted_positive} positive "
            f"predictions were false alarms)"
        )
    )

    return assert_true(
        passed,
        name="healthcare.ppv",
        message=message,
        severity=severity,
        ppv=ppv,
        min_ppv=min_ppv,
        tp=tp,
        fp=fp,
        n_predicted_positive=n_predicted_positive,
    )


@timed_assertion
def assert_npv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_npv: float = 0.9,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that negative predictive value meets a minimum threshold.

    NPV (negative predictive value) answers: when the model says
    "negative," how often is it actually correct?

    This is critical for ruling-out tests.  If a patient gets a
    negative result, NPV tells them the probability they truly do
    NOT have the disease.  In emergency settings (e.g., ruling out
    heart attack in the ER), NPV must be extremely high -- sending
    a patient home with a missed MI is catastrophic.

    Formula:
        npv = TN / (TN + FN)

    When TN + FN = 0 (model never predicts negative), the metric
    is undefined and the assertion fails.

    Args:
        y_true: Ground truth binary labels (0=negative, 1=positive).
        y_pred: Predicted binary labels (0=negative, 1=positive).
        min_npv: Minimum acceptable NPV (default 0.9).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with ``npv``, ``min_npv``, ``tn``, ``fn``,
        and ``n_predicted_negative`` in details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
        >>> y_pred = np.array([0, 0, 0, 1, 0, 1, 0, 0])
        >>> result = assert_npv(y_true, y_pred, min_npv=0.8)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    validation_err = _validate_binary_inputs(
        y_t, y_p, "domains.healthcare.npv", severity,
    )
    if validation_err is not None:
        return validation_err

    _tp, tn, _fp, fn = _confusion_counts(y_t, y_p)
    n_predicted_negative = tn + fn

    if n_predicted_negative == 0:
        return assert_true(
            False,
            name="healthcare.npv",
            message=(
                "NPV undefined: model never predicted negative "
                "(TN + FN = 0)"
            ),
            severity=severity,
            npv=float("nan"),
            min_npv=min_npv,
            tn=tn,
            fn=fn,
            n_predicted_negative=0,
        )

    npv = tn / n_predicted_negative
    passed = npv >= min_npv

    message = (
        f"NPV: {npv:.4f} >= {min_npv} "
        f"({tn}/{n_predicted_negative} negative predictions "
        f"correct)"
        if passed
        else (
            f"Low NPV: {npv:.4f} < {min_npv} "
            f"({fn} of {n_predicted_negative} negative "
            f"predictions missed actual positives)"
        )
    )

    return assert_true(
        passed,
        name="healthcare.npv",
        message=message,
        severity=severity,
        npv=npv,
        min_npv=min_npv,
        tn=tn,
        fn=fn,
        n_predicted_negative=n_predicted_negative,
    )


@timed_assertion
def assert_clinical_agreement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_kappa: float = 0.6,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that clinical agreement (Cohen's Kappa) meets a threshold.

    Raw accuracy is misleading on imbalanced clinical datasets.  A
    model predicting "healthy" for every patient in a population
    with 95% healthy individuals achieves 95% accuracy but catches
    zero diseases.  Cohen's Kappa corrects for this by measuring
    agreement BEYOND what random chance would produce.

    Interpretation (Landis & Koch, 1977):
        - < 0.00: Less than chance agreement
        - 0.01-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement

    A model and a doctor agreeing 95% of the time might only yield
    Kappa = 0.2 if the base rate is 95% -- meaning the model adds
    almost nothing beyond always guessing the majority class.

    Formula:
        p_o = observed agreement = (TP + TN) / N
        p_e = expected agreement by chance
            = (TP+FN)(TP+FP)/N^2 + (TN+FP)(TN+FN)/N^2
        kappa = (p_o - p_e) / (1 - p_e)

    When p_e = 1.0 (degenerate case where chance explains all
    agreement), kappa is undefined and the assertion fails.

    Args:
        y_true: Ground truth binary labels (0=negative, 1=positive).
        y_pred: Predicted binary labels (0=negative, 1=positive).
        min_kappa: Minimum acceptable Cohen's Kappa (default 0.6).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with ``kappa``, ``min_kappa``, ``p_observed``,
        ``p_expected``, and ``n_samples`` in details.

    Example:
        >>> import numpy as np
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        >>> y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> result = assert_clinical_agreement(
        ...     y_true, y_pred, min_kappa=0.6
        ... )
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    validation_err = _validate_binary_inputs(
        y_t, y_p,
        "domains.healthcare.clinical_agreement",
        severity,
    )
    if validation_err is not None:
        return validation_err

    tp, tn, fp, fn = _confusion_counts(y_t, y_p)
    n = len(y_t)

    # Observed agreement
    p_observed = (tp + tn) / n

    # Expected agreement by chance
    # P(both say positive) + P(both say negative)
    p_yes = ((tp + fn) / n) * ((tp + fp) / n)
    p_no = ((tn + fp) / n) * ((tn + fn) / n)
    p_expected = p_yes + p_no

    if abs(1.0 - p_expected) < 1e-10:
        return assert_true(
            False,
            name="healthcare.clinical_agreement",
            message=(
                "Kappa undefined: expected agreement = 1.0 "
                "(degenerate case, all samples in one class "
                "and model predicts same class)"
            ),
            severity=severity,
            kappa=float("nan"),
            min_kappa=min_kappa,
            p_observed=p_observed,
            p_expected=p_expected,
            n_samples=n,
        )

    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    passed = kappa >= min_kappa

    message = (
        f"Cohen's Kappa: {kappa:.4f} >= {min_kappa} "
        f"(observed={p_observed:.4f}, "
        f"chance={p_expected:.4f}, n={n})"
        if passed
        else (
            f"Low agreement: Kappa={kappa:.4f} "
            f"< {min_kappa} "
            f"(observed={p_observed:.4f}, "
            f"chance={p_expected:.4f}, n={n})"
        )
    )

    return assert_true(
        passed,
        name="healthcare.clinical_agreement",
        message=message,
        severity=severity,
        kappa=kappa,
        min_kappa=min_kappa,
        p_observed=p_observed,
        p_expected=p_expected,
        n_samples=n,
    )
