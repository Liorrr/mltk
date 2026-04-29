"""mltk — pytest for ML. Unified testing across the entire ML lifecycle."""

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__ = _pkg_version("mltk")
except PackageNotFoundError:
    __version__ = "0.12.5"  # fallback for uninstalled source tree

# Convenience imports for the most common assertions
from mltk.data import (
    assert_dtypes,
    assert_freshness,
    assert_no_drift,
    assert_no_nulls,
    assert_no_pii,
    assert_range,
    assert_row_count,
    assert_schema,
    assert_unique,
)
from mltk.model import (
    assert_calibration,
    assert_metric,
    assert_intersectional_fairness,
    assert_no_bias,
    assert_no_regression,
    assert_slice_performance,
)
from mltk.container import (
    assert_container_vulnerabilities,
    assert_no_secrets_in_image,
)

__all__ = [
    "__version__",
    # Data
    "assert_schema",
    "assert_no_nulls",
    "assert_dtypes",
    "assert_range",
    "assert_unique",
    "assert_freshness",
    "assert_row_count",
    "assert_no_drift",
    "assert_no_pii",
    # Model
    "assert_metric",
    "assert_no_regression",
    "assert_slice_performance",
    "assert_calibration",
    "assert_intersectional_fairness",
    "assert_no_bias",
    # Container
    "assert_container_vulnerabilities",
    "assert_no_secrets_in_image",
]
