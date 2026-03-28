"""Polars DataFrame bridge -- transparent conversion for mltk assertions.

mltk assertions accept pandas DataFrames and Series. Teams using Polars
(the fastest-growing DataFrame library) can't use mltk without manual
conversion. This bridge does it automatically.

Three tools:
- to_pandas(): convert a single value if it's Polars, pass through otherwise
- is_polars(): check without importing polars
- coerce_dataframe(): decorator that converts all Polars args before calling
"""

from __future__ import annotations

import functools
from typing import Any


def is_polars(data: Any) -> bool:
    """Check if data is a Polars DataFrame or Series without importing polars.

    Uses the module name from the object's type to avoid importing polars
    when it isn't installed. This keeps the check zero-cost for pandas-only
    users.

    Args:
        data: Any Python object.

    Returns:
        True if data is a polars DataFrame or Series, False otherwise.
    """
    if data is None:
        return False
    type_module = type(data).__module__ or ""
    type_name = type(data).__name__
    if not type_module.startswith("polars"):
        return False
    return type_name in ("DataFrame", "Series", "LazyFrame")


def to_pandas(data: Any) -> Any:
    """Auto-convert Polars DataFrames/Series to pandas equivalents.

    WHY: mltk assertions accept pandas DataFrames. Teams using Polars
    (the fastest-growing DataFrame library) can't use mltk without
    manual conversion. This bridge does it automatically.

    If data is already pandas, returns as-is (zero overhead).
    If data is Polars, converts via .to_pandas().
    If data is neither, returns as-is (let downstream handle it).

    Args:
        data: Any Python object. Polars DataFrames/Series are converted;
            everything else passes through unchanged.

    Returns:
        pandas DataFrame/Series if input was Polars, otherwise the
        original object unchanged.

    Raises:
        ImportError: If data is Polars but pandas is not installed
            (unlikely in mltk, but polars.to_pandas() requires it).
    """
    if data is None:
        return data
    if is_polars(data):
        return data.to_pandas()
    return data


def coerce_dataframe(func):  # type: ignore[no-untyped-def]
    """Decorator that auto-converts Polars args to pandas before calling.

    Inspects function arguments. Any that are Polars DataFrame/Series
    get converted to pandas via to_pandas(). This lets existing assertions
    work with Polars data transparently.

    Only converts positional and keyword arguments -- does not touch
    *args or **kwargs that aren't in the function signature.

    Args:
        func: Any callable whose arguments may include Polars objects.

    Returns:
        Wrapped function that converts Polars inputs to pandas first.

    Example:
        >>> @coerce_dataframe
        ... def check(df):
        ...     return df.columns.tolist()
        >>> check(polars_df)  # receives pandas inside
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        converted_args = [to_pandas(arg) for arg in args]
        converted_kwargs = {
            key: to_pandas(value) for key, value in kwargs.items()
        }
        return func(*converted_args, **converted_kwargs)

    return wrapper
