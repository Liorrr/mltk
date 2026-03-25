"""Plugin system — register and discover third-party assertion functions.

Third-party packages can extend mltk by:
1. Installing a package named ``mltk_plugin_*``.
2. Decorating assertion functions with ``@register_assertion``.

mltk will auto-discover installed plugins via ``discover_plugins()``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable

# Module-level registry: name -> callable
_ASSERTION_REGISTRY: dict[str, Callable] = {}


def register_assertion(name: str | None = None) -> Callable:
    """Decorator that registers a custom assertion function.

    The decorated function is stored in the global registry under ``name``
    (or the function's own ``__name__`` if ``name`` is omitted), then
    returned unchanged so it remains directly callable.

    Usage — explicit name::

        @register_assertion("my_custom_check")
        def assert_my_check(data, threshold=0.5):
            ...

    Usage — inferred name::

        @register_assertion()
        def assert_my_check(data):
            ...

    Args:
        name: Registry key for the assertion. Defaults to the function's
              ``__name__``.

    Returns:
        Decorator that registers and returns the wrapped function.
    """

    def decorator(fn: Callable) -> Callable:
        registry_key = name if name is not None else fn.__name__
        _ASSERTION_REGISTRY[registry_key] = fn
        return fn

    # Support both @register_assertion and @register_assertion("name")
    # If called as @register_assertion (no parens, name is the function itself),
    # handle that edge case gracefully.
    if callable(name):
        # Called without parentheses: @register_assertion
        fn = name  # type: ignore[assignment]
        _ASSERTION_REGISTRY[fn.__name__] = fn  # type: ignore[union-attr]
        return fn  # type: ignore[return-value]

    return decorator


def get_registered_assertions() -> dict[str, Callable]:
    """Return a copy of all registered assertion functions.

    Returns:
        Dict mapping assertion name to callable.
    """
    return dict(_ASSERTION_REGISTRY)


def discover_plugins(package_prefix: str = "mltk_plugin_") -> list[str]:
    """Discover and import installed mltk plugin packages.

    Scans all installed Python packages for names starting with
    ``package_prefix`` (default: ``"mltk_plugin_"``). Each matching package
    is imported, which triggers any ``@register_assertion`` decorators at
    module level.

    Args:
        package_prefix: Prefix that identifies mltk plugin packages.

    Returns:
        List of discovered (and successfully imported) package names.

    Example:
        >>> discovered = discover_plugins()
        >>> print(discovered)
        ['mltk_plugin_finance', 'mltk_plugin_cv']
    """
    discovered: list[str] = []

    try:
        packages = importlib.metadata.packages_distributions()
    except Exception:
        return discovered

    seen: set[str] = set()
    for dist_name in packages.values():
        for pkg in dist_name:
            if pkg not in seen and pkg.startswith(package_prefix):
                seen.add(pkg)
                try:
                    importlib.import_module(pkg)
                    discovered.append(pkg)
                except ImportError:
                    pass

    return discovered
