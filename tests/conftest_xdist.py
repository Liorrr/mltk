"""pytest-xdist and pytest-randomly configuration for mltk.

Parallel execution
------------------
Install: pip install pytest-xdist

Run all tests across CPU cores with scope-based grouping::

    pytest -n auto --dist loadscope

The ``loadscope`` strategy keeps tests in the same module (or class)
on the same worker, which prevents fixture conflicts while still
distributing across all available cores.

To limit workers explicitly::

    pytest -n 4 --dist loadscope

Random ordering
---------------
Install: pip install pytest-randomly

Enable random test ordering to surface hidden inter-test dependencies::

    pytest -p randomly

Combine with a fixed seed for reproducible random order::

    pytest -p randomly --randomly-seed=12345

Print the seed used (shown in the header by default)::

    pytest -p randomly -v

Combined usage::

    pytest -n auto --dist loadscope -p randomly

Notes
-----
- Both packages are dev dependencies. Add them to ``pyproject.toml``
  under ``[project.optional-dependencies] dev``::

      dev = [
          "pytest>=8.0",
          "pytest-cov>=4.0",
          "pytest-xdist>=3.5",
          "pytest-randomly>=3.15",
          "ruff>=0.3",
          "mypy>=1.8",
      ]

- ``pytest-xdist`` does NOT share fixtures across workers.
  Each worker gets its own fixture instances.  Tests that
  rely on shared mutable state (files, databases) need
  session-scoped tmp directories or locks.

- ``pytest-randomly`` re-orders at collection time.  To
  disable for a single run without uninstalling::

      pytest -p no:randomly
"""
