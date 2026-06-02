# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pytest-based tests for docstring examples in the manager module.

This test suite validates that all code examples in the docstrings
of the manager module execute correctly without errors.
"""

import doctest


def test_docstring_examples_predictor():
    """Test all docstring examples in manager.predictor module."""
    import spotforecast2_safe.manager.predictor as predictor_module

    results = doctest.testmod(
        predictor_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )

    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_trainer():
    """Test all docstring examples in manager.trainer module."""
    import spotforecast2_safe.manager.trainer as trainer_module

    results = doctest.testmod(
        trainer_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )

    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_metrics():
    """Test all docstring examples in manager.demo_metrics module."""
    import spotforecast2_safe.manager.demo_metrics as metrics_module

    results = doctest.testmod(
        metrics_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )

    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


if __name__ == "__main__":
    # Run all tests when executed directly
    print("=" * 70)
    print("Testing manager.predictor docstring examples...")
    print("=" * 70)
    test_docstring_examples_predictor()

    print("\n" + "=" * 70)
    print("Testing manager.trainer docstring examples...")
    print("=" * 70)
    test_docstring_examples_trainer()

    print("\n" + "=" * 70)
    print("Testing manager.demo_metrics docstring examples...")
    print("=" * 70)
    test_docstring_examples_metrics()

    print("\n" + "=" * 70)
    print("All docstring example tests passed!")
    print("=" * 70)
