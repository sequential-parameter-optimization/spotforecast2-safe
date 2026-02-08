# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pytest-based tests for docstring examples in the manager module.

This test suite validates that all code examples in the docstrings
of the manager module execute correctly without errors.
"""

import doctest
import sys


def test_docstring_examples_logger():
    """Test all docstring examples in manager.logger module."""
    import spotforecast2_safe.manager.logger as logger_module
    
    results = doctest.testmod(
        logger_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_tools():
    """Test all docstring examples in manager.tools module."""
    import spotforecast2_safe.manager.tools as tools_module
    
    results = doctest.testmod(
        tools_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_persistence():
    """Test all docstring examples in manager.persistence module."""
    import spotforecast2_safe.manager.persistence as persistence_module
    
    results = doctest.testmod(
        persistence_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_predictor():
    """Test all docstring examples in manager.predictor module."""
    import spotforecast2_safe.manager.predictor as predictor_module
    
    results = doctest.testmod(
        predictor_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_trainer():
    """Test all docstring examples in manager.trainer module."""
    import spotforecast2_safe.manager.trainer as trainer_module
    
    results = doctest.testmod(
        trainer_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


def test_docstring_examples_metrics():
    """Test all docstring examples in manager.metrics module."""
    import spotforecast2_safe.manager.metrics as metrics_module
    
    results = doctest.testmod(
        metrics_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


if __name__ == "__main__":
    # Run all tests when executed directly
    print("=" * 70)
    print("Testing manager.logger docstring examples...")
    print("=" * 70)
    test_docstring_examples_logger()
    
    print("\n" + "=" * 70)
    print("Testing manager.tools docstring examples...")
    print("=" * 70)
    test_docstring_examples_tools()
    
    print("\n" + "=" * 70)
    print("Testing manager.persistence docstring examples...")
    print("=" * 70)
    test_docstring_examples_persistence()
    
    print("\n" + "=" * 70)
    print("Testing manager.predictor docstring examples...")
    print("=" * 70)
    test_docstring_examples_predictor()
    
    print("\n" + "=" * 70)
    print("Testing manager.trainer docstring examples...")
    print("=" * 70)
    test_docstring_examples_trainer()
    
    print("\n" + "=" * 70)
    print("Testing manager.metrics docstring examples...")
    print("=" * 70)
    test_docstring_examples_metrics()
    
    print("\n" + "=" * 70)
    print("All docstring example tests passed!")
    print("=" * 70)
