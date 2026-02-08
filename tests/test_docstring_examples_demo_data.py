# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pytest-based tests for docstring examples in the datasets module.

This test suite validates that all code examples in the docstrings
of the datasets module execute correctly without errors.
"""

import doctest


def test_docstring_examples_demo_data():
    """Test all docstring examples in datasets.demo_data module."""
    import spotforecast2_safe.manager.datasets.demo_data as demo_data_module
    
    results = doctest.testmod(
        demo_data_module,
        verbose=True,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    
    assert results.failed == 0, f"Docstring examples failed: {results.failed} failures"


if __name__ == "__main__":
    print("=" * 70)
    print("Testing datasets.demo_data docstring examples...")
    print("=" * 70)
    test_docstring_examples_demo_data()
    
    print("\n" + "=" * 70)
    print("All docstring example tests passed!")
    print("=" * 70)
