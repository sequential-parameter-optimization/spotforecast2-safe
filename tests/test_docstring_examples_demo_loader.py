# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test suite for demo_loader docstring examples.

This module validates all examples in the demo_loader module documentation
to ensure they execute correctly and produce expected results.
"""

import doctest

from spotforecast2_safe.manager.datasets import demo_loader


def test_docstring_examples_demo_loader():
    """Test all docstring examples in demo_loader module."""
    results = doctest.testmod(demo_loader, verbose=False)
    assert results.failed == 0, f"Doctest failed: {results.failed} failures"
