# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pytest tests for merge_build_manual docstring examples (doctest validation).
"""

from spotforecast2_safe.downloader import entsoe


def test_merge_build_manual_doctest():
    """Run doctest on merge_build_manual docstring examples."""
    import doctest

    failures, _ = doctest.testmod(
        entsoe, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    assert failures == 0
