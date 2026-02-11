# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from spotforecast2_safe.model_selection import TimeSeriesFold


def test_repr_example():
    """
    Test the example from TimeSeriesFold.__repr__ docstring.
    """
    cv = TimeSeriesFold(steps=1)
    repr_str = repr(cv)

    # Basic assertions to ensure repr contains key information
    assert "TimeSeriesFold" in repr_str
    assert "Steps                 = 1" in repr_str
    assert "Initial train size    = None" in repr_str
