# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the data layer's fail-safe missing-value contract.

``data.fetch_data._apply_on_missing`` is the central promise of the data layer:
by default it refuses to return a series that silently embeds imputed values.
Testing it directly (it is a pure function) exercises every branch without I/O.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.data.fetch_data import _apply_on_missing


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, name="load")


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="on_missing must be"):
        _apply_on_missing(_series([1.0, 2.0]), "bogus", "load", Path("x.csv"))


def test_raise_on_nan_lists_gaps():
    y = _series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="missing value"):
        _apply_on_missing(y, "raise", "load", Path("x.csv"))


def test_raise_without_nan_returns_unchanged():
    y = _series([1.0, 2.0, 3.0])
    out = _apply_on_missing(y, "raise", "load", Path("x.csv"))
    assert out.equals(y)


def test_passthrough_keeps_nan():
    y = _series([1.0, np.nan])
    out = _apply_on_missing(y, "passthrough", "load", Path("x.csv"))
    assert out.isna().any()


def test_ffill_bfill_repairs_nan():
    y = _series([np.nan, 2.0, np.nan])
    out = _apply_on_missing(y, "ffill_bfill", "load", Path("x.csv"))
    assert not out.isna().any()
